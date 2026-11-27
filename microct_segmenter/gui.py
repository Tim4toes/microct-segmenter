"""A small napari dock with native file dialogs and background I/O."""
from __future__ import annotations

import shutil
import tempfile
import threading
import traceback
import weakref
from dataclasses import asdict
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import QEvent, QSettings, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressDialog,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from . import __version__
from .core import (
    Cancelled, Correction, LabelSource, ValidationError, combine_plane,
    correct_mask, export_masks, inspect_dataset, load_volume, read_project,
    restore_labels, save_project, validate_labels, validate_layer_transform,
)

NATIVE_PRESET = "Native pixels: no correction"
DEFAULT_PRESET = "Saved default"
CUSTOM_PRESET = "Custom"
DEFAULT_KEY = "mask_coverage/default"
PRESETS_KEY = "mask_coverage/presets"
LOCK_KEY = "mask_coverage/locked"


class Job(QThread):
    progressed = Signal(int, int, str)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function
        self.cancel = threading.Event()
        self.result = self.error = self.details = None

    def run(self):
        try:
            self.result = self.function(self.progressed.emit, self.cancel)
        except BaseException as exc:
            self.error, self.details = exc, traceback.format_exc()


class ZoomView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = self.scene().addPixmap(QPixmap())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(300, 300)

    def set_array(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.item.setPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self.item.boundingRect())

    def fit(self):
        self.fitInView(self.item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()


class Preview(QDialog):
    def __init__(self, owner, sources, settings):
        super().__init__(owner)
        self.owner, self.sources, self.settings = owner, sources, settings
        self.setWindowTitle("Export preview — red is ROI; wheel to zoom, drag to pan")
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        self.caption = QLabel()
        layout.addWidget(self.caption)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Slice index (0-based):"))
        self.index = QSpinBox()
        self.index.setRange(0, owner.dataset.shape[0] - 1)
        self.index.setValue(owner.current_z())
        self.index.setKeyboardTracking(False)
        controls.addWidget(self.index)
        fit = QPushButton("Fit both")
        native = QPushButton("1 image pixel = 1 screen pixel")
        controls.addWidget(fit)
        controls.addWidget(native)
        layout.addLayout(controls)
        row = QHBoxLayout()
        self.before, self.after = ZoomView(), ZoomView()
        for title, view in [("Editable segmentation", self.before), ("Mask that will be exported", self.after)]:
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            column.addWidget(view)
            row.addLayout(column)
        layout.addLayout(row)
        details = QLabel(f"Shift: X {settings.x_shift:+d}, Y {settings.y_shift:+d} px; "
                         f"2D square dilation: {settings.dilation} px. No Z shift or resizing.\n"
                         "The preview combines the checked layers/IDs. Separate exports use the same corrections.")
        details.setWordWrap(True)
        layout.addWidget(details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.index.valueChanged.connect(self.render)
        fit.clicked.connect(lambda: [v.fit() for v in (self.before, self.after)])
        native.clicked.connect(lambda: [v.resetTransform() for v in (self.before, self.after)])
        self.render()
        QTimer.singleShot(0, lambda: [v.fit() for v in (self.before, self.after)])

    def render(self, *_):
        z = self.index.value()
        raw = np.asarray(self.owner.raw.data[z], dtype=np.float32)
        lo, hi = self.owner.raw.contrast_limits
        grey = np.clip((raw - lo) * 255 / max(hi - lo, 1e-12), 0, 255).astype(np.uint8)
        before = combine_plane(self.sources, z)
        after = correct_mask(before, self.settings)
        for view, mask in [(self.before, before), (self.after, after)]:
            rgb = np.repeat(grey[..., None], 3, axis=2)
            rgb[mask] = (rgb[mask].astype(np.float32) * 0.55 + np.array([255, 60, 60]) * 0.45).astype(np.uint8)
            view.set_array(rgb)
        self.caption.setText(f"{self.owner.dataset.slices[z].name}  |  "
                             f"{raw.shape[1]} × {raw.shape[0]} px  |  "
                             f"ROI pixels: {int(before.sum()):,} → {int(after.sum()):,}")


class MicroCTWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._main_window = self.viewer.window._qt_window
        self.settings_store = QSettings("MicroCT_Optimisation", "MicroCT_Segmenter")
        self.dataset = self.raw = self.worker = self.nninteractive = None
        self.cache_dirs = []
        self.saved_project = None
        self.dirty = False
        self.allow_close = False
        self._watched = weakref.WeakSet()
        self._items = []
        self._restoring = False
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._layers_changed)
        self.viewer.layers.events.removed.connect(self._layers_changed)
        self.viewer.layers.events.reordered.connect(self._layers_changed)
        # Qt has no public napari close-event hook. This is the sole private API use.
        self._main_window.installEventFilter(self)
        self._layers_changed()

    def _button(self, layout, title, slot):
        button = QPushButton(title)
        button.clicked.connect(lambda checked=False: self._guard(slot))
        layout.addWidget(button)
        return button

    def _build_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        body = QWidget()
        layout = QVBoxLayout(body)
        title = QLabel(f"MicroCT Segmenter  {__version__}")
        title.setStyleSheet("font-size: 17px; font-weight: bold;")
        layout.addWidget(title)
        row = QHBoxLayout()
        self._button(row, "Open dataset folder", self.open_folder)
        self._button(row, "Choose one slice", self.open_slice)
        layout.addLayout(row)
        self._button(layout, "Reopen saved project", self.open_project)
        self.disk_cache = QCheckBox("Use disk cache for large datasets")
        self.disk_cache.setChecked(True)
        self.disk_cache.setToolTip("Cache the full-size source image on disk. nnInteractive allocates its own RAM and GPU memory.")
        layout.addWidget(self.disk_cache)
        self._button(layout, "Choose cache location…", self.choose_cache)
        self.info = QLabel("Choose the folder containing the original image slices.")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info)

        group = QGroupBox("Segment and choose bones")
        group_layout = QVBoxLayout(group)
        self._button(group_layout, "Open nnInteractive", self.open_nninteractive)
        hint = QLabel("Select the original image in nnInteractive and initialise it to use points or scribbles. Check the segmentation results to export.")
        hint.setWordWrap(True)
        group_layout.addWidget(hint)
        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(100)
        self.layer_list.setMaximumHeight(150)
        group_layout.addWidget(self.layer_list)
        self._button(group_layout, "Refresh layer list", self._layers_changed)
        form = QFormLayout()
        self.ids = QLineEdit()
        self.ids.setPlaceholderText("Blank = all positive IDs; e.g. 1,3")
        form.addRow("Label IDs:", self.ids)
        self.mode = QComboBox()
        self.mode.addItem("Combine checked layers", "combined")
        self.mode.addItem("Separate folder for each layer", "layers")
        self.mode.addItem("Separate folder for each label ID", "ids")
        self.mode.installEventFilter(self)
        form.addRow("Export:", self.mode)
        group_layout.addLayout(form)
        layout.addWidget(group)

        group = QGroupBox("Mask coverage")
        form = QFormLayout(group)
        self.preset = QComboBox()
        form.addRow("Preset:", self.preset)
        self.dx, self.dy, self.dilation = QSpinBox(), QSpinBox(), QSpinBox()
        self.dx.setRange(-100, 100)
        self.dy.setRange(-100, 100)
        self.dilation.setRange(0, 100)
        form.addRow("X shift (+ right):", self.dx)
        form.addRow("Y shift (+ down):", self.dy)
        form.addRow("2D dilation (px):", self.dilation)
        self.depth = QComboBox()
        self.depth.addItem("1-bit BMP (CTAn monochrome ROI)", 1)
        self.depth.addItem("8-bit BMP (0 and 255)", 8)
        form.addRow("BMP format:", self.depth)
        self._coverage_controls = (self.preset, self.dx, self.dy, self.dilation, self.depth)
        for control in self._coverage_controls:
            control.installEventFilter(self)
        self.coverage_lock = QCheckBox("Lock mask coverage")
        self.coverage_lock.setToolTip("Prevent changes to the preset, shifts, dilation and BMP format.")
        self.coverage_lock.toggled.connect(self._set_coverage_locked)
        self.coverage_lock.setChecked(self.settings_store.value(LOCK_KEY, False, type=bool))
        form.addRow(self.coverage_lock)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        for spin in (self.dx, self.dy, self.dilation):
            spin.valueChanged.connect(self._custom_preset)
        self.depth.currentIndexChanged.connect(self._custom_preset)
        actions = QHBoxLayout()
        self._button(actions, "Save as default", self.save_coverage_default)
        self._button(actions, "Save preset…", self.save_coverage_preset)
        form.addRow(actions)
        self._reload_coverage_presets()
        layout.addWidget(group)
        self._button(layout, "Preview current slice / alignment", self.preview)
        self._button(layout, "Save editable project", self.save)
        self._button(layout, "Export CTAn masks", self.export)
        self.destination = QLabel("Masks and projects will be saved beside the source dataset folder.")
        self.destination.setWordWrap(True)
        self.destination.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.destination)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._button(layout, "Open last output folder", self.show_output)
        self.last_output = None
        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def _guard(self, action):
        if self.worker is not None:
            return
        try:
            action()
        except Exception as exc:
            QMessageBox.critical(self, "MicroCT Segmenter", str(exc))

    def _preset_changed(self, index):
        if index >= 0:
            settings = self.preset.itemData(index)
            if settings is not None:
                self._apply_coverage(settings)

    def _set_coverage_locked(self, locked):
        for control in self._coverage_controls:
            control.setDisabled(locked)
        self.settings_store.setValue(LOCK_KEY, locked)

    def _custom_preset(self, *_):
        self.preset.setCurrentText(CUSTOM_PRESET)

    def correction(self):
        return Correction(self.dx.value(), self.dy.value(), self.dilation.value(), self.depth.currentData())

    def _apply_coverage(self, settings):
        for spin, value in zip((self.dx, self.dy, self.dilation),
                               (settings.x_shift, settings.y_shift, settings.dilation)):
            with QSignalBlocker(spin):
                spin.setValue(value)
        with QSignalBlocker(self.depth):
            self.depth.setCurrentIndex(self.depth.findData(settings.bit_depth))

    def _stored_coverage(self, values):
        try:
            settings = Correction(values["x_shift"], values["y_shift"], values["dilation"], values["bit_depth"])
            settings.validate()
        except (KeyError, TypeError, ValidationError):
            return None
        if abs(settings.x_shift) <= self.dx.maximum() and abs(settings.y_shift) <= self.dy.maximum():
            return settings
        return None

    def _named_coverage(self):
        values = self.settings_store.value(PRESETS_KEY)
        return values if isinstance(values, dict) else {}

    def _reload_coverage_presets(self, selected=None):
        default = self._stored_coverage(self.settings_store.value(DEFAULT_KEY))
        with QSignalBlocker(self.preset):
            self.preset.clear()
            if default is not None:
                self.preset.addItem(DEFAULT_PRESET, default)
            self.preset.addItem(NATIVE_PRESET, Correction(y_shift=0, dilation=0))
            for name, values in sorted(self._named_coverage().items(), key=lambda item: item[0].casefold()):
                settings = self._stored_coverage(values)
                if settings is not None:
                    self.preset.addItem(name, settings)
            self.preset.addItem(CUSTOM_PRESET)
            self.preset.setCurrentText(selected or (DEFAULT_PRESET if default is not None else NATIVE_PRESET))
        self._preset_changed(self.preset.currentIndex())

    def save_coverage_default(self):
        self.settings_store.setValue(DEFAULT_KEY, asdict(self.correction()))
        self.settings_store.sync()
        self._reload_coverage_presets(DEFAULT_PRESET)
        self.status.setText("Mask coverage default saved for new datasets.")

    def save_coverage_preset(self):
        name, accepted = QInputDialog.getText(self, "Save mask coverage preset", "Preset name:")
        name = name.strip()
        if not accepted or not name:
            return
        reserved = {label.casefold() for label in (NATIVE_PRESET, DEFAULT_PRESET, CUSTOM_PRESET)}
        if name.casefold() in reserved:
            raise ValidationError("Choose a different preset name. This name is reserved.")
        presets = self._named_coverage()
        existing = next((label for label in presets if label.casefold() == name.casefold()), None)
        if existing is not None:
            choice = QMessageBox.question(self, "Replace mask coverage preset", f'Replace the preset "{existing}"?',
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if choice != QMessageBox.StandardButton.Yes:
                return
            name = existing
        presets[name] = asdict(self.correction())
        self.settings_store.setValue(PRESETS_KEY, presets)
        self.settings_store.sync()
        self._reload_coverage_presets(name)
        self.status.setText(f'Mask coverage preset "{name}" saved.')

    def _restore_settings(self, metadata):
        settings = Correction(**metadata["settings"])
        settings.validate()
        self._apply_coverage(settings)
        self._custom_preset()
        self.mode.setCurrentIndex(self.mode.findData(metadata.get("mode", "combined")))

    def _layers_changed(self, *_):
        old = {id(layer): self.layer_list.item(i).checkState() for i, layer in enumerate(self._items)}
        self.layer_list.clear()
        # The plugin implements its scribble input as a Labels subclass.
        # Its segmentation results use ordinary Labels, even when users rename them.
        self._items = [layer for layer in self.viewer.layers
                       if isinstance(layer, napari.layers.Labels)
                       and not any(cls.__module__ == "napari_nninteractive.layers.scribble_layer"
                                   for cls in type(layer).__mro__)]
        for layer in self._items:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(old.get(id(layer), Qt.CheckState.Checked))
            self.layer_list.addItem(item)
            if layer not in self._watched:
                layer.events.data.connect(self._mark_dirty)
                if hasattr(layer.events, "paint"):
                    layer.events.paint.connect(self._mark_dirty)
                layer.events.name.connect(self._layers_changed)
                self._watched.add(layer)
                if self.dataset and not self._restoring:
                    self.dirty = True

    def _mark_dirty(self, *_):
        self.dirty = True

    def _require_dataset(self):
        if self.dataset is None or self.raw not in self.viewer.layers:
            raise ValidationError("Open the source dataset using the MicroCT panel first.")
        if tuple(self.raw.data.shape) != self.dataset.shape:
            raise ValidationError("The original image layer dimensions have changed. Reopen the source dataset.")

    def sources(self, all_layers=False):
        self._require_dataset()
        ids = None
        if not all_layers and self.ids.text().strip():
            try:
                ids = tuple(sorted({int(t.strip()) for t in self.ids.text().split(",")}))
            except ValueError as exc:
                raise ValidationError("Use comma-separated positive label IDs, such as 1,3.") from exc
        self._layers_changed()
        selected = []
        for i, layer in enumerate(self._items):
            if all_layers or self.layer_list.item(i).checkState() == Qt.CheckState.Checked:
                validate_layer_transform(self.raw, layer)
                selected.append(LabelSource(layer.name, layer.data, ids))
        validate_labels(self.dataset, selected)
        return selected

    def _run_job(self, title, function, success):
        if self.worker is not None:
            return
        self.worker = Job(function, self)
        job = self.worker
        dialog = QProgressDialog(title, "Cancel", 0, 1000, self._main_window)
        dialog.setWindowTitle("MicroCT Segmenter")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(job.cancel.set)
        dialog.setValue(0)
        dialog.show()
        # All interactive edits are disabled while reading layer arrays.
        self._main_window.setEnabled(False)
        dialog.setEnabled(True)
        self.status.setText(title)

        def progress(done, total, message):
            if job.cancel.is_set():
                dialog.setLabelText("Cancelling after the current slice…")
            else:
                dialog.setLabelText(message)
                dialog.setValue(min(999, round(1000 * done / max(total, 1))))

        def finished():
            dialog.close()
            dialog.deleteLater()
            self._main_window.setEnabled(True)
            self.worker = None
            if job.error:
                self.status.setText(str(job.error))
                if not isinstance(job.error, Cancelled):
                    box = QMessageBox(QMessageBox.Icon.Critical, "MicroCT Segmenter", str(job.error), parent=self)
                    box.setDetailedText(job.details)
                    box.exec()
            else:
                try:
                    success(job.result)
                except Exception as exc:
                    QMessageBox.critical(self, "MicroCT Segmenter", str(exc))
            job.deleteLater()

        job.progressed.connect(progress)
        job.finished.connect(finished)
        job.start()

    def choose_cache(self):
        chosen = QFileDialog.getExistingDirectory(self, "Cache location (fast drive with enough free space)",
                                                  str(self.settings_store.value("cache_base", tempfile.gettempdir())))
        if chosen:
            self.settings_store.setValue("cache_base", chosen)
            self.status.setText(f"Cache location: {chosen}")

    def _cache(self):
        base = Path(str(self.settings_store.value("cache_base", tempfile.gettempdir())))
        base.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="microct_session_", dir=base))
        self.cache_dirs.append(folder)
        return folder

    def _confirm_replace(self):
        if not self.viewer.layers:
            return True
        answer = QMessageBox.question(self, "Open another dataset", "Opening another dataset replaces the current "
                                      "viewer layers. Save an editable project first if you need to keep your work.\n\n"
                                      "Replace the current session?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def open_folder(self):
        chosen = QFileDialog.getExistingDirectory(self, "Choose original dataset folder", str(self.settings_store.value("last_folder", "")))
        if chosen:
            self.load_selection(chosen)

    def open_slice(self):
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose one slice; its complete numbered series will open",
                                              str(self.settings_store.value("last_folder", "")),
                                              "Image slices (*.bmp *.png *.tif *.tiff *.jpg *.jpeg)")
        if chosen:
            self.load_selection(chosen)

    def load_selection(self, selection, project_path=None, metadata=None):
        if not self._confirm_replace():
            return
        cache = self._cache()
        on_disk = self.disk_cache.isChecked()

        def load(progress, cancel):
            dataset = inspect_dataset(selection, progress, cancel)
            raw = load_volume(dataset, cache / "source.npy" if on_disk else None, progress, cancel)
            restored = restore_labels(project_path, metadata, dataset, cache, progress, cancel) if metadata else []
            return dataset, raw, restored

        def loaded(result):
            dataset, raw, restored = result
            self._restoring = True
            self.viewer.layers.clear()
            self.dataset = dataset
            self.raw = self.viewer.add_image(raw, name=f"{dataset.name} — original", colormap="gray", rgb=False,
                                             metadata={"microct_source": dataset.to_dict()})
            self.raw.editable = False
            self.viewer.dims.axis_labels = ("slice (Z)", "Y", "X")
            self.viewer.dims.ndisplay = 2
            self.viewer.dims.set_current_step(0, dataset.shape[0] // 2)
            for source in restored:
                self.viewer.add_labels(source.data, name=source.name)
            self._restoring = False
            self._layers_changed()
            self.dirty = False
            self.saved_project = project_path
            if metadata:
                self._restore_settings(metadata)
            else:
                self.ids.clear()
                self._reload_coverage_presets()
            self.settings_store.setValue("last_folder", str(dataset.directory))
            self.info.setText(f"{dataset.directory}\n{dataset.shape[0]:,} slices · "
                              f"{dataset.shape[2]} × {dataset.shape[1]} px · {dataset.dtype}\n"
                              f"Image array: {dataset.nbytes / 1024**3:.2f} GiB ({'disk cache' if on_disk else 'RAM'})")
            self.destination.setText(f"Output parent folder:\n{dataset.directory.parent}")
            self.status.setText("Dataset ready. Segment a bone, then preview and export.")

        self._run_job("Opening dataset…", load, loaded)

    def open_project(self):
        chosen, _ = QFileDialog.getOpenFileName(self, "Reopen editable project", "", "MicroCT project (project.json)")
        if not chosen:
            return
        path, metadata, source = read_project(chosen)
        if not source.is_dir():
            moved = QFileDialog.getExistingDirectory(self, "Locate the original dataset folder for this project")
            if not moved:
                return
            source = Path(moved)
        first = source / metadata["dataset"]["slices"][0]["name"]
        if not first.is_file():
            raise ValidationError("The saved first source slice is missing from the selected dataset folder.")
        self.load_selection(first, path, metadata)

    def open_nninteractive(self):
        self._require_dataset()
        self.viewer.layers.selection.active = self.raw
        dock, self.nninteractive = self.viewer.window.add_plugin_dock_widget("napari-nninteractive", "nnInteractive")
        dock.show()
        dock.raise_()

    def session_info(self):
        panel = self.nninteractive
        if getattr(panel, "session", None) is None:
            return None
        checkpoint = getattr(panel, "checkpoint_path", None)
        if not checkpoint:
            return None
        path = Path(checkpoint)
        result = {"local_checkpoint_folder": str(path)}
        for parent in (path, *path.parents):
            if parent.parent.name == "snapshots":
                result["snapshot"] = parent.name
                break
        return result

    def current_z(self):
        point = self.viewer.dims.point
        coordinate = self.raw.world_to_data(point)
        return int(np.clip(round(coordinate[0]), 0, self.dataset.shape[0] - 1))

    def preview(self):
        sources = self.sources()
        Preview(self, sources, self.correction()).exec()

    def export(self):
        sources, settings, dataset = self.sources(), self.correction(), self.dataset
        mode = self.mode.currentData()
        session = self.session_info()

        def done(folder):
            self.last_output = folder
            self.status.setText(f"Export complete.\n{folder}")
            QMessageBox.information(self, "Masks exported", f"Verified BMP masks saved to:\n{folder}\n\n"
                                    "Use CTAN_IMPORT.txt for the import steps. Save an editable project "
                                    "as well if you want to resume this segmentation.")

        self._run_job("Exporting CTAn masks…", lambda p, c: export_masks(dataset, sources, settings, mode, p, c, session=session), done)

    def save(self, after=None):
        sources, settings, dataset = self.sources(all_layers=True), self.correction(), self.dataset
        mode = self.mode.currentData()
        session = self.session_info()

        def done(path):
            self.saved_project = path
            self.last_output = path.parent
            self.dirty = False
            self.status.setText(f"Editable project saved.\n{path}")
            if after:
                after()

        self._run_job("Saving editable project…", lambda p, c: save_project(dataset, sources, settings, mode, p, c, session=session), done)

    def show_output(self):
        if self.last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and (
            watched is getattr(self, "mode", None) or watched in getattr(self, "_coverage_controls", ())
        ):
            # Let the scroll area handle the wheel, even when an editor has focus.
            event.ignore()
            return True
        if watched is getattr(self, "_main_window", None) and event.type() == QEvent.Type.Close:
            if self.worker is not None:
                event.ignore()
                self.status.setText("Wait for the operation or cancel it before closing.")
                return True
            if not self.allow_close and self.dataset is not None and self._items:
                # Ask even after a save: some plugins change arrays in place without events.
                box = QMessageBox(QMessageBox.Icon.Question, "Keep editable segmentation?",
                                  "Save a new editable project before closing?\n"
                                  "BMP exports do not preserve separate label IDs.", parent=self)
                box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
                box.setDefaultButton(QMessageBox.StandardButton.Save)
                answer = box.exec()
                if answer == QMessageBox.StandardButton.Cancel:
                    event.ignore()
                    return True
                if answer == QMessageBox.StandardButton.Save:
                    event.ignore()
                    def close_after_save():
                        self.allow_close = True
                        self._main_window.close()
                    self._guard(lambda: self.save(after=close_after_save))
                    return True
        return super().eventFilter(watched, event)

    def cleanup(self):
        # On Windows numpy mappings may remain referenced by plugins until process exit.
        # Remove only this app instance's temporary folders; never other sessions' caches.
        if self.worker is not None:
            return
        self.viewer.layers.clear()
        self.raw = None
        self._items.clear()
        import gc
        gc.collect()
        for folder in self.cache_dirs:
            shutil.rmtree(folder, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MicroCT Segmenter: napari to CTAn")
    parser.add_argument("dataset", nargs="?", help="Optional original dataset folder or slice filename")
    args = parser.parse_args()
    viewer = napari.Viewer(title="MicroCT Segmenter")
    widget = MicroCTWidget(viewer)
    viewer.window.add_dock_widget(widget, name="MicroCT Segmenter", area="right")
    if args.dataset:
        QTimer.singleShot(0, lambda: widget._guard(lambda: widget.load_selection(args.dataset)))
    try:
        napari.run()
    finally:
        widget.cleanup()


if __name__ == "__main__":
    main()
