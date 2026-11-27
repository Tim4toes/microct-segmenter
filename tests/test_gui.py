"""Qt/napari tests; use QT_QPA_PLATFORM=offscreen on a headless test machine."""
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PIL import Image
import napari
from qtpy.QtWidgets import QApplication, QInputDialog, QMessageBox, QProgressDialog, QPushButton
from qtpy.QtCore import QPoint, QPointF, QSettings, Qt
from qtpy.QtGui import QWheelEvent

from microct_segmenter.core import Correction, ValidationError, validate_layer_transform
from microct_segmenter.gui import MicroCTWidget, Preview


@pytest.fixture
def ui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings_path = str(tmp_path / "settings.ini")
    monkeypatch.setattr("microct_segmenter.gui.QSettings",
                        lambda *_: QSettings(settings_path, QSettings.Format.IniFormat))
    viewer = napari.Viewer(show=False)
    widget = MicroCTWidget(viewer)
    viewer.window.add_dock_widget(widget, name='MicroCT Segmenter')
    widget.settings_store.setValue('cache_base', str(tmp_path / 'cache'))
    widget._confirm_replace = lambda: True
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: QMessageBox.StandardButton.Ok)
    failures = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **k: failures.append(a))
    yield app, viewer, widget, failures
    widget.allow_close = True
    viewer.close()
    widget.cleanup()


def wait_job(app, widget):
    deadline = time.monotonic() + 30
    while widget.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert widget.worker is None, 'Background job did not finish'


def test_gui_open_export_preview_save_reopen(ui, tmp_path):
    app, viewer, widget, failures = ui
    source = tmp_path / 'images'
    source.mkdir()
    for z in range(3):
        plane = np.zeros((31, 29), np.uint8)
        plane[7:22, 8:21] = 150
        Image.fromarray(plane).save(source / f'scan{z:08d}.bmp')
    widget.dx.setValue(2)
    widget.save_coverage_default()
    widget.load_selection(source)
    wait_job(app, widget)
    assert widget.dataset.shape == (3, 31, 29)
    assert widget.correction() == Correction(2, 0, 0, 1)
    labels = np.zeros((3, 31, 29), np.uint16)
    labels[1, 10:20, 10:20] = 2
    layer = viewer.add_labels(labels, name='parietal')
    widget.ids.setText('2')
    sources = widget.sources()
    assert sources[0].ids == (2,)
    preview = Preview(widget, sources, widget.correction())
    preview.render()
    assert not preview.after.item.pixmap().isNull()
    preview.close()
    widget.export()
    wait_job(app, widget)
    assert (widget.last_output / 'COMPLETE.txt').exists()
    assert (widget.last_output / 'scan00000001.bmp').exists()
    widget.dy.setValue(3)
    widget.save()
    wait_job(app, widget)
    path = widget.saved_project
    assert path.exists()
    from microct_segmenter.core import read_project
    path, metadata, folder = read_project(path)
    widget.coverage_lock.setChecked(True)
    widget.dy.setValue(0)
    widget.load_selection(folder / 'scan00000000.bmp', path, metadata)
    wait_job(app, widget)
    assert np.array_equal(widget._items[0].data, labels)
    assert widget.correction() == Correction(2, 3, 0, 1)
    assert not widget.dy.isEnabled()
    widget.load_selection(source)
    wait_job(app, widget)
    assert widget.correction() == Correction(2, 0, 0, 1)
    assert not failures


def test_transform_mismatch_is_rejected(ui):
    app, viewer, widget, failures = ui
    raw = viewer.add_image(np.zeros((3, 19, 23)), rgb=False)
    labels = viewer.add_labels(np.zeros((3, 19, 23), np.uint8))
    validate_layer_transform(raw, labels)
    labels.translate = (0, 1, 0)
    with pytest.raises(ValidationError, match='translated'):
        validate_layer_transform(raw, labels)


def test_named_coverage_and_default_persist_separately(ui, monkeypatch):
    _, _, widget, failures = ui
    assert [widget.preset.itemText(i) for i in range(widget.preset.count())] == [
        "Native pixels: no correction", "Custom",
    ]
    assert widget.correction() == Correction(0, 0, 0, 1)
    widget.dx.setValue(3)
    buttons = {button.text(): button for button in widget.findChildren(QPushButton)}
    buttons["Save as default"].click()
    widget.dilation.setValue(2)
    widget.depth.setCurrentIndex(widget.depth.findData(8))
    assert widget.preset.currentText() == "Custom"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (" Skull / dense ", True))
    buttons["Save preset…"].click()
    assert widget.preset.currentText() == "Skull / dense"

    viewer = napari.Viewer(show=False)
    reopened = MicroCTWidget(viewer)
    try:
        assert reopened.preset.currentText() == "Saved default"
        assert reopened.correction() == Correction(3, 0, 0, 1)
        reopened.preset.setCurrentText("Skull / dense")
        assert reopened.correction() == Correction(3, 0, 2, 8)
        reopened.preset.setCurrentText("Native pixels: no correction")
        assert reopened.correction() == Correction(0, 0, 0, 1)
        reopened.depth.setCurrentIndex(reopened.depth.findData(8))
        assert reopened.preset.currentText() == "Custom"
        assert not failures
    finally:
        viewer.close()


def test_editors_ignore_wheel_events(ui):
    app, _, widget, _ = ui
    # Check focused controls too. That is where a wheel movement used to change coverage.
    for control in (widget.mode, widget.preset, widget.dx, widget.dy, widget.dilation, widget.depth):
        control.setFocus()
        center = control.rect().center()
        wheel = QWheelEvent(QPointF(center), QPointF(control.mapToGlobal(center)),
                            QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        app.sendEvent(control, wheel)
        assert not wheel.isAccepted()
        assert widget.correction() == Correction(0, 0, 0, 1)
        assert widget.preset.currentText() == "Native pixels: no correction"
        assert widget.mode.currentData() == "combined"


def test_coverage_lock_persists_and_leaves_export_available(ui):
    _, _, widget, _ = ui
    button = next(b for b in widget.findChildren(QPushButton) if b.text() == "Export CTAn masks")
    widget.dy.setValue(3)
    widget.coverage_lock.click()
    assert widget.correction() == Correction(0, 3, 0, 1)
    assert all(not control.isEnabled() for control in (widget.preset, widget.dx, widget.dy, widget.dilation, widget.depth))
    assert button.isEnabled()

    viewer = napari.Viewer(show=False)
    reopened = MicroCTWidget(viewer)
    try:
        assert reopened.coverage_lock.isChecked()
        assert not reopened.preset.isEnabled()
        reopened.coverage_lock.click()
        reopened.dy.setValue(2)
        assert reopened.dy.isEnabled()
        assert reopened.correction() == Correction(0, 2, 0, 1)
    finally:
        viewer.close()


def test_progress_cancel_button_remains_usable(ui):
    app, viewer, widget, failures = ui
    from microct_segmenter.core import checkpoint
    completed = []
    def slow(progress, cancel):
        for i in range(1000):
            checkpoint(cancel)
            progress(i, 1000, 'Cancellable operation')
            time.sleep(0.01)
    widget._run_job('Testing cancellation', slow, completed.append)
    app.processEvents()
    dialogs = widget._main_window.findChildren(QProgressDialog)
    assert dialogs
    button = next(b for b in dialogs[-1].findChildren(QPushButton) if b.text() == 'Cancel')
    assert button.isEnabled()
    button.click()
    wait_job(app, widget)
    assert not completed
    assert 'Cancelled' in widget.status.text()
    assert widget._main_window.isEnabled()


def test_nninteractive_entry_and_result_selection(ui, tmp_path, monkeypatch):
    app, viewer, widget, failures = ui
    buttons = {button.text(): button for button in widget.findChildren(QPushButton)}
    assert "New bone Labels layer" not in buttons
    assert not any(".roi" in text for text in buttons)
    source = tmp_path / "images"
    source.mkdir()
    Image.fromarray(np.zeros((17, 23), np.uint8)).save(source / "scan0000.bmp")
    widget.load_selection(source)
    wait_job(app, widget)
    from unittest.mock import Mock
    opened = Mock(return_value=(Mock(), Mock()))
    monkeypatch.setattr(viewer.window, "add_plugin_dock_widget", opened)
    buttons["Open nnInteractive"].click()
    opened.assert_called_once_with("napari-nninteractive", "nnInteractive")
    assert viewer.layers.selection.active is widget.raw
    shape = widget.dataset.shape
    viewer.add_labels(np.zeros(shape, np.uint8), name="parietal")
    viewer.add_labels(np.zeros(shape, np.uint8), name="frontal")
    scribble = type("ScribbleLayer", (napari.layers.Labels,),
                    {"__module__": "napari_nninteractive.layers.scribble_layer"})
    viewer.add_layer(scribble(np.ones(shape, np.uint8), name="renamed prompt"))
    widget.layer_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert [s.name for s in widget.sources()] == ["frontal"]
    assert [s.name for s in widget.sources(all_layers=True)] == ["parietal", "frontal"]
    assert not failures


def test_installed_plugin_dock_and_its_real_scribble_layer(ui, tmp_path, monkeypatch):
    plugin_module = pytest.importorskip("napari_nninteractive")
    from napari_nninteractive.layers.scribble_layer import ScribbleLayer
    from napari_nninteractive._version_check import VersionChecker
    monkeypatch.setattr(VersionChecker, "start", lambda self: None)
    app, viewer, widget, failures = ui
    source = tmp_path / "slices"
    source.mkdir()
    for z in range(3):
        Image.fromarray(np.zeros((19, 27), np.uint8)).save(source / f"rat{z:05d}.bmp")
    widget.load_selection(source)
    wait_job(app, widget)
    widget.open_nninteractive()
    panel = widget.nninteractive
    assert isinstance(panel, plugin_module.nnInteractiveWidget)
    assert panel.image_selection.findText(widget.raw.name) >= 0
    prompt = ScribbleLayer(data=np.zeros(widget.dataset.shape, np.uint8), name="input")
    viewer.add_layer(prompt)
    prompt.name = "renamed input"
    result = viewer.add_labels(np.ones(widget.dataset.shape, np.uint8), name="finished bone")
    assert [layer.name for layer in widget.sources()] == [result.name]
    assert [layer.name for layer in widget.sources(all_layers=True)] == [result.name]
    assert not failures
