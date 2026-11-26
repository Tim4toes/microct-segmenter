"""Image I/O and CTAn export, independent of Qt and napari.

Arrays are always Z,Y,X. No resizing, transposition, thresholding of source
intensities, or slice interpolation is performed. Corrections affect masks only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import importlib.metadata
import json
import os
import re
import shutil
import struct
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

from . import __version__

EXTENSIONS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}
Progress = Callable[[int, int, str], None]


class ValidationError(ValueError):
    """An input cannot safely be mapped to the original pixel grid."""


class Cancelled(Exception):
    pass


def checkpoint(cancel=None):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled; no completed output was written.")


def report(progress, done, total, message):
    if progress:
        progress(done, max(total, 1), message)


def natural_key(value):
    return tuple((1, int(s)) if s.isdigit() else (0, s.casefold())
                 for s in re.split(r"(\d+)", str(value)))


def series_key(path: Path):
    m = re.fullmatch(r"(.*?)(\d+)(\D*)", path.stem)
    return (m[1].casefold(), m[3].casefold(), path.suffix.lower()) if m else (
        path.stem.casefold(), "", path.suffix.lower())


def select_files(selection: str | Path):
    selection = Path(selection).expanduser().resolve()
    directory = selection if selection.is_dir() else selection.parent
    if not directory.is_dir():
        raise ValidationError("Dataset folder does not exist.")
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if selection.is_file():
        if selection.suffix.lower() not in EXTENSIONS:
            raise ValidationError("Select a BMP, PNG, or single-page TIFF slice.")
        key = series_key(selection)
        files = [p for p in files if series_key(p) == key]
    elif len({series_key(p) for p in files}) > 1:
        raise ValidationError("This folder contains several image series. Use 'Choose one slice' "
                              "and select a slice from the intended series.")
    if not files:
        raise ValidationError("No supported image slices were found.")
    files.sort(key=lambda p: natural_key(p.name))
    stems = [p.stem.casefold() for p in files]
    if len(set(stems)) != len(stems):
        raise ValidationError("Duplicate output BMP names would be produced.")
    numbers = [re.fullmatch(r"(.*?)(\d+)(\D*)", p.stem) for p in files]
    if len(files) > 1:
        if not all(numbers):
            raise ValidationError("A multi-slice dataset needs a numbered filename sequence.")
        indices = [int(m[2]) for m in numbers]
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValidationError("Duplicate or missing slice numbers detected. Restore the complete "
                                  "consecutive stack before loading; missing slices are not invented.")
    return directory, files


def _ppm(im, path):
    if path.suffix.lower() == ".bmp":
        with path.open("rb") as stream:
            header = stream.read(54)
        if len(header) >= 54 and struct.unpack_from("<I", header, 14)[0] >= 40:
            return struct.unpack_from("<ii", header, 38)
    dpi = im.info.get("dpi")
    if dpi:
        return tuple(round(float(x) / 0.0254) for x in dpi[:2])
    return (0, 0)


def _dtype(im):
    if getattr(im, "n_frames", 1) != 1:
        raise ValidationError("Multi-page images are not supported. Use one 2D slice per file.")
    if im.mode in {"1", "L"}:
        return np.dtype("uint8")
    if im.mode == "P":
        palette = np.asarray(im.getpalette(), dtype=np.uint8).reshape(-1, 3)
        if not np.all(palette[:, 0] == palette[:, 1]) or not np.all(palette[:, 1] == palette[:, 2]):
            raise ValidationError("Colour palette images are not greyscale CT slices.")
        return np.dtype("uint8")
    if im.mode.startswith("I;16"):
        return np.dtype("uint16")
    raise ValidationError(f"Unsupported image mode {im.mode!r}. Use greyscale 8/16-bit slices.")


def decode_slice(blob):
    with Image.open(io.BytesIO(blob)) as im:
        dtype = _dtype(im)
        if im.mode in {"P", "1"}:
            im = im.convert("L")
        return np.array(im, dtype=dtype)


@dataclass
class SliceRecord:
    name: str
    size: int
    mtime_ns: int
    ppm: tuple[int, int]
    sha256: str = ""


@dataclass
class Dataset:
    directory: Path
    shape: tuple[int, int, int]
    dtype: str
    slices: list[SliceRecord]

    @property
    def name(self):
        return self.directory.name

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def to_dict(self):
        return {"directory": str(self.directory), "shape_zyx": list(self.shape),
                "dtype": self.dtype, "slices": [asdict(s) for s in self.slices]}


def inspect_dataset(selection, progress=None, cancel=None):
    directory, paths = select_files(selection)
    shape, dtype, records = None, None, []
    for i, path in enumerate(paths):
        checkpoint(cancel)
        try:
            with Image.open(path) as im:
                this_shape, this_dtype = (im.height, im.width), _dtype(im).name
                ppm = _ppm(im, path)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Cannot use {path.name}: {exc}") from exc
        if shape is None:
            shape, dtype = this_shape, this_dtype
        elif this_shape != shape or this_dtype != dtype:
            raise ValidationError(f"{path.name} differs in dimensions or bit depth from the stack.")
        stat = path.stat()
        records.append(SliceRecord(path.name, stat.st_size, stat.st_mtime_ns, ppm))
        report(progress, i + 1, len(paths), f"Checking {path.name}")
    return Dataset(directory, (len(paths), *shape), dtype, records)


def ensure_space(folder, nbytes):
    free = shutil.disk_usage(folder).free
    required = int(nbytes) + 64 * 1024**2
    if free < required:
        raise ValidationError(f"Insufficient disk space in {folder}. Need about "
                              f"{required / 1024**3:.2f} GiB; {free / 1024**3:.2f} GiB free.")


def allocate_array(shape, dtype, cache_path=None):
    if cache_path is None:
        return np.zeros(shape, dtype=dtype)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(cache_path.parent, int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return np.lib.format.open_memmap(cache_path, mode="w+", dtype=dtype, shape=shape)


def load_volume(dataset, cache_path=None, progress=None, cancel=None):
    volume = allocate_array(dataset.shape, dataset.dtype, cache_path)
    for i, record in enumerate(dataset.slices):
        checkpoint(cancel)
        path = dataset.directory / record.name
        stat = path.stat()
        if stat.st_size != record.size or stat.st_mtime_ns != record.mtime_ns:
            raise ValidationError(f"Source changed during loading: {record.name}")
        blob = path.read_bytes()
        plane = decode_slice(blob)
        if plane.shape != dataset.shape[1:] or plane.dtype.name != dataset.dtype:
            raise ValidationError(f"Unexpected image format in {record.name}")
        volume[i] = plane
        record.sha256 = hashlib.sha256(blob).hexdigest()
        report(progress, i + 1, len(dataset.slices), f"Loading {record.name}")
    if isinstance(volume, np.memmap):
        volume.flush()
    return volume


def verify_source(dataset, i):
    record = dataset.slices[i]
    if not record.sha256:
        raise ValidationError("The source stack must be loaded and fingerprinted before export.")
    path = dataset.directory / record.name
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"Cannot read original slice {record.name}: {exc}") from exc
    if digest.hexdigest() != record.sha256:
        raise ValidationError(f"Original slice changed since loading: {record.name}. Reopen the dataset.")


@dataclass(frozen=True)
class Correction:
    x_shift: int = 0
    y_shift: int = 1
    dilation: int = 2
    bit_depth: int = 1

    def validate(self):
        if any(type(v) is not int for v in (self.x_shift, self.y_shift, self.dilation, self.bit_depth)):
            raise ValidationError("Correction settings must be integers.")
        if self.dilation < 0 or self.dilation > 100:
            raise ValidationError("Dilation must be between 0 and 100 pixels.")
        if self.bit_depth not in (1, 8):
            raise ValidationError("BMP bit depth must be 1 or 8.")


def correct_mask(mask, settings: Correction):
    settings.validate()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValidationError("Each mask slice must be two-dimensional.")
    h, w = mask.shape
    dy, dx = settings.y_shift, settings.x_shift
    result = np.zeros_like(mask)
    if abs(dy) < h and abs(dx) < w:
        sy, sx = max(0, -dy), max(0, -dx)
        ty, tx = max(0, dy), max(0, dx)
        nh, nw = h - abs(dy), w - abs(dx)
        result[ty:ty + nh, tx:tx + nw] = mask[sy:sy + nh, sx:sx + nw]
    if settings.dilation:
        result = maximum_filter(result, size=2 * settings.dilation + 1, mode="constant", cval=0)
    return result


@dataclass
class LabelSource:
    name: str
    data: object
    ids: tuple[int, ...] | None = None

    def plane(self, index):
        a = np.asarray(self.data[index])
        return a > 0 if self.ids is None else np.isin(a, self.ids)


def validate_labels(dataset, labels):
    if not labels:
        raise ValidationError("Select at least one Labels layer to export.")
    for source in labels:
        if tuple(source.data.shape) != dataset.shape:
            raise ValidationError(f"'{source.name}' has shape {source.data.shape}; the original stack "
                                  f"is {dataset.shape}. Resized/cropped labels cannot be exported "
                                  "without an explicit spatial mapping.")
        if np.dtype(source.data.dtype).kind not in "bui":
            raise ValidationError(f"'{source.name}' must contain integer label IDs.")
        if source.ids is not None and (not source.ids or any(type(n) is not int or n < 1 for n in source.ids)):
            raise ValidationError("Label IDs must be positive integers; 0 is background.")


def validate_layer_transform(raw, label):
    """Equal array sizes alone do not guarantee that layers occupy the same grid."""
    if getattr(raw, "multiscale", False) or getattr(label, "multiscale", False):
        raise ValidationError("Export requires single-resolution image and label layers.")
    if raw.ndim != 3 or label.ndim != 3:
        raise ValidationError("Export requires 3D Z,Y,X image and label layers.")
    points = np.vstack([np.zeros(3), np.eye(3), np.array(raw.data.shape) - 1])
    for point in points:
        if not np.allclose(raw.data_to_world(point), label.data_to_world(point), rtol=0, atol=1e-6):
            raise ValidationError(f"'{label.name}' is translated, scaled, or rotated relative to the "
                                  "source. Restore its native grid before exporting.")


def safe_name(name):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")[:80] or "mask"
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", value):
        value = "mask_" + value
    return value


def write_bmp(path, mask, ppm=(0, 0), bit_depth=1):
    """Uncompressed Windows BMP, exact PPM, top-left logical image, padded bottom-up rows."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValidationError("Cannot write an empty or non-2D bitmap.")
    h, w = mask.shape
    if bit_depth == 1:
        palette = b"\0\0\0\0\xff\xff\xff\0"
        payload = np.packbits(mask[::-1], axis=1, bitorder="big")
    elif bit_depth == 8:
        palette = b"".join(bytes((i, i, i, 0)) for i in range(256))
        payload = mask[::-1].astype(np.uint8) * 255
    else:
        raise ValidationError("Only 1-bit and 8-bit BMP output is supported.")
    row_bytes = ((w * bit_depth + 31) // 32) * 4
    rows = np.zeros((h, row_bytes), dtype=np.uint8)
    rows[:, :payload.shape[1]] = payload
    offset = 14 + 40 + len(palette)
    header = struct.pack("<2sIHHI", b"BM", offset + rows.nbytes, 0, 0, offset)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, bit_depth, 0, rows.nbytes,
                      int(ppm[0]), int(ppm[1]), 2**bit_depth, 2**bit_depth)
    with Path(path).open("xb") as stream:
        stream.write(header + dib + palette)
        stream.write(rows.tobytes())


def verify_bmp(path, expected, ppm, bit_depth):
    with Path(path).open("rb") as stream:
        header = stream.read(54)
    if struct.unpack_from("<H", header, 28)[0] != bit_depth:
        raise ValidationError(f"BMP bit depth verification failed: {path}")
    if struct.unpack_from("<ii", header, 38) != tuple(ppm):
        raise ValidationError(f"BMP pixel-size verification failed: {path}")
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"))
    if a.shape != expected.shape or not np.array_equal(a, expected.astype(np.uint8) * 255):
        raise ValidationError(f"BMP pixel verification failed: {path}")


def combine_plane(labels, index):
    result = np.zeros(labels[0].data.shape[1:], dtype=bool)
    for source in labels:
        result |= source.plane(index)
    return result


def _targets(labels, mode, progress=None, cancel=None):
    if mode == "combined":
        return [(".", labels)]
    if mode == "layers":
        return [(f"{i + 1:02d}_{safe_name(s.name)}", [s]) for i, s in enumerate(labels)]
    if mode != "ids":
        raise ValidationError("Export mode must be combined, layers, or ids.")
    targets = []
    total = sum(s.data.shape[0] for s in labels)
    done = 0
    for li, source in enumerate(labels):
        ids = set()
        for zi in range(source.data.shape[0]):
            checkpoint(cancel)
            ids.update(int(n) for n in np.unique(source.data[zi]) if n > 0)
            done += 1
            report(progress, done, total, "Finding separate label IDs")
        if source.ids is not None:
            ids.intersection_update(source.ids)
        for identifier in sorted(ids):
            targets.append((f"{li + 1:02d}_{safe_name(source.name)}_id{identifier}",
                            [LabelSource(source.name, source.data, (identifier,))]))
    if not targets:
        raise ValidationError("The selected labels contain no foreground voxels.")
    return targets


def _new_paths(dataset, kind):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = dataset.directory.parent / f"{safe_name(dataset.name)}_{kind}_{stamp}_{uuid.uuid4().hex[:6]}"
    staging = final.with_name(final.name + ".partial")
    return final, staging


def runtime_info(session=None):
    packages = {}
    for name in ("napari", "napari-nninteractive", "nninteractive", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version.split()[0], "packages": packages, "open_nninteractive_session": session}


def export_masks(dataset, labels, settings=Correction(), mode="combined", progress=None, cancel=None, *, session=None):
    validate_labels(dataset, labels)
    settings.validate()
    targets = _targets(labels, mode, progress, cancel)
    final, staging = _new_paths(dataset, "masks")
    h, w = dataset.shape[1:]
    one_bmp_size = (((w * settings.bit_depth + 31) // 32) * 4 * h) + 1078
    ensure_space(final.parent, one_bmp_size * dataset.shape[0] * len(targets))
    staging.mkdir()
    totals = {name: 0 for name, _ in targets}
    edge_slices = {name: 0 for name, _ in targets}
    try:
        for name, _ in targets:
            if name != ".":
                (staging / name).mkdir()
        with (staging / "slice_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["z_index", "source_filename", "mask_relative_path", "width", "height",
                             "foreground_pixels", "touches_image_edge", "source_sha256", "mask_sha256"])
            for zi, record in enumerate(dataset.slices):
                checkpoint(cancel)
                verify_source(dataset, zi)
                for name, sources in targets:
                    checkpoint(cancel)
                    plane = correct_mask(combine_plane(sources, zi), settings)
                    filename = Path(record.name)
                    target = staging / name / (filename.name if filename.suffix.lower() == ".bmp" else filename.with_suffix(".bmp").name)
                    write_bmp(target, plane, record.ppm, settings.bit_depth)
                    verify_bmp(target, plane, record.ppm, settings.bit_depth)
                    count = int(plane.sum())
                    touches = bool(plane[0].any() or plane[-1].any() or plane[:, 0].any() or plane[:, -1].any())
                    totals[name] += count
                    edge_slices[name] += int(touches)
                    writer.writerow([zi, record.name, target.relative_to(staging).as_posix(), w, h,
                                     count, touches, record.sha256, hashlib.sha256(target.read_bytes()).hexdigest()])
                report(progress, zi + 1, dataset.shape[0], f"Exporting and verifying {record.name}")
        if not any(totals.values()):
            raise ValidationError("The export is completely empty after selecting labels and applying corrections.")
        manifest = {"software": "MicroCT Segmenter", "version": __version__, "runtime": runtime_info(session),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": dataset.to_dict(),
                    "axis_order": "ZYX", "settings": asdict(settings), "mode": mode,
                    "foreground": "white (255 decoded)", "background": "black (0)",
                    "operation_order": "select positive IDs; union; XY zero-fill shift; 2D square dilation",
                    "outputs": [{"folder": n, "foreground_voxels": totals[n],
                                 "slices_touching_image_edge": edge_slices[n],
                                 "layers": [{"name": s.name, "ids": s.ids} for s in group]}
                                for n, group in targets]}
        (staging / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging / "CTAN_IMPORT.txt").write_text(CTAN_IMPORT, encoding="utf-8")
        checkpoint(cancel)
        (staging / "COMPLETE.txt").write_text("All BMP pixels, dimensions, bit depths and PPM verified.\n", encoding="utf-8")
        staging.rename(final)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def save_project(dataset, labels, settings=Correction(), mode="combined", progress=None, cancel=None, *, session=None):
    validate_labels(dataset, labels)
    final, staging = _new_paths(dataset, "segmentation")
    ensure_space(final.parent, sum(int(np.prod(s.data.shape)) * np.dtype(s.data.dtype).itemsize for s in labels))
    staging.mkdir()
    target = None
    try:
        entries = []
        total = dataset.shape[0] * (len(labels) + 1)
        done = 0
        for zi in range(dataset.shape[0]):
            checkpoint(cancel)
            verify_source(dataset, zi)
            done += 1
            report(progress, done, total, "Verifying original data")
        for li, source in enumerate(labels):
            filename = f"labels_{li + 1:03d}.npy"
            target = allocate_array(dataset.shape, source.data.dtype, staging / filename)
            digest = hashlib.sha256()
            for zi in range(dataset.shape[0]):
                checkpoint(cancel)
                plane = np.ascontiguousarray(source.data[zi])
                target[zi] = plane
                digest.update(plane.tobytes())
                done += 1
                report(progress, done, total, f"Saving editable labels: {source.name}")
            target.flush()
            del target
            target = None
            entries.append({"name": source.name, "file": filename, "dtype": str(source.data.dtype),
                            "ids": source.ids, "pixels_sha256": digest.hexdigest()})
        metadata = {"format": "microct-segmentation-v1", "version": __version__, "runtime": runtime_info(session),
                    "source_relative": os.path.relpath(dataset.directory, final), "dataset": dataset.to_dict(),
                    "labels": entries, "settings": asdict(settings), "mode": mode}
        (staging / "project.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        checkpoint(cancel)
        staging.rename(final)
        return final / "project.json"
    except BaseException:
        if target is not None:
            del target
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_project(path):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "project.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != "microct-segmentation-v1":
        raise ValidationError("This is not a supported MicroCT Segmenter project.")
    relative = (path.parent / metadata["source_relative"]).resolve()
    absolute = Path(metadata["dataset"]["directory"])
    source = relative if relative.is_dir() else absolute
    return path, metadata, source


def restore_labels(path, metadata, dataset, cache_dir, progress=None, cancel=None):
    expected = metadata["dataset"]
    if list(dataset.shape) != expected["shape_zyx"] or dataset.dtype != expected["dtype"]:
        raise ValidationError("The selected dataset does not match the saved project dimensions/bit depth.")
    if [(s.name, s.sha256) for s in dataset.slices] != [(s["name"], s["sha256"]) for s in expected["slices"]]:
        raise ValidationError("The original filenames or image contents do not match this project.")
    result = []
    for li, item in enumerate(metadata["labels"]):
        source_path = (Path(path).parent / item["file"]).resolve()
        if source_path.parent != Path(path).parent.resolve():
            raise ValidationError("Project label filenames must be inside the project folder.")
        data = np.load(source_path, mmap_mode="r", allow_pickle=False)
        ids = None if item.get("ids") is None else tuple(item["ids"])
        validate_labels(dataset, [LabelSource(item["name"], data, ids)])
        target = allocate_array(dataset.shape, data.dtype, Path(cache_dir) / f"restored_{li}.npy")
        digest = hashlib.sha256()
        for zi in range(dataset.shape[0]):
            checkpoint(cancel)
            plane = np.ascontiguousarray(data[zi])
            digest.update(plane.tobytes())
            target[zi] = plane
            report(progress, li * dataset.shape[0] + zi + 1, len(metadata["labels"]) * dataset.shape[0],
                   f"Restoring {item['name']}")
        if digest.hexdigest() != item["pixels_sha256"]:
            raise ValidationError(f"Saved labels failed the integrity check: {item['name']}")
        target.flush()
        result.append(LabelSource(item["name"], target, ids))
    return result


CTAN_IMPORT = """CTAn 1.20.3.0: importing a native-size binary ROI stack

1. Open the ORIGINAL dataset in CTAn, keeping its original slice numbers.
2. Open 'Load the regions of interest' and select 'Bitmap files (*.bmp)'.
   Select the corresponding BMP in this mask folder (or a per-bone subfolder).
   'Load Image ROI' is the single-image command; use the stack/VOI import for a stack.
3. In Preferences > Region of Interest, use no resizing and no centering offset
   for these full-size, pixel-aligned masks. Width, height and BMP PPM match source.
4. Check the first, middle, last and thin-bone slices visually before analysis.
   Keep the original filenames/postfix digit count (e.g. 8 for Tra00001919).
5. Keep analysing the original intensity images, with the white mask as the ROI.
   Do not use the binary mask as the intensity dataset for density measurements.

These are bitmap ROIs, not proprietary .roi files. CTAn can save its imported VOI
as a .roi file. An old .roi file may refer to different slice ranges or geometry;
it is not needed to import this complete per-slice mask stack.

0 is background; white is ROI. Blank slices are retained intentionally.
export.json records settings and source hashes. slice_mapping.csv records each
slice correspondence, foreground area, image-edge contact and output hash.
Masks receive the recorded shift/dilation once, on export, without altering the
editable segmentation. Dilation expands the measurement ROI; inspect adjacent bones.
"""
