import csv
import hashlib
import json
import shutil
import struct
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from microct_segmenter import core


def stack(tmp_path, shape=(3, 19, 23), dtype=np.uint8, start=8):
    folder = tmp_path / 'original'
    folder.mkdir()
    raw = (np.arange(np.prod(shape)).reshape(shape) % 233).astype(dtype)
    for i, plane in enumerate(raw):
        suffix = '.tif' if dtype == np.uint16 else '.bmp'
        Image.fromarray(plane).save(folder / f'scan{i + start}{suffix}', dpi=(3642, 3642))
    ds = core.inspect_dataset(folder)
    volume = core.load_volume(ds)
    return ds, volume


def test_natural_order_and_intensity_identity(tmp_path):
    ds, raw = stack(tmp_path)
    assert [s.name for s in ds.slices] == ['scan8.bmp', 'scan9.bmp', 'scan10.bmp']
    assert ds.shape == (3, 19, 23)
    for i, rec in enumerate(ds.slices):
        assert np.array_equal(raw[i], np.asarray(Image.open(ds.directory / rec.name)))
        assert rec.sha256 == hashlib.sha256((ds.directory / rec.name).read_bytes()).hexdigest()


def test_16_bit_is_preserved(tmp_path):
    ds, _ = stack(tmp_path, dtype=np.uint16)
    plane = np.full((19, 23), 53000, dtype=np.uint16)
    Image.fromarray(plane).save(ds.directory / 'scan8.tif')
    ds = core.inspect_dataset(ds.directory)
    loaded = core.load_volume(ds, tmp_path / 'cache.npy')
    assert loaded.dtype == np.uint16
    assert loaded[0, 10, 10] == 53000
    assert isinstance(loaded, np.memmap)


def test_multiple_series_requires_explicit_selection(tmp_path):
    ds, _ = stack(tmp_path)
    Image.fromarray(np.zeros((19, 23), np.uint8)).save(ds.directory / 'other8.bmp')
    with pytest.raises(core.ValidationError, match='several image series'):
        core.inspect_dataset(ds.directory)
    assert len(core.inspect_dataset(ds.directory / 'scan9.bmp').slices) == 3


def test_missing_slices_rejected(tmp_path):
    ds, _ = stack(tmp_path)
    (ds.directory / 'scan9.bmp').unlink()
    with pytest.raises(core.ValidationError, match='missing slice'):
        core.inspect_dataset(ds.directory)


def test_mismatched_shapes_rejected(tmp_path):
    ds, _ = stack(tmp_path)
    Image.fromarray(np.zeros((5, 23), np.uint8)).save(ds.directory / 'scan9.bmp')
    with pytest.raises(core.ValidationError, match='dimensions'):
        core.inspect_dataset(ds.directory)


def test_multipage_rejected(tmp_path):
    first = Image.fromarray(np.zeros((5, 7), np.uint8))
    path = tmp_path / 'stack.tif'
    first.save(path, save_all=True, append_images=[first])
    with pytest.raises(core.ValidationError, match='Multi-page'):
        core.inspect_dataset(path)


@pytest.mark.parametrize('bits', [1, 8])
@pytest.mark.parametrize('width', [1, 7, 8, 9, 23, 31, 32, 33, 2752])
def test_bmp_padding_orientation_ppm_and_pixels(tmp_path, bits, width):
    plane = np.zeros((11, width), bool)
    plane[0, 0] = True
    plane[-1, -1] = True
    plane[3:7, ::3] = True
    path = tmp_path / 'test.bmp'
    core.write_bmp(path, plane, (143386, 143386), bits)
    core.verify_bmp(path, plane, (143386, 143386), bits)
    with Image.open(path) as image:
        assert image.size == (width, 11)
        assert set(np.unique(np.asarray(image.convert('L')))) <= {0, 255}
    b = path.read_bytes()
    assert struct.unpack_from('<I', b, 2)[0] == len(b)


@pytest.mark.parametrize('shift', [(0, 0), (1, 0), (-1, 2), (5, -6), (100, 0), (0, -100)])
def test_shift_has_zero_fill_and_no_wrap(shift):
    dy, dx = shift
    plane = np.zeros((9, 11), bool)
    plane[2, 4] = True
    out = core.correct_mask(plane, core.Correction(dx, dy, 0))
    expected = np.zeros_like(plane)
    if 0 <= 2 + dy < 9 and 0 <= 4 + dx < 11:
        expected[2 + dy, 4 + dx] = True
    assert np.array_equal(out, expected)
    assert plane.sum() == 1


def test_original_script_correction_oracle():
    # Independent oracle: zero-fill downshift, then TWO separate 3x3 dilations.
    rng = np.random.default_rng(9)
    plane = rng.random((31, 29)) > 0.9
    expected = np.zeros_like(plane)
    expected[1:] = plane[:-1]
    for _ in range(2):
        padded = np.pad(expected, 1)
        result = np.zeros_like(expected)
        for dy in range(3):
            for dx in range(3):
                result |= padded[dy:dy + 31, dx:dx + 29]
        expected = result
    assert np.array_equal(core.correct_mask(plane, core.Correction()), expected)


def test_zero_dilation_is_identity_not_until_convergence():
    plane = np.zeros((9, 13), bool)
    plane[4, 6] = True
    assert np.array_equal(core.correct_mask(plane, core.Correction(0, 0, 0)), plane)
    assert core.correct_mask(plane, core.Correction(0, 0, 2)).sum() == 25


def test_export_native_names_blank_slices_and_sibling_layout(tmp_path):
    ds, raw = stack(tmp_path)
    labels = np.zeros_like(raw)
    labels[1, 5:8, 6:10] = 1
    before = {s.name: (ds.directory / s.name).read_bytes() for s in ds.slices}
    out = core.export_masks(ds, [core.LabelSource('parietal', labels)], core.Correction(0, 0, 0))
    assert out.parent == ds.directory.parent
    assert out != ds.directory
    assert (out / 'COMPLETE.txt').is_file()
    assert sorted(p.name for p in out.glob('*.bmp')) == sorted(before)
    for i, record in enumerate(ds.slices):
        core.verify_bmp(out / record.name, labels[i] > 0, record.ppm, 1)
        assert before[record.name] == (ds.directory / record.name).read_bytes()
    rows = list(csv.DictReader((out / 'slice_mapping.csv').open()))
    assert [int(r['foreground_pixels']) for r in rows] == [0, 12, 0]
    assert [r['source_filename'] for r in rows] == [s.name for s in ds.slices]


def test_separate_ids_and_layer_names(tmp_path):
    ds, raw = stack(tmp_path)
    labels = np.zeros_like(raw)
    labels[0, 5, 6] = 1
    labels[2, 8, 10] = 2
    out = core.export_masks(ds, [core.LabelSource('bone/name', labels)], core.Correction(0, 0, 0), 'ids')
    assert (out / '01_bone_name_id1').is_dir()
    assert (out / '01_bone_name_id2').is_dir()
    assert np.asarray(Image.open(out / '01_bone_name_id1' / 'scan10.bmp')).sum() == 0
    assert np.asarray(Image.open(out / '01_bone_name_id2' / 'scan10.bmp')).sum() == 1


def test_label_id_filter_and_union(tmp_path):
    ds, raw = stack(tmp_path)
    a, b = np.zeros_like(raw), np.zeros_like(raw)
    a[1, 3, 4] = 2
    a[1, 4, 4] = 1
    b[1, 7, 9] = 3
    out = core.export_masks(ds, [core.LabelSource('one', a, (2,)), core.LabelSource('two', b, (3,))],
                            core.Correction(0, 0, 0))
    mask = np.asarray(Image.open(out / 'scan9.bmp'))
    assert mask.sum() == 2 and mask[3, 4] and mask[7, 9]


def test_separate_duplicate_layer_names_are_distinct(tmp_path):
    ds, raw = stack(tmp_path)
    a = np.ones_like(raw)
    out = core.export_masks(ds, [core.LabelSource('bone', a), core.LabelSource('bone', a)], mode='layers')
    assert (out / '01_bone').is_dir() and (out / '02_bone').is_dir()


def test_cropped_labels_rejected(tmp_path):
    ds, raw = stack(tmp_path)
    with pytest.raises(core.ValidationError, match='explicit spatial mapping'):
        core.export_masks(ds, [core.LabelSource('wrong', raw[:, :-1])])


def test_source_change_blocks_export(tmp_path):
    ds, raw = stack(tmp_path)
    Image.fromarray(np.zeros_like(raw[0])).save(ds.directory / 'scan8.bmp')
    with pytest.raises(core.ValidationError, match='changed since loading'):
        core.export_masks(ds, [core.LabelSource('mask', np.ones_like(raw))])
    assert not list(tmp_path.glob('*masks*'))


def test_cancel_has_no_completed_or_partial_export(tmp_path):
    ds, raw = stack(tmp_path)
    cancel = threading.Event()
    def progress(done, total, message):
        if done == 1:
            cancel.set()
    with pytest.raises(core.Cancelled):
        core.export_masks(ds, [core.LabelSource('mask', np.ones_like(raw))], progress=progress, cancel=cancel)
    assert not list(tmp_path.glob('*masks*'))


def test_failed_write_cleans_export(tmp_path, monkeypatch):
    ds, raw = stack(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('simulated disk-full condition')
    monkeypatch.setattr(core, 'write_bmp', fail)
    with pytest.raises(OSError, match='disk-full'):
        core.export_masks(ds, [core.LabelSource('mask', np.ones_like(raw))])
    assert not list(tmp_path.glob('*masks*'))


def test_empty_export_is_rejected(tmp_path):
    ds, raw = stack(tmp_path)
    with pytest.raises(core.ValidationError, match='completely empty'):
        core.export_masks(ds, [core.LabelSource('empty', np.zeros_like(raw))])
    assert not list(tmp_path.glob('*masks*'))


def test_repeated_exports_do_not_overwrite(tmp_path):
    ds, raw = stack(tmp_path)
    source = core.LabelSource('mask', np.ones_like(raw))
    a, b = core.export_masks(ds, [source]), core.export_masks(ds, [source])
    assert a != b and a.is_dir() and b.is_dir()


def test_project_roundtrip_and_resume_without_mutating_saved_labels(tmp_path):
    ds, raw = stack(tmp_path)
    labels = (raw % 4).astype(np.uint16)
    path = core.save_project(ds, [core.LabelSource('parietal', labels)], core.Correction(2, -1, 3, 8), 'ids')
    path, meta, source = core.read_project(path)
    assert source == ds.directory
    restored = core.restore_labels(path, meta, ds, tmp_path / 'cache')
    assert restored[0].name == 'parietal'
    assert np.array_equal(restored[0].data, labels)
    restored[0].data[0, 0, 0] = 1000
    stored = np.load(path.parent / 'labels_001.npy', allow_pickle=False)
    assert stored[0, 0, 0] != 1000
    assert meta['settings'] == {'x_shift': 2, 'y_shift': -1, 'dilation': 3, 'bit_depth': 8}


def test_saved_label_corruption_rejected(tmp_path):
    ds, raw = stack(tmp_path)
    path = core.save_project(ds, [core.LabelSource('mask', raw)])
    path, meta, _ = core.read_project(path)
    damaged = np.load(path.parent / 'labels_001.npy', mmap_mode='r+')
    damaged[0, 1, 2] += 1
    damaged.flush()
    with pytest.raises(core.ValidationError, match='integrity'):
        core.restore_labels(path, meta, ds, tmp_path / 'cache')


def test_path_names_are_safe_on_windows():
    for source in ['CON', 'LPT1', 'nul.txt', '../../mask', '', 'a:b/c']:
        result = core.safe_name(source)
        assert '/' not in result and ':' not in result and result not in {'.', '..', ''}
        assert result.lower() not in {'con', 'lpt1', 'nul.txt'}


def test_mixed_case_bmp_names_and_blank_edges_follow_source(tmp_path):
    folder = tmp_path / "original"
    folder.mkdir()
    filenames = ["Mouse0041.BMP", "Mouse0042.BmP", "Mouse0043.bmp"]
    for name in filenames:
        Image.fromarray(np.zeros((17, 29), np.uint8)).save(folder / name)
    ds = core.inspect_dataset(folder)
    core.load_volume(ds)
    labels = np.zeros(ds.shape, np.uint8)
    labels[1, 8, 11] = 2
    session = {"local_checkpoint_folder": "lab model"}
    out = core.export_masks(ds, [core.LabelSource("nnInteractive result", labels)], session=session)
    assert {f.name for f in out.iterdir() if f.suffix.lower() == ".bmp"} == set(filenames)
    assert out.parent == folder.parent
    for name, pixels in zip(filenames, (0, 25, 0)):
        with Image.open(out / name) as bitmap:
            assert bitmap.size == (29, 17) and bitmap.mode == "1"
            assert np.count_nonzero(bitmap) == pixels
    assert json.loads((out / "export.json").read_text())["runtime"]["open_nninteractive_session"] == session
