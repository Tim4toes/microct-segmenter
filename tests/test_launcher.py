"""Exercise release transactions and the install commands without a GPU or live update server."""
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("microct_manager", Path(__file__).parents[1] / "portable" / "manage.py")
mgr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mgr)


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "local profile & spaces" / "MicroCTSegmenter"
    folder.mkdir(parents=True)
    return folder


def fake_install(root, ver, prepare=None):
    exe = root.parent / f"candidate-{ver}.exe"
    exe.write_bytes(b"test launcher " + ver.encode())
    def ready(root, slot, exe, release, bundle, log):
        python = mgr.python_in(slot)
        python.parent.mkdir(parents=True)
        python.write_bytes(b"test interpreter")
        (slot / mgr.EXE).write_bytes(exe.read_bytes())
        info = {"app": release["version"]}
        mgr.save_json(slot / "ready.json", info)
        return info
    return mgr.install(root, exe, root.parent, {"version": ver}, prepare or ready)


def test_first_install_update_and_rollback(root):
    project = root.parent / "study" / "project.json"
    project.parent.mkdir()
    project.write_bytes(b"original project")
    fake_install(root, "1.1.0")
    first = mgr.state(root)["active"]
    stable = (root / mgr.EXE).read_bytes()
    assert mgr.state(root)["previous"] is None
    assert b"--updates" in (root / "Update_MicroCT.cmd").read_bytes()
    mgr.save_json(root / "update-folder.json", "lab share")
    fake_install(root, "1.2.0")
    second = mgr.state(root)["active"]
    assert mgr.state(root)["previous"] == first
    assert mgr.rollback(root) == "1.1.0"
    assert mgr.state(root) == {"active": first, "previous": second, "skip_offer": "1.2.0"}
    assert (root / mgr.EXE).read_bytes() == stable
    assert mgr.read_json(root / "update-folder.json") == "lab share"
    assert project.read_bytes() == b"original project"
    assert mgr.rollback(root) == "1.2.0"
    fake_install(root, "1.3.0")
    assert mgr.state(root)["previous"] == second
    assert not (root / "r" / first).exists()
    assert mgr.installed(root, second)[1]["app"] == "1.2.0"


def test_incomplete_setup_and_downgrade_keep_active(root):
    fake_install(root, "1.1.0")
    before = mgr.state(root)
    def broken(root, slot, *_):
        (slot / "unfinished").write_bytes(b"partial environment")
        raise RuntimeError("health check failed")
    with pytest.raises(mgr.SetupError, match="health check failed"):
        fake_install(root, "1.2.0", broken)
    assert mgr.state(root) == before
    assert len(list((root / "r").iterdir())) == 1
    with pytest.raises(mgr.SetupError, match="already installed"):
        fake_install(root, "1.0.0")
    assert mgr.state(root) == before


def test_sessions_allow_two_readers_and_block_updates(root):
    with mgr.session_lock(root, exclusive=False), mgr.session_lock(root, exclusive=False):
        with pytest.raises(mgr.SetupError, match="Close open"):
            fake_install(root, "1.1.0")
    fake_install(root, "1.1.0")
    with mgr.session_lock(root, exclusive=True):
        with pytest.raises(mgr.SetupError):
            with mgr.session_lock(root, exclusive=False):
                pass


def archive(path, ver, exe=b"new version", sha=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("microct-release.json", json.dumps({"format": "microct-windows-v1", "version": ver,
                    "exe_sha256": sha or hashlib.sha256(exe).hexdigest()}))
        z.writestr(mgr.EXE, exe)


def test_update_from_zip_requires_consent_integrity_and_newer_version(root, tmp_path):
    fake_install(root, "1.1.0")
    z = tmp_path / "release.zip"
    archive(z, "1.2.0")
    invoked = []
    def run(argv):
        invoked.append((Path(argv[0]).read_bytes(), argv[1:]))
        return subprocess.CompletedProcess(argv, 0)
    assert not mgr.apply_zip(root, z, lambda m: False, run)
    assert not invoked
    assert mgr.apply_zip(root, z, lambda m: True, run)
    assert invoked == [(b"new version", ["--apply", "1.2.0"])]
    assert not list(root.glob("incoming-*"))
    archive(z, "1.2.0", sha="0" * 64)
    with pytest.raises(mgr.SetupError, match="checksum"):
        mgr.apply_zip(root, z, lambda m: True, run)
    archive(z, "1.1.0")
    with pytest.raises(mgr.SetupError, match="as recent"):
        mgr.apply_zip(root, z, lambda m: True, run)
    assert len(invoked) == 1


def test_share_discovers_newest_and_survives_disconnection(root, tmp_path):
    share = tmp_path / "lab"
    share.mkdir()
    def package(ver):
        path = share / f"MicroCT_Segmenter_v{ver}_Windows_x64.zip"
        path.touch()
        return path
    low, high = package("1.2.0"), package("1.10.0")
    package("1.0.0")
    (share / "unrelated.zip").touch()
    mgr.save_json(root / "update-folder.json", str(share))
    assert mgr.release_offer(root, "1.1.0", None) == high
    assert mgr.release_offer(root, "1.1.0", "1.10.0") == low
    mgr.save_json(root / "update-folder.json", str(share / "unavailable"))
    assert mgr.release_offer(root, "1.1.0", None) is None


def test_partial_download_resumes_and_verified_wheel_is_reused(root, monkeypatch):
    contents = b"abcdef"
    sha = hashlib.sha256(contents).hexdigest()
    filename = "pkg-1-py3-none-any.whl"
    partial = root / "wheels" / sha[:16] / (filename + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(contents[:3])
    class Response(BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6", "Content-Length": "3"}
    def stream(request, **kw):
        assert request.get_header("Range") == "bytes=3-"
        return Response(contents[3:])
    monkeypatch.setattr(mgr, "urlopen", stream)
    args = root, "https://example.invalid/wheel", filename, sha, lambda msg: None
    wheel = mgr.cached_wheel(*args)
    assert wheel.read_bytes() == contents
    monkeypatch.setattr(mgr, "urlopen", lambda *a, **k: pytest.fail("network used on cache hit"))
    assert mgr.cached_wheel(*args) == wheel


def test_invalid_download_is_rejected_before_install(root, monkeypatch):
    class Response(BytesIO):
        status = 200
        headers = {"Content-Length": "3"}
    monkeypatch.setattr(mgr, "urlopen", lambda *a, **k: Response(b"bad"))
    monkeypatch.setattr(mgr.time, "sleep", lambda _: None)
    with pytest.raises(mgr.SetupError, match="checksum"):
        mgr.cached_wheel(root, "https://example.invalid/wheel", "pkg.whl", "0" * 64, lambda _: None)
    assert not list(root.rglob("*.whl"))


def test_real_isolated_environment_and_offline_pip_install(root, tmp_path, monkeypatch):
    """Use the production venv, pip, check and activation steps with a small local wheel."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "windows-wheels.lock").write_text("")
    wheel = bundle / "microct_fixture-1.1.0-py3-none-any.whl"
    dist = "microct_fixture-1.1.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("microct_fixture.py", "VALUE = 17\n")
        z.writestr(dist + "/METADATA", "Metadata-Version: 2.1\nName: microct-fixture\nVersion: 1.1.0\n")
        z.writestr(dist + "/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(dist + "/RECORD", "\n".join(n+",," for n in (*z.namelist(),dist+"/RECORD"))+"\n")
    (bundle / "runtime_check.py").write_text(
        "import json,sys,microct_fixture\nfrom pathlib import Path\n"
        "from importlib.metadata import version\nassert microct_fixture.VALUE == 17\n"
        "Path(sys.argv[1]).write_text(json.dumps({'app':version('microct-fixture')}))\n")
    exe = tmp_path / mgr.EXE
    exe.write_bytes(b"test dispatcher")
    if sys.platform != "win32":
        monkeypatch.setattr(mgr, "python_in", lambda slot: slot / "env" / "bin" / "python")
    info = mgr.install(root, exe, bundle, {"version": "1.1.0", "app_wheel": wheel.name,
                                           "app_sha256": mgr.digest(wheel)})
    assert info["app"] == "1.1.0"
    assert mgr.installed(root, mgr.state(root)["active"])[1] == info


def test_manifest_checks_and_locked_windows_gpu_wheels():
    lock = Path(mgr.__file__).parent / "windows-wheels.lock"
    rows = list(mgr.wheels(lock))
    assert len({name for name, *_ in rows}) == len(rows)
    names = {name: filename for name, _, filename, _ in rows}
    assert {"napari", "napari-nninteractive", "nninteractive", "nninteractive-client", "pyqt6"} <= names.keys()
    assert names["torch"].endswith("+cu128-cp312-cp312-win_amd64.whl")
    assert names["torchvision"].endswith("+cu128-cp312-cp312-win_amd64.whl")
    for invalid in ("../1.2.0", "stable", None):
        with pytest.raises(mgr.SetupError):
            mgr.version(invalid)


def test_interrupted_install_is_cleaned_without_removing_neighbouring_data(root):
    abandoned = root / "r" / "1.9.0-deadbeef"
    abandoned.mkdir(parents=True)
    (abandoned / ".microct-managed").touch()
    (abandoned / "unfinished").write_text("partial")
    neighbour = root / "r" / "1.9.0-feedbeef"
    neighbour.mkdir()
    fake_install(root, "1.1.0")
    assert not abandoned.exists()
    assert neighbour.is_dir()
