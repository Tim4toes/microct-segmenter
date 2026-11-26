"""Build a self-contained Windows x64 bootstrap from a verified CPython runtime.

Usage: python portable/build_release.py --runtime python-windows.tar.gz --output dist
Build host needs Python 3.12+, pip, setuptools, wheel, Go and MinGW windres.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
PYTHON_TARBALL = "cpython-3.12.16+20261124-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
PYTHON_SHA = "96694b50dcfde13d56e40346ddaa552a9cd650290e9b933cecf83c1d82226960"
SOURCE_FILES = ["pyproject.toml", "launch_microct.py", "README.md", "START_HERE.txt", "Start_MicroCT.bat",
                "microct_segmenter/__init__.py", "microct_segmenter/core.py", "microct_segmenter/gui.py",
                "microct_segmenter/napari.yaml", "tests/test_core.py", "tests/test_gui.py", "tests/test_launcher.py",
                "portable/build_release.py", "portable/manage.py", "portable/runtime_check.py",
                "portable/windows-wheels.lock", "portable/bootstrap/main.go", "portable/bootstrap/dialog_windows.go",
                "portable/bootstrap/main_test.go", "portable/bootstrap/go.mod", "portable/bootstrap/launcher.rc", "portable/bootstrap/launcher.manifest"]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(runtime, output, go, windres):
    if sha(runtime) != PYTHON_SHA:
        raise ValueError(f"Runtime checksum mismatch. Use Astral's {PYTHON_TARBALL}")
    ver = tomllib.loads((SOURCE / "pyproject.toml").read_text())["project"]["version"]
    output.mkdir(parents=True, exist_ok=True)
    result = output / f"MicroCT_Segmenter_v{ver}_Windows_x64.zip"
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source = work / "source"
        for name in SOURCE_FILES:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, target)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--no-index", "-w", str(work), str(source)], check=True)
        wheel, = work.glob("*.whl")
        release = {"version": ver, "app_wheel": wheel.name, "app_sha256": sha(wheel),
                   "python_distribution": PYTHON_TARBALL, "python_sha256": PYTHON_SHA}
        boot = source / "portable" / "bootstrap"
        with zipfile.ZipFile(boot / "payload.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as payload:
            with tarfile.open(runtime) as archive:
                for member in archive:
                    if member.isdir():
                        continue
                    if not member.isfile() or not member.name.startswith("python/") or ".." in Path(member.name).parts:
                        raise ValueError(f"Unexpected runtime entry {member.name}")
                    with archive.extractfile(member) as file:
                        payload.writestr(member.name, file.read())
            for name in ["manage.py", "runtime_check.py", "windows-wheels.lock"]:
                payload.write(HERE / name, name)
            payload.write(wheel, wheel.name)
            payload.writestr("release.json", json.dumps(release, indent=2))

        (boot / "embedded.go").write_text(
            'package main\nimport _ "embed"\n//go:embed payload.zip\nvar embedded []byte\n'
            'func init() { payload = embedded }\n', encoding="utf-8")

        # Explicitly request normal user permissions in the native Windows executable.
        subprocess.run([windres, "--preprocessor=cpp", "-i", "launcher.rc", "-o", "launcher_windows_amd64.syso"], cwd=boot, check=True)
        exe = work / "MicroCT_Segmenter.exe"
        env = os.environ | {"GOOS": "windows", "GOARCH": "amd64", "CGO_ENABLED": "0", "GOTOOLCHAIN": "local"}
        subprocess.run([go, "build", "-trimpath", "-buildvcs=false", "-ldflags",
                        f"-s -w -X main.releaseVersion={ver}", "-o", str(exe), "."], cwd=boot, env=env, check=True)
        goroot = Path(subprocess.check_output([go, "env", "GOROOT"], text=True).strip())
        header = {"format": "microct-windows-v1", "version": ver, "exe_sha256": sha(exe)}
        tmpzip = work / result.name
        with zipfile.ZipFile(tmpzip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(exe, exe.name)
            archive.writestr("microct-release.json", json.dumps(header, indent=2))
            for name, flag in [("Update_MicroCT.cmd", "--updates"), ("Roll_Back_MicroCT.cmd", "--rollback")]:
                archive.writestr(name, f'@echo off\r\n"%~dp0{exe.name}" {flag}\r\n')
            for name in ["START_HERE.txt", "README.md", "VALIDATION.md", "Start_MicroCT.bat"]:
                archive.write(SOURCE / name, name)
            archive.write(goroot / "LICENSE", "notices/Go-LICENSE.txt")
            with tarfile.open(runtime) as rt:
                archive.writestr("notices/Python-LICENSE.txt", rt.extractfile("python/LICENSE.txt").read())
            for name in SOURCE_FILES:
                archive.write(SOURCE / name, "source/" + name)
            archive.write(SOURCE / "VALIDATION.md", "source/VALIDATION.md")
        # Copy and replace only once the complete archive and its CRCs have been checked.
        with zipfile.ZipFile(tmpzip) as archive:
            if archive.testzip() is not None:
                raise ValueError("Release ZIP verification failed")
        staged = output / (result.name + ".partial")
        shutil.copyfile(tmpzip, staged)
        os.replace(staged, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help=PYTHON_TARBALL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go", default="go")
    parser.add_argument("--windres", default="x86_64-w64-mingw32-windres")
    options = parser.parse_args()
    build(options.runtime.resolve(), options.output.resolve(), options.go, options.windres)
