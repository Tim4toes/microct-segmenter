"""User-local Windows installer, launcher and release updates. Uses only the standard library."""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
from http.client import IncompleteRead
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

EXE = "MicroCT_Segmenter.exe"
TITLE = "MicroCT Segmenter"
BUNDLE = Path(__file__).resolve().parent
ZIP_PATTERN = re.compile(r"MicroCT_Segmenter_v(\d+\.\d+\.\d+)_Windows_x64(?:\s*\(\d+\))?\.zip", re.I)
VERSION = re.compile(r"\d+\.\d+\.\d+")
SLOT = re.compile(r"\d+\.\d+\.\d+-[0-9a-f]{8}")
BLOCK = 1024 * 1024


class SetupError(Exception):
    pass


def version(value):
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise SetupError("Unrecognised release version.")
    return tuple(map(int, value.split(".")))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    # Write the active and rollback pointers together, including after a power loss mid-write.
    temporary = path.with_name(path.name + "-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def state(root):
    path = root / "versions.json"
    data = read_json(path) if path.exists() else {"active": None, "previous": None}
    for key in ("active", "previous"):
        value = data[key]
        if value is not None and (not isinstance(value, str) or not SLOT.fullmatch(value)):
            raise SetupError(f"Invalid installed version record in {path}")
    return data


def python_in(slot):
    return slot / "env" / "Scripts" / "python.exe"


def installed(root, slot_id):
    if not slot_id or not SLOT.fullmatch(slot_id):
        raise SetupError("No installed version is available.")
    slot = root / "r" / slot_id
    info = read_json(slot / "ready.json")
    if info["app"] != slot_id.split("-")[0] or not python_in(slot).is_file() or not (slot / EXE).is_file():
        raise SetupError(f"Incomplete installation at {slot}")
    return slot, info


@contextlib.contextmanager
def session_lock(root, *, exclusive):
    """Many open viewers, or one install/rollback. Windows releases the lock on process exit."""
    with (root / "sessions.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            from ctypes import wintypes

            class Overlapped(ctypes.Structure):
                _fields_ = [("internal", ctypes.c_size_t), ("internal_high", ctypes.c_size_t),
                            ("offset", wintypes.DWORD), ("offset_high", wintypes.DWORD),
                            ("event", wintypes.HANDLE)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            lock = kernel.LockFileEx
            lock.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                             wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
            lock.restype = wintypes.BOOL
            ov = Overlapped()
            # FAIL_IMMEDIATELY, with EXCLUSIVE_LOCK only for maintenance.
            if not lock(msvcrt.get_osfhandle(handle.fileno()), 1 | (2 if exclusive else 0), 0, 1, 0, ctypes.byref(ov)):
                error = ctypes.get_last_error()
                if error != 33:  # ERROR_LOCK_VIOLATION
                    raise ctypes.WinError(error)
                raise SetupError("Close open MicroCT Segmenter sessions and let any installation finish, then try again.")
        else:
            import fcntl
            try:
                fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SetupError("Close open MicroCT Segmenter sessions and let any installation finish, then try again.") from exc
        yield


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheels(lockfile):
    for line in lockfile.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, source = line.split(" @ ", 1)
        url, sha = source.split(" --hash=sha256:")
        url_path = urlsplit(url)
        filename = unquote(url_path.path.rsplit("/", 1)[-1])
        if (url_path.scheme != "https" or not re.fullmatch(r"[\w.+-]+\.whl", filename)
                or not re.fullmatch(r"[a-f0-9]{64}", sha)):
            raise SetupError(f"Invalid locked wheel for {name}")
        yield name, url, filename, sha


def cached_wheel(root, url, filename, sha, output):
    directory = root / "wheels" / sha[:16]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.is_file() and digest(target) == sha:
        return target
    partial = directory / (filename + ".part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "MicroCT-Segmenter", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        partial.unlink(missing_ok=True)
                        raise OSError("Unexpected range in download response.")
                else:
                    offset = 0
                length = int(response.headers.get("Content-Length", "0"))
                total = length + offset if length else 0
                with partial.open("ab" if offset else "wb") as file:
                    done, last_print = offset, time.monotonic()
                    while block := response.read(BLOCK):
                        file.write(block)
                        done += len(block)
                        if time.monotonic() - last_print > 5:
                            suffix = f" / {total / 2**20:,.0f} MiB" if total else " MiB"
                            output(f"    {done / 2**20:,.0f}{suffix}")
                            last_print = time.monotonic()
                if total and done != total:
                    raise OSError("The download was interrupted.")
            if digest(partial) != sha:
                partial.unlink()
                raise OSError("Download checksum mismatch.")
            os.replace(partial, target)
            return target
        except (OSError, EOFError, IncompleteRead) as exc:
            if getattr(exc, "code", None) == 416:
                # A complete partial file may be left if the process closed before checking its hash.
                if partial.exists() and digest(partial) == sha:
                    os.replace(partial, target)
                    return target
                partial.unlink(missing_ok=True)
            if attempt == 2:
                raise SetupError(f"Download failed for {filename}. Run setup again to resume.\n{exc}") from exc
            output(f"    Retrying download ({exc})")
            time.sleep(attempt + 1)


def process_env(slot=None):
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "QT_PLUGIN_PATH",
                 "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH", "CONDA_PREFIX", "TCL_LIBRARY", "TK_LIBRARY"):
        env.pop(name, None)
    env["QT_API"] = "pyqt6"
    env["PYTHONNOUSERSITE"] = "1"
    if slot is not None:
        env["VIRTUAL_ENV"] = str(slot / "env")
        qt_bin = slot / "env" / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"
        env["PATH"] = os.pathsep.join((str(python_in(slot).parent), str(qt_bin), env.get("PATH", "")))
    return env


def prepare(root, slot, exe, release, bundle, log):
    def output(message):
        print(message, flush=True)
        print(message, file=log, flush=True)

    def run(args, label, env):
        output(label)
        result = subprocess.run(args, env=env, cwd=slot, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SetupError(f"{label} failed (exit {result.returncode}). See the setup log.")

    lock = list(wheels(bundle / "windows-wheels.lock"))
    local = []
    output(f"Preparing {len(lock)} packages. Already verified downloads can be reused.")
    for i, (name, url, filename, sha) in enumerate(lock, 1):
        output(f"[{i}/{len(lock)}] {filename}")
        path = cached_wheel(root, url, filename, sha, output)
        local.append(f"{path.as_uri()} --hash=sha256:{sha}")
    wheel = bundle / release["app_wheel"]
    if digest(wheel) != release["app_sha256"]:
        raise SetupError("The bundled application wheel failed its checksum.")
    local.append(f"{wheel.as_uri()} --hash=sha256:{release['app_sha256']}")
    requirements = slot / "install.lock"
    requirements.write_text("\n".join(local) + "\n", encoding="utf-8")
    # Create the environment at its permanent path. Windows venv scripts contain absolute paths.
    run([sys.executable, "-I", "-m", "venv", str(slot / "env")], "Creating the private environment", process_env())
    python = python_in(slot)
    env = process_env(slot)
    pip = [python, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check"]
    run([*pip, "install", "--no-index", "--no-cache-dir", "--no-compile", "--no-deps", "--only-binary=:all:",
         "--require-hashes", "-r", requirements], "Installing the verified packages", env)
    run([*pip, "check"], "Checking package compatibility", env)
    run([python, "-I", bundle / "runtime_check.py", slot / "check.json"], "Checking napari, Qt and nnInteractive", env)
    info = read_json(slot / "check.json")
    if info["app"] != release["version"]:
        raise SetupError("The installed application version differs from the release.")
    shutil.copy2(exe, slot / EXE)
    save_json(slot / "ready.json", info)
    return info


def install(root, exe, bundle, release, prepare_slot=prepare):
    with session_lock(root, exclusive=True):
        old = state(root)
        active = old["active"]
        if active and version(active.split("-")[0]) >= version(release["version"]):
            raise SetupError("This release is already installed or an installed release is newer.")
        versions = root / "r"
        versions.mkdir(exist_ok=True)
        # A Windows shutdown can interrupt pip before the normal exception cleanup.
        # Keep both published slots. Only remove directories marked by this installer.
        for abandoned in versions.iterdir():
            if (SLOT.fullmatch(abandoned.name) and abandoned.name not in (active, old["previous"])
                    and not abandoned.is_symlink() and (abandoned / ".microct-managed").is_file()):
                shutil.rmtree(abandoned, ignore_errors=True)
        slot_id = f"{release['version']}-{uuid.uuid4().hex[:8]}"
        slot = versions / slot_id
        slot.mkdir()
        (slot / ".microct-managed").touch()
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        log_path = logs / f"setup-{slot_id}.log"
        try:
            with log_path.open("w", encoding="utf-8", errors="backslashreplace") as log:
                info = prepare_slot(root, slot, exe, release, bundle, log)
            stable = root / EXE
            if not stable.exists():
                temp = root / ("launcher-" + uuid.uuid4().hex + ".tmp")
                try:
                    shutil.copy2(exe, temp)
                    os.replace(temp, stable)
                finally:
                    temp.unlink(missing_ok=True)
            for filename, action in [("Start_MicroCT.bat", ""), ("Update_MicroCT.cmd", "--updates"), ("Roll_Back_MicroCT.cmd", "--rollback")]:
                # Static commands continue to target the installed launcher after later updates.
                (root / filename).write_bytes((f'@echo off\r\n"%~dp0{EXE}" {action}\r\n').encode("ascii"))
            save_json(root / "versions.json", {"active": slot_id, "previous": active})
        except BaseException as exc:
            shutil.rmtree(slot, ignore_errors=True)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise SetupError(f"{exc}\n\nSetup log\n{log_path}") from exc
        # Only prune a known managed release after its replacement is ready and active.
        obsolete = old["previous"]
        if obsolete and obsolete not in (slot_id, active):
            shutil.rmtree(root / "r" / obsolete, ignore_errors=True)
        return info


def rollback(root):
    with session_lock(root, exclusive=True):
        old = state(root)
        if not old["previous"]:
            raise SetupError("A previous version will be available after your first successful update.")
        _, info = installed(root, old["previous"])
        save_json(root / "versions.json", {"active": old["previous"], "previous": old["active"],
                                          "skip_offer": old["active"].split("-")[0]})
        return info["app"]


def release_header(archive):
    header = archive.getinfo("microct-release.json")
    if header.file_size > 8192:
        raise SetupError("Invalid Windows release metadata.")
    data = json.loads(archive.read(header))
    version(data.get("version"))
    if data.get("format") != "microct-windows-v1" or not re.fullmatch(r"[0-9a-f]{64}", data.get("exe_sha256", "")):
        raise SetupError("Choose a MicroCT Segmenter Windows release ZIP.")
    return data


def apply_zip(root, path, confirm, runner=subprocess.run):
    with zipfile.ZipFile(path) as archive:
        data = release_header(archive)
        active = state(root)["active"]
        if active and version(data["version"]) <= version(active.split("-")[0]):
            raise SetupError(f"Installed version {active.split('-')[0]} is already as recent as this ZIP ({data['version']}).")
        if not confirm(f"Install version {data['version']} from this release?\n\n{path}"):
            return False
        with tempfile.TemporaryDirectory(prefix="incoming-", dir=root) as temp:
            candidate = Path(temp) / EXE
            with archive.open(EXE) as source, candidate.open("xb") as target:
                shutil.copyfileobj(source, target, BLOCK)
            if digest(candidate) != data["exe_sha256"]:
                raise SetupError("The launcher checksum does not match the release ZIP.")
            result = runner([str(candidate), "--apply", data["version"]])
            if result.returncode:
                raise SetupError("The update did not finish. The setup window and log give the reason.")
    return True


def newest_in(folder, current, skipped=None):
    found = []
    for path in Path(folder).iterdir():
        match = ZIP_PATTERN.fullmatch(path.name)
        if match and match[1] != skipped and version(match[1]) > version(current) and path.is_file():
            found.append((version(match[1]), path))
    return max(found)[1] if found else None


def release_offer(root, active, skipped):
    """A disconnected network drive must never hold up ordinary startup."""
    config = root / "update-folder.json"
    if not config.is_file():
        return None
    found = []
    def scan():
        try:
            folder = read_json(config)
            if folder:
                found.append(newest_in(folder, active, skipped))
        except (OSError, ValueError, TypeError):
            pass
    thread = threading.Thread(target=scan, daemon=True)
    thread.start()
    thread.join(0.75)
    return found[0] if found else None


def ask(message):
    from tkinter import messagebox
    return messagebox.askyesno(TITLE, message)


def info(message):
    from tkinter import messagebox
    messagebox.showinfo(TITLE, message)


def maintenance(root):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Tk()
    window.title("MicroCT Segmenter updates")
    window.resizable(True, False)
    body = ttk.Frame(window, padding=16)
    body.pack(fill="both", expand=True)
    details = tk.StringVar()
    folder_text = tk.StringVar()
    config = root / "update-folder.json"

    def refresh():
        st = state(root)
        current = st["active"].split("-")[0]
        previous = st["previous"].split("-")[0] if st["previous"] else "none yet"
        details.set(f"Installed {current}    |    Previous {previous}")
        dest = read_json(config) if config.is_file() else None
        folder_text.set(dest or "Choose a folder to check for future releases at startup.")
        check_button.configure(state="normal" if dest else "disabled")
        clear_button.configure(state="normal" if dest else "disabled")
        rollback_button.configure(state="normal" if st["previous"] else "disabled")

    def guarded(fn):
        def action():
            try:
                fn()
                refresh()
            except Exception as exc:
                messagebox.showerror(TITLE, str(exc), parent=window)
        return action

    def select_zip():
        path = filedialog.askopenfilename(parent=window, title="Select a MicroCT Segmenter release ZIP",
                                          filetypes=[("MicroCT Segmenter Windows release", "*.zip")])
        if path and apply_zip(root, Path(path), ask):
            info("Update installed. Open MicroCT Segmenter to use it.")

    def set_folder():
        path = filedialog.askdirectory(parent=window, title="Choose your trusted lab release folder")
        if path:
            save_json(config, path)

    def check():
        current = state(root)["active"].split("-")[0]
        path = newest_in(read_json(config), current)
        if path:
            if apply_zip(root, path, ask):
                info("Update installed.")
        else:
            info(f"This folder contains no release newer than {current}.")

    def revert():
        if ask("Switch this PC back to its previous installed version?"):
            info(f"Version {rollback(root)} is now active.")

    def button(text, fn):
        b = ttk.Button(body, text=text, command=guarded(fn))
        b.pack(fill="x", pady=3)
        return b

    ttk.Label(body, textvariable=details).pack(anchor="w", pady=(0, 7))
    button("Install from a release ZIP…", select_zip)
    ttk.Separator(body).pack(fill="x", pady=8)
    ttk.Label(body, text="Shared release folder").pack(anchor="w")
    ttk.Label(body, textvariable=folder_text, wraplength=460).pack(anchor="w", pady=3)
    button("Choose folder…", set_folder)
    check_button = button("Check folder now", check)
    clear_button = button("Stop checking this folder", lambda: save_json(config, None))
    ttk.Separator(body).pack(fill="x", pady=8)
    rollback_button = button("Use previous version", revert)
    ttk.Button(body, text="Close", command=window.destroy).pack(anchor="e", pady=(8, 0))
    refresh()
    window.mainloop()


def main(root, exe, args):
    release = read_json(BUNDLE / "release.json")
    own = release["version"]
    version(own)
    root.mkdir(parents=True, exist_ok=True)
    mode = args[0] if args and args[0].startswith("--") else "--start"
    rest = args[1:] if args and args[0].startswith("--") else args
    if mode == "--version":
        print(f"MicroCT Segmenter Windows launcher {own}")
        return 0
    if mode == "--apply":
        if rest != [own]:
            raise SetupError("The launcher version differs from the selected release.")
        install(root, exe, BUNDLE, release)
        return 0
    if mode not in ("--start", "--start-installed", "--updates", "--rollback") or len(rest) > (1 if mode.startswith("--start") else 0):
        raise SetupError("Use MicroCT_Segmenter.exe [dataset], --updates, --rollback or --version.")
    old = state(root)
    current = old["active"].split("-")[0] if old["active"] else None
    if current is None or (mode == "--start" and not exe.is_relative_to(root) and version(own) > version(current)):
        if ask(f"Install MicroCT Segmenter {own} for this Windows account?\n\nLocation: {root}\n\n"
               "First setup downloads about 3.8 GB for napari and local nnInteractive. Allow about 15 GB free for setup. Follow progress in the console."):
            report = install(root, exe, BUNDLE, release)
            text = f"Installed version {own}. You can run the app here:\n{root / EXE}"
            if not report.get("gpu"):
                text += "\n\nLocal CUDA inference is unavailable on this PC. Viewing and mask export are available."
            info(text)
        elif current is None:
            return 0
    st = state(root)
    slot, active = installed(root, st["active"])
    if own != active["app"]:
        return subprocess.call([slot / EXE, mode, *rest])
    if mode == "--updates":
        maintenance(root)
        return 0
    if mode == "--rollback":
        if ask("Switch this PC to its previous installed version?"):
            info(f"Version {rollback(root)} is now active.")
        return 0
    if mode == "--start":
        offer = release_offer(root, active["app"], st.get("skip_offer"))
        if offer:
            try:
                if apply_zip(root, offer, ask):
                    new, _ = installed(root, state(root)["active"])
                    return subprocess.call([new / EXE, "--start-installed", *rest])
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, SetupError) as exc:
                info(f"Continuing with the installed release.\n\n{exc}")
    with session_lock(root, exclusive=False):
        # Re-read under the run lock if another launcher switched versions during the offer.
        slot, _ = installed(root, state(root)["active"])
        dataset = [str(Path(rest[0]).resolve())] if rest else []
        code = subprocess.call([python_in(slot), "-I", "-m", "microct_segmenter.gui", *dataset],
                               env=process_env(slot), cwd=root)
        if code:
            raise SetupError(f"napari exited with code {code}. See the console for the application error.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]).resolve(), sys.argv[3:]))
    except Exception as error:
        traceback.print_exc()
        from tkinter import messagebox
        messagebox.showerror(TITLE, str(error))
        sys.exit(1)
