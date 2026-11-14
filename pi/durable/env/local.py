"""Local execution environment (port of ``env/node.ts``).

A filesystem and shell over the host's files, implemented with ``os``/``io`` and
``asyncio`` subprocesses. Every local environment sees the same files: ``id`` is
``"local"``.

Porting deviations from the Node implementation:
- positional reads on one reader serialize through a lock (``seek`` + ``read``
  share the handle's cursor, unlike Node's position reads);
- ``file://`` paths are not resolved;
- the legacy WSL ``bash -s`` stdin transport is kept;
- watching defaults to ``native`` on every platform, where Node polls on
  Windows: the watcher holds no handle on a watched target itself there (see
  ``watch.py``), so renaming watched paths and their parents keeps working.
"""

from __future__ import annotations

import asyncio
import errno as errno_module
import math
import os
import shutil
import signal
import stat as stat_module
import subprocess as subprocess_module
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from . import (
    BinaryReader,
    Context,
    DirReader,
    ExecutionError,
    FileError,
    FileInfo,
    LineScan,
    Result,
    ShellExecOptions,
    ShellExecResult,
    ShellOutputInfo,
    ShellOutputSkip,
    StreamDecoder,
    TextLine,
    WatchTarget,
    err,
    ok,
)
from .line_scan import LineScanner
from .watch import DEFAULT_MAX_DIRECTORIES, DEFAULT_POLL_INTERVAL_MS, LocalFileWatcher

MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000
EXIT_STDIO_GRACE_MS = 100
SPILL_BUFFER_LIMIT = 1024 * 1024
# Largest single read of a reader, so a huge ``length`` allocates only as much as the file yields.
BINARY_READ_CHUNK = 1024 * 1024
READ_CHUNK = 64 * 1024

WINDOWS = sys.platform == "win32"

_ERRNO_CODES = {
    errno_module.ENOENT: "not_found",
    errno_module.EACCES: "permission_denied",
    errno_module.EPERM: "permission_denied",
    errno_module.ENOTDIR: "not_directory",
    errno_module.EISDIR: "is_directory",
    errno_module.EINVAL: "invalid",
    errno_module.ELOOP: "invalid",
}


def resolve_path(cwd: str, path: str) -> str:
    normalized = path
    if normalized == "~":
        normalized = os.path.expanduser("~")
    elif normalized.startswith("~/") or (WINDOWS and normalized.startswith("~\\")):
        normalized = os.path.join(os.path.expanduser("~"), normalized[2:])
    return os.path.normpath(normalized if os.path.isabs(normalized) else os.path.join(cwd, normalized))


def _kind_of(stat: os.stat_result) -> Optional[str]:
    if stat_module.S_ISREG(stat.st_mode):
        return "file"
    if stat_module.S_ISDIR(stat.st_mode):
        return "directory"
    if stat_module.S_ISLNK(stat.st_mode):
        return "symlink"
    return None


def file_info_from_stat(path: str, stat: os.stat_result) -> Result:
    kind = _kind_of(stat)
    if kind is None:
        return err(FileError("invalid", "Unsupported file type", path))
    return ok(
        FileInfo(
            name=os.path.basename(path),
            path=path,
            kind=kind,
            size=stat.st_size,
            mtime_ms=stat.st_mtime * 1000,
        )
    )


def to_file_error(error: BaseException, fallback_path: Optional[str] = None) -> FileError:
    if isinstance(error, FileError):
        return error
    if isinstance(error, OSError):
        code = _ERRNO_CODES.get(error.errno)
        if code is not None:
            return FileError(code, str(error), getattr(error, "filename", None) or fallback_path, error)
    return FileError("unknown", str(error), fallback_path, error)


def _aborted_result(context: Context, path: Optional[str] = None) -> Optional[Result]:
    if context.aborted:
        return err(FileError("aborted", "aborted", path))
    return None


def _closed_result(what: str, path: str) -> Result:
    return err(FileError("invalid", f"{what} is closed", path))


def _symlink_refused(path: str, cause: Optional[BaseException] = None) -> FileError:
    return FileError("invalid", "Refusing to follow a symbolic link", path, cause)


if WINDOWS:
    import ctypes
    import msvcrt

    _WIN32_ERRNOS = {
        2: errno_module.ENOENT,  # ERROR_FILE_NOT_FOUND
        3: errno_module.ENOENT,  # ERROR_PATH_NOT_FOUND
        5: errno_module.EACCES,  # ERROR_ACCESS_DENIED
        32: errno_module.EACCES,  # ERROR_SHARING_VIOLATION
    }
    _kernel32 = ctypes.windll.kernel32
    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    def _windows_shared_open(path: str) -> int:
        """Open a file for reading with full sharing, so renaming or removing it while it is open works."""
        handle = _kernel32.CreateFileW(
            path,
            0x80000000,  # GENERIC_READ
            0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
            None,
            3,  # OPEN_EXISTING
            0x80,  # FILE_ATTRIBUTE_NORMAL
            None,
        )
        if handle in (None, _INVALID_HANDLE):
            winerror = ctypes.GetLastError()
            raise OSError(_WIN32_ERRNOS.get(winerror, errno_module.EIO), f"CreateFileW failed ({winerror})")
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))

else:

    def _windows_shared_open(path: str) -> int:  # pragma: no cover - Windows only
        raise NotImplementedError


class LocalTextLineReader:
    """Strict LF reader over one opened file."""

    def __init__(self, handle: Any, path: str) -> None:
        self._handle = handle
        self._path = path
        self._decoder = StreamDecoder()
        self._buffered = ""
        self._ended = False
        self._closed = False
        self._lock = asyncio.Lock()

    async def read_line(self, context: Context) -> Result:
        aborted = _aborted_result(context, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return err(FileError("invalid", "Text line reader is closed", self._path))
        async with self._lock:
            while True:
                newline = self._buffered.find("\n")
                if newline != -1:
                    text = self._buffered[:newline]
                    self._buffered = self._buffered[newline + 1:]
                    return ok(TextLine(text=text, terminated=True))
                if self._ended:
                    if len(self._buffered) == 0:
                        return ok(None)
                    text = self._buffered
                    self._buffered = ""
                    return ok(TextLine(text=text, terminated=False))
                data = self._handle.read(READ_CHUNK)
                after = _aborted_result(context, self._path)
                if after is not None:
                    return after
                if len(data) == 0:
                    self._buffered += self._decoder.decode()
                    self._ended = True
                else:
                    self._buffered += self._decoder.decode(data)

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffered = ""
        try:
            self._handle.close()
        except OSError:
            pass


class LocalBinaryReader:
    """Positional reads from one opened regular file, serialized through a lock."""

    def __init__(self, handle: Any, path: str) -> None:
        self._handle = handle
        self._path = path
        self._closed = False
        self._lock = asyncio.Lock()

    async def info(self, context: Context) -> Result:
        aborted = _aborted_result(context, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return _closed_result("Binary reader", self._path)
        try:
            return file_info_from_stat(self._path, os.fstat(self._handle.fileno()))
        except OSError as exc:
            return err(to_file_error(exc, self._path))

    async def read(self, offset: int, length: int, context: Context) -> Result:
        aborted = _aborted_result(context, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return _closed_result("Binary reader", self._path)
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or not isinstance(length, int)
            or isinstance(length, bool)
            or offset < 0
            or length < 0
        ):
            return err(FileError("invalid", "Offset and length must be non-negative safe integers", self._path))
        chunks: List[bytes] = []
        total = 0
        try:
            async with self._lock:
                while total < length:
                    size = min(length - total, BINARY_READ_CHUNK)
                    self._handle.seek(offset + total)
                    data = self._handle.read(size)
                    after = _aborted_result(context, self._path)
                    if after is not None:
                        return after
                    if not data:
                        break
                    chunks.append(data)
                    total += len(data)
        except OSError as exc:
            return err(to_file_error(exc, self._path))
        if len(chunks) == 1:
            return ok(chunks[0])
        return ok(b"".join(chunks))

    async def scan_lines(self, options: Dict[str, int], context: Context) -> Result:
        aborted = _aborted_result(context, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return _closed_result("Binary reader", self._path)
        try:
            scanner = LineScanner(options["start_line"], options.get("end_line"))
        except (KeyError, TypeError, ValueError):
            return err(FileError("invalid", "Invalid line range", self._path))
        try:
            async with self._lock:
                position = 0
                while True:
                    self._handle.seek(position)
                    data = self._handle.read(READ_CHUNK)
                    after = _aborted_result(context, self._path)
                    if after is not None:
                        return after
                    if not data:
                        return ok(scanner.finish())
                    scanner.push(data)
                    position += len(data)
        except OSError as exc:
            return err(to_file_error(exc, self._path))

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.close()
        except OSError:
            pass


class LocalDirReader:
    """Pages of one directory's entries in file system order."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._entries = os.scandir(path)
        self._done = False
        self._closed = False

    async def next(self, max_entries: int, context: Context) -> Result:
        aborted = _aborted_result(context, self._path)
        if aborted is not None:
            return aborted
        if self._closed:
            return _closed_result("Directory reader", self._path)
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries <= 0:
            return err(FileError("invalid", "maxEntries must be a positive safe integer", self._path))
        infos: List[FileInfo] = []
        try:
            while not self._done and len(infos) < max_entries:
                try:
                    entry = next(self._entries)
                except StopIteration:
                    self._done = True
                    break
                after = _aborted_result(context, self._path)
                if after is not None:
                    return after
                entry_path = os.path.join(self._path, entry.name)
                try:
                    stat = os.lstat(entry_path)
                except FileNotFoundError:
                    continue  # Removed between enumeration and lstat: not part of the listing any more.
                info = file_info_from_stat(entry_path, stat)
                if info.ok:
                    infos.append(info.value)
            return ok({"entries": infos, "done": self._done})
        except OSError as exc:
            return err(to_file_error(exc, self._path))

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._entries.close()
        except OSError:
            pass


class ShellConfig:
    def __init__(self, shell: str, args: List[str], command_transport: Optional[str] = None) -> None:
        self.shell = shell
        self.args = args
        self.command_transport = command_transport  # None | "stdin"


def _is_legacy_wsl_bash(path: str) -> bool:
    import re

    normalized = path.replace("/", "\\").lower()
    return bool(re.match(r"^[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe$", normalized))


async def find_bash_on_path() -> Optional[str]:
    return shutil.which("bash.exe" if WINDOWS else "bash")


async def get_shell_config(custom_shell_path: Optional[str]) -> Result:
    if custom_shell_path:
        if os.path.exists(custom_shell_path):
            return ok(_bash_shell_config(custom_shell_path))
        return err(ExecutionError("shell_unavailable", f"Custom shell path not found: {custom_shell_path}"))
    if WINDOWS:
        candidates: List[str] = []
        program_files = os.environ.get("ProgramFiles")
        if program_files:
            candidates.append(os.path.join(program_files, "Git", "bin", "bash.exe"))
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        if program_files_x86:
            candidates.append(os.path.join(program_files_x86, "Git", "bin", "bash.exe"))
        # A Git on PATH at <root>\cmd\git.exe ships bash at <root>\bin\bash.exe.
        git_on_path = shutil.which("git.exe")
        if git_on_path:
            git_root = os.path.dirname(os.path.dirname(os.path.abspath(git_on_path)))
            candidates.append(os.path.join(git_root, "bin", "bash.exe"))
        for candidate in candidates:
            if os.path.exists(candidate):
                return ok(_bash_shell_config(candidate))
        bash_on_path = await find_bash_on_path()
        if bash_on_path:
            return ok(_bash_shell_config(bash_on_path))
        searched = "\n".join(f"  {path}" for path in candidates)
        return err(
            ExecutionError(
                "shell_unavailable",
                "No bash shell found. Options:\n"
                "  1. Install Git for Windows: https://git-scm.com/download/win\n"
                "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
                "  3. Configure an explicit shellPath\n\n"
                f"Searched Git Bash in:\n{searched}",
            )
        )
    if os.path.exists("/bin/bash"):
        return ok(_bash_shell_config("/bin/bash"))
    bash_on_path = await find_bash_on_path()
    if bash_on_path:
        return ok(_bash_shell_config(bash_on_path))
    return ok(ShellConfig("sh", ["-c"]))


def _bash_shell_config(shell: str) -> ShellConfig:
    if _is_legacy_wsl_bash(shell):
        return ShellConfig(shell, ["-s"], command_transport="stdin")
    return ShellConfig(shell, ["-c"])


def _shell_env(
    shell_env: Optional[Dict[str, str]], extra_env: Optional[Dict[str, str]], inherit_env: bool
) -> Dict[str, str]:
    if not inherit_env:
        return dict(extra_env or {})
    merged = dict(os.environ)
    if shell_env:
        merged.update(shell_env)
    if extra_env:
        merged.update(extra_env)
    return merged


async def kill_process_tree(pid: int) -> None:
    if WINDOWS:
        try:
            taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
            proc = await asyncio.create_subprocess_exec(
                taskkill,
                "/F",
                "/T",
                "/PID",
                str(pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except (OSError, ProcessLookupError):
            pass
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


class LocalExecutionEnv:
    """A filesystem and shell over the host's files."""

    id = "local"

    def __init__(
        self,
        cwd: str,
        shell_path: Optional[str] = None,
        shell_env: Optional[Dict[str, str]] = None,
        watch_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        watch_mode: str = "auto",  # "auto" | "native" | "polling"
        watch_max_directories: int = DEFAULT_MAX_DIRECTORIES,
    ) -> None:
        self.cwd = cwd
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._watch_interval_ms = watch_interval_ms
        self._watch_mode = watch_mode
        self._watch_max_directories = watch_max_directories
        self._active_children: set = set()

    # -- paths ----------------------------------------------------------------

    async def absolute_path(self, path: str, context: Context) -> Result:
        return ok(resolve_path(self.cwd, path))

    async def join_path(self, parts: Sequence[str], context: Context) -> Result:
        return ok(os.path.join(*parts))

    # -- watchers -------------------------------------------------------------

    async def watch(self, targets: Sequence[WatchTarget], on_change, context: Context) -> Result:
        aborted = _aborted_result(context)
        if aborted is not None:
            return aborted
        try:
            watcher = await LocalFileWatcher.open(
                list(targets),
                lambda path: resolve_path(self.cwd, path),
                on_change,
                mode=self._watch_mode,
                poll_interval_ms=self._watch_interval_ms,
                max_directories=self._watch_max_directories,
            )
        except Exception as exc:  # noqa: BLE001
            return err(to_file_error(exc))
        if context.aborted:
            await watcher.close(context)
            return _aborted_result(context)
        return ok(watcher)

    # -- reads ----------------------------------------------------------------

    async def open_text_line_reader(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            if WINDOWS:
                handle = open(_windows_shared_open(resolved), "rb", closefd=True)
            else:
                handle = open(resolved, "rb")
        except OSError as exc:
            return err(to_file_error(exc, resolved))
        if context.aborted:
            handle.close()
            return _aborted_result(context, resolved)
        return ok(LocalTextLineReader(handle, resolved))

    async def read_text_file(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            with open(resolved, "rb") as handle:
                return ok(handle.read().decode("utf-8", "replace"))
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def read_text_lines(self, path: str, options: Optional[Dict[str, int]], context: Context) -> Result:
        if options is not None and options.get("max_lines") is not None and options["max_lines"] <= 0:
            return ok([])
        opened = await self.open_text_line_reader(path, context)
        if not opened.ok:
            return opened
        reader: LocalTextLineReader = opened.value
        lines: List[str] = []
        try:
            while options is None or options.get("max_lines") is None or len(lines) < options["max_lines"]:
                line = await reader.read_line(context)
                if not line.ok:
                    return line
                if line.value is None:
                    break
                lines.append(line.value.text)
            return ok(lines)
        finally:
            await reader.close(context)

    async def read_binary_file(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            with open(resolved, "rb") as handle:
                return ok(handle.read())
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def open_binary_reader(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        no_follow = options is not None and options.get("no_follow") is True
        handle = None
        try:
            if WINDOWS:
                # Windows has no O_NOFOLLOW; check the final component before opening (not race-free).
                # A directory open reports EACCES there, so classify it before opening.
                resolved_stat = os.lstat(resolved)
                if no_follow and stat_module.S_ISLNK(resolved_stat.st_mode):
                    return err(_symlink_refused(resolved))
                if stat_module.S_ISDIR(resolved_stat.st_mode):
                    return err(FileError("is_directory", "Is a directory", resolved))
                handle = open(_windows_shared_open(resolved), "rb", buffering=0, closefd=True)
            else:
                flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                if no_follow:
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(resolved, flags)
                handle = open(fd, "rb", buffering=0, closefd=True)
            info = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(info.st_mode):
                handle.close()
                handle = None
                if stat_module.S_ISDIR(info.st_mode):
                    return err(FileError("is_directory", "Is a directory", resolved))
                return err(FileError("invalid", "Not a regular file", resolved))
        except OSError as exc:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            if no_follow and exc.errno in (errno_module.ELOOP, errno_module.EMLINK):
                return err(_symlink_refused(resolved, exc))
            return err(to_file_error(exc, resolved))
        if context.aborted:
            handle.close()
            return _aborted_result(context, resolved)
        return ok(LocalBinaryReader(handle, resolved))

    # -- writes ---------------------------------------------------------------

    async def write_file(self, path: str, content, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            if context.aborted:
                return _aborted_result(context, resolved)
            data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
            with open(resolved, "wb") as handle:
                handle.write(data)
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def append_file(self, path: str, content, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            if context.aborted:
                return _aborted_result(context, resolved)
            data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
            with open(resolved, "ab") as handle:
                handle.write(data)
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def truncate_file(self, path: str, size: int, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return err(FileError("invalid", "File size must be a non-negative safe integer", resolved))
        try:
            with open(resolved, "r+b") as handle:
                handle.truncate(size)
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def flush_file(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            if os.path.isdir(resolved):
                return err(FileError("is_directory", "Is a directory", resolved))
            with open(resolved, "r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def rename_file(self, source_path: str, destination_path: str, context: Context) -> Result:
        source = resolve_path(self.cwd, source_path)
        destination = resolve_path(self.cwd, destination_path)
        aborted = _aborted_result(context, destination)
        if aborted is not None:
            return aborted
        try:
            os.replace(source, destination)
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, source))

    # -- listing --------------------------------------------------------------

    async def file_info(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            return file_info_from_stat(resolved, os.lstat(resolved))
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def list_dir(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            infos: List[FileInfo] = []
            for name in os.listdir(resolved):
                after = _aborted_result(context, resolved)
                if after is not None:
                    return after
                entry_path = os.path.join(resolved, name)
                info = file_info_from_stat(entry_path, os.lstat(entry_path))
                if info.ok:
                    infos.append(info.value)
            return ok(infos)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def open_dir_reader(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            reader = LocalDirReader(resolved)
        except NotADirectoryError as exc:
            return err(FileError("not_directory", "Not a directory", resolved, exc))
        except FileNotFoundError as exc:
            return err(FileError("not_found", "No such file or directory", resolved, exc))
        except OSError as exc:
            return err(to_file_error(exc, resolved))
        if context.aborted:
            await reader.close(context)
            return _aborted_result(context, resolved)
        return ok(reader)

    async def canonical_path(self, path: str, context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            os.stat(resolved)
        except OSError as exc:
            return err(to_file_error(exc, resolved))
        return ok(os.path.realpath(resolved))

    async def exists(self, path: str, context: Context) -> Result:
        result = await self.file_info(path, context)
        if result.ok:
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    async def create_dir(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        recursive = True if options is None else bool(options.get("recursive", True))
        try:
            if recursive:
                os.makedirs(resolved, exist_ok=True)
            else:
                os.mkdir(resolved)
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def remove(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:
        resolved = resolve_path(self.cwd, path)
        aborted = _aborted_result(context, resolved)
        if aborted is not None:
            return aborted
        recursive = bool(options.get("recursive", False)) if options else False
        force = bool(options.get("force", False)) if options else False
        try:
            if os.path.isdir(resolved) and not os.path.islink(resolved):
                import shutil as _shutil

                _shutil.rmtree(resolved, ignore_errors=force)
            elif force:
                try:
                    os.remove(resolved)
                except FileNotFoundError:
                    pass
            else:
                os.remove(resolved)
            return ok(None)
        except OSError as exc:
            return err(to_file_error(exc, resolved))

    async def create_temp_dir(self, prefix: Optional[str], context: Context) -> Result:
        aborted = _aborted_result(context)
        if aborted is not None:
            return aborted
        try:
            return ok(tempfile.mkdtemp(prefix=prefix or "tmp-"))
        except OSError as exc:
            return err(to_file_error(exc))

    async def create_temp_file(self, options: Optional[Dict[str, str]], context: Context) -> Result:
        directory = await self.create_temp_dir("tmp-", context)
        if not directory.ok:
            return directory
        prefix = (options or {}).get("prefix", "")
        suffix = (options or {}).get("suffix", "")
        file_path = os.path.join(directory.value, f"{prefix}{uuid.uuid4()}{suffix}")
        try:
            with open(file_path, "wb"):
                pass
            return ok(file_path)
        except OSError as exc:
            return err(to_file_error(exc, file_path))

    # -- shell ----------------------------------------------------------------

    async def exec(self, command, options: Optional[ShellExecOptions], context: Context) -> Result:
        if context.aborted:
            return err(ExecutionError("aborted", "aborted"))
        timeout = None
        if options is not None and options.timeout is not None:
            timeout = options.timeout
            if (
                not isinstance(timeout, (int, float))
                or isinstance(timeout, bool)
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                return err(ExecutionError("timeout", "Invalid timeout: must be a finite number of seconds"))
            if timeout * 1000 > MAX_TIMEOUT_MS:
                return err(
                    ExecutionError("timeout", f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")
                )

        cwd = resolve_path(self.cwd, options.cwd) if options is not None and options.cwd else self.cwd
        stdin_command: Optional[str] = None
        if isinstance(command, str):
            config = await get_shell_config(self._shell_path)
            if not config.ok:
                return config
            shell: ShellConfig = config.value
            if shell.command_transport == "stdin":
                program, args = shell.shell, list(shell.args)
                stdin_command = command
            else:
                program, args = shell.shell, [*shell.args, command]
        else:
            argv = list(command)
            if not argv:
                return err(ExecutionError("spawn_error", "Empty argv: no program to run"))
            program, args = argv[0], argv[1:]
        if not os.path.exists(cwd):
            return err(
                ExecutionError(
                    "spawn_error",
                    f"Working directory does not exist: {cwd}\nCannot execute bash commands.",
                )
            )

        runner = _CommandRunner(self, program, args, cwd, options or ShellExecOptions(), context, stdin_command)
        return await runner.run()

    async def cleanup(self, context: Context) -> None:
        for pid in list(self._active_children):
            await kill_process_tree(pid)
        self._active_children.clear()


class _CommandRunner:
    """One ``exec`` invocation: streaming, spill, window, timeout, and abort."""

    def __init__(
        self,
        env: LocalExecutionEnv,
        program: str,
        args: List[str],
        cwd: str,
        options: ShellExecOptions,
        context: Context,
        stdin_command: Optional[str],
    ) -> None:
        self._env = env
        self._program = program
        self._args = args
        self._cwd = cwd
        self._options = options
        self._context = context
        self._stdin_command = stdin_command
        self._timed_out = False
        self._callback_error: Optional[ExecutionError] = None
        self._spill_error: Optional[ExecutionError] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._spill_path: Optional[str] = None
        self._spill_handle = None
        self._spill_starting: Optional[asyncio.Task] = None
        self._spill_buffer: List[bytes] = []
        self._spill_buffer_bytes = 0
        self._seen_bytes = 0
        self._seen_newlines = 0
        self._stdout_decoder = StreamDecoder()
        self._stderr_decoder = StreamDecoder()
        # Windowed emission state.
        self._held = ""
        self._held_bytes = 0
        self._held_newlines = 0
        self._pending_skip: Optional[ShellOutputSkip] = None
        self._last_stream = "stdout"
        self._last_emit = 0.0
        self._abort_task: Optional[asyncio.Task] = None
        self._timeout_handle = None

    async def run(self) -> Result:
        options = self._options
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._program,
                *self._args,
                cwd=self._cwd,
                env=self._resolved_env(),
                stdin=(
                    asyncio.subprocess.PIPE if self._stdin_command is not None else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **self._process_flags(),
            )
        except (OSError, ValueError, NotImplementedError) as exc:
            return err(ExecutionError("spawn_error", str(exc), exc))
        if self._process.pid:
            self._env._active_children.add(self._process.pid)
        if self._stdin_command is not None and self._process.stdin is not None:
            try:
                self._process.stdin.write(self._stdin_command.encode("utf-8"))
                await self._process.stdin.drain()
                self._process.stdin.close()
            except (OSError, BrokenPipeError):
                pass

        loop = asyncio.get_running_loop()
        self._abort_task = loop.create_task(self._watch_abort())
        if options.timeout is not None:
            self._timeout_handle = loop.call_later(options.timeout, self._on_timeout)

        stdout_task = loop.create_task(self._pump("stdout", self._stdout_decoder))
        stderr_task = loop.create_task(self._pump("stderr", self._stderr_decoder))

        exit_code = await self._process.wait()
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
        if self._abort_task is not None:
            self._abort_task.cancel()

        # Drain the streams; a descendant holding stdio open only gets a short grace window.
        grace = EXIT_STDIO_GRACE_MS / 1000
        while True:
            done, still = await asyncio.wait([stdout_task, stderr_task], timeout=grace)
            if not still:
                break
            spill_draining = self._spill_starting is not None and not self._spill_starting.done()
            if not spill_draining:
                break
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        self._emit_final()
        if self._spill_handle is not None:
            try:
                self._spill_handle.close()
            except OSError:
                pass
        if self._process.pid:
            self._env._active_children.discard(self._process.pid)

        if self._callback_error is not None:
            return err(self._callback_error)
        if self._timed_out:
            error = ExecutionError("timeout", f"timeout:{self._options.timeout}")
            if self._spill_path is not None:
                error.spill_path = self._spill_path
            return err(error)
        if self._context.aborted:
            error = ExecutionError("aborted", "aborted")
            if self._spill_path is not None:
                error.spill_path = self._spill_path
            return err(error)
        if self._spill_error is not None:
            return err(self._spill_error)
        code = exit_code
        if code < 0:
            # Killed by a signal (e.g. OOM killer): map to the conventional 128 + signal number.
            code = 128 + min(-code, 31)
        return ok(ShellExecResult(exit_code=code, spill_path=self._spill_path))

    def _resolved_env(self) -> Dict[str, str]:
        return _shell_env(
            self._env._shell_env,
            self._options.env,
            True if self._options.inherit_env is None else self._options.inherit_env,
        )

    def _process_flags(self) -> Dict[str, Any]:
        if WINDOWS:
            return {"creationflags": getattr(subprocess_module, "CREATE_NO_WINDOW", 0)}
        return {"start_new_session": True}

    def _on_timeout(self) -> None:
        self._timed_out = True
        self._kill()

    def _kill(self) -> None:
        process = self._process
        if process is not None and process.pid and process.returncode is None:
            asyncio.get_running_loop().create_task(kill_process_tree(process.pid))

    async def _watch_abort(self) -> None:
        signal_ = self._context.abort_signal
        if signal_ is None:
            return
        await signal_.wait()
        self._kill()

    def _fail_callback(self, error: BaseException) -> None:
        if self._callback_error is not None:
            return
        cause = error if isinstance(error, Exception) else RuntimeError(str(error))
        self._callback_error = ExecutionError("callback_error", str(cause), cause)
        self._kill()

    # -- output ---------------------------------------------------------------

    def _emit(self, text: str, stream: str, skipped: Optional[ShellOutputSkip] = None) -> None:
        if self._settled or (text == "" and skipped is None) or self._options.on_output is None or self._callback_error is not None:
            return
        self._last_stream = stream
        try:
            self._options.on_output(text, self._context, ShellOutputInfo(stream=stream, skipped=skipped))
        except Exception as exc:  # noqa: BLE001 - a failing output callback fails the command
            self._fail_callback(exc)

    @property
    def _settled(self) -> bool:
        return getattr(self, "_done", False)

    async def _pump(self, stream: str, decoder: StreamDecoder) -> None:
        process = self._process
        reader = process.stdout if stream == "stdout" else process.stderr
        if reader is None:
            return
        while True:
            chunk = await reader.read(READ_CHUNK)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if self._options.window is not None:
                self._window(text, stream)
            else:
                self._emit(text, stream)
            self._record_spill(chunk)
            await self._pace(text)
        residual = decoder.decode()
        if residual:
            if self._options.window is not None:
                self._window(residual, stream)
            else:
                self._emit(residual, stream)

    async def _pace(self, text: str) -> None:
        window = self._options.window
        if window is None:
            return
        now = time.monotonic()
        elapsed_ms = (now - self._last_emit) * 1000
        size_pause = (
            (len(text.encode("utf-8")) / window.bytes_per_second) * 1000 if window.bytes_per_second else 0
        )
        pause = max(window.min_interval_ms - elapsed_ms, size_pause, 0) / 1000
        if pause > 0:
            await asyncio.sleep(pause)
        self._last_emit = time.monotonic()

    def _window(self, text: str, stream: str) -> None:
        self._held += text
        self._held_bytes += len(text.encode("utf-8"))
        self._held_newlines += text.count("\n")
        if self._pending_skip is not None:
            # Deliver the tail kept at the last omission plus everything since; together it is more
            # than the window by at least one byte or line.
            self._flush_window(stream)
            return
        skip = self._window_overflow()
        if skip is not None:
            self._pending_skip = skip

    def _window_overflow(self) -> Optional[ShellOutputSkip]:
        window = self._options.window
        if window is None or not self._held:
            return None
        if self._held_bytes <= window.max_bytes and self._held_newlines <= window.max_lines:
            return None
        data = self._held.encode("utf-8")
        # Cut to the margin: the shortest suffix with more than maxBytes bytes or more than maxLines
        # newlines, so everything delivered after an omission is more than the window, even at flush.
        byte_cut = 0
        if len(data) > window.max_bytes:
            end = len(data) - window.max_bytes - 1
            while end > 0 and (data[end] & 0xC0) == 0x80:
                end -= 1
            byte_cut = end
        line_cut = 0
        newlines = 0
        index = data.rfind(b"\n")
        while index != -1:
            newlines += 1
            if newlines > window.max_lines:
                line_cut = index
                break
            if index == 0:
                break
            index = data.rfind(b"\n", 0, index)
        cut = max(byte_cut, line_cut)
        if cut <= 0:
            return None
        omitted = data[:cut]
        self._held = data[cut:].decode("utf-8", "replace")
        self._held_bytes = len(self._held.encode("utf-8"))
        self._held_newlines = self._held.count("\n")
        return ShellOutputSkip(
            bytes=len(omitted),
            newlines=omitted.count(b"\n"),
            ends_with_newline=omitted.endswith(b"\n"),
        )

    def _flush_window(self, stream: str) -> None:
        skipped = self._pending_skip
        text, self._held = self._held, ""
        self._held_bytes = 0
        self._held_newlines = 0
        self._pending_skip = None
        if text:
            self._emit(text, stream, skipped)
        elif skipped is not None and skipped.bytes:
            # Nothing rides with the omission; report it with the next chunk or the final flush.
            self._pending_skip = skipped

    def _emit_final(self) -> None:
        """Deliver what the window still holds when the stream ends."""
        try:
            if self._options.window is None:
                return
            if self._held or (self._pending_skip is not None and self._pending_skip.bytes):
                self._emit(self._held, self._last_stream, self._pending_skip)
            self._held = ""
            self._pending_skip = None
        finally:
            self._done = True

    # -- spill ----------------------------------------------------------------

    def _record_spill(self, chunk: bytes) -> None:
        spill = self._options.spill
        if spill is None or len(chunk) == 0:
            return
        if self._spill_handle is None and self._spill_starting is None:
            self._seen_bytes += len(chunk)
            self._seen_newlines += chunk.count(b"\n")
            lines = self._seen_newlines + (0 if chunk[-1] == 0x0A else 1)
            if self._seen_bytes <= spill.after_bytes and lines <= spill.after_lines:
                self._spill_buffer.append(chunk)
                self._spill_buffer_bytes += len(chunk)
                return
            buffer = b"".join(self._spill_buffer)
            self._spill_buffer = []
            self._spill_buffer_bytes = 0
            self._spill_starting = asyncio.get_running_loop().create_task(self._start_spill(buffer + chunk))
            return
        self._queue_spill(chunk)

    def _queue_spill(self, chunk: bytes) -> None:
        if self._spill_handle is not None:
            try:
                self._spill_handle.write(chunk)
            except OSError as exc:
                self._fail_spill(exc)
            return
        self._spill_buffer.append(chunk)
        self._spill_buffer_bytes += len(chunk)
        if self._spill_buffer_bytes > SPILL_BUFFER_LIMIT:
            # Do not buffer unboundedly while the temp file opens.
            self._fail_spill(RuntimeError("spill start backpressure"))

    async def _start_spill(self, prefix: bytes) -> None:
        try:
            created = await self._env.create_temp_file({"prefix": "pi-output-", "suffix": ".log"}, self._context)
            if not created.ok:
                raise created.error
            self._spill_path = created.value
            self._spill_handle = open(self._spill_path, "ab", buffering=0)
            if prefix:
                self._spill_handle.write(prefix)
            for queued in self._spill_buffer:
                self._spill_handle.write(queued)
            self._spill_buffer = []
            self._spill_buffer_bytes = 0
        except Exception as exc:  # noqa: BLE001
            self._fail_spill(exc)

    def _fail_spill(self, error: BaseException) -> None:
        if self._spill_error is not None:
            return
        cause = error if isinstance(error, Exception) else RuntimeError(str(error))
        self._spill_error = ExecutionError("unknown", f"Failed to preserve complete shell output: {cause}", cause)
        self._kill()
