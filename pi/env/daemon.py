"""pi-env daemon (port of ``packages/env/daemon/src``).

Gives a Pi Durable host an execution environment on this machine over
stdin/stdout (``docs/protocol.md``). Started as ``python -m pi.env.daemon serve
--token <hex>``, usually through ``ssh``. The first output is the sync line
``PI-ENV <token>``; after it, stdout carries only frames.

The daemon applies Node's semantics where the Python standard library differs
(the ``semantics`` notes of the TS package): recursive ``mkdir`` reports
``ENOTDIR`` for a file in the way of a parent, ``rm`` reports
``ERR_FS_EISDIR`` for a directory without ``recursive``, directory names are
decoded with replacement characters and lstat'ed under that name, commands run
in their own session, and so on.

Porting deviations from the Rust daemon:
- file operations run on the event loop's default executor instead of a fixed
  16-thread pool; chunk writes to one handle are serialized with a per-handle
  lock instead of a worker-queue; spill files are written on a dedicated
  one-thread executor;
- command output paces through one writer task on its own thread, like Rust's
  ``Output`` writer thread; the bulk limit (4 MiB unsent) is measured on the
  queued frames;
- ``rm`` of a directory without ``recursive`` reports ``EISDIR`` (the client's
  ``is_directory``), matching ``LocalExecutionEnv`` rather than Node's
  ``ERR_FS_EISDIR``;
- ``watch`` reuses Durable's Python ``LocalFileWatcher``, whose Windows default
  is native watching (the Rust daemon polls on Windows);
- the shared-aware Windows open helper is ``durable.env.local``'s private
  ``_windows_shared_open`` (same repository, one implementation).

Run on Python 3.9+; no third-party dependencies.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import contextlib
import errno
import json as json_module
import os
import platform
import shutil
import signal
import stat as stat_module
import struct
import subprocess as subprocess_module
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..durable.env import FileError, LineScanner, StreamDecoder, WatchTarget
from ..durable.env.local import _windows_shared_open, get_shell_config
from ..durable.env.watch import LocalFileWatcher

WINDOWS = sys.platform == "win32"

if WINDOWS:
    import ctypes

PROTOCOL = 1
VERSION = "0.1.0"
PING_INTERVAL = 5.0
SILENCE_LIMIT_MS = 30_000
SCAN_CHUNK = 64 * 1024
# Open handles per connection, beyond which opening fails with ``EMFILE``.
MAX_HANDLES = 4096
# Replies with larger payloads queue behind command output instead of going first.
CONTROL_PAYLOAD = 64 * 1024
# Unsent bulk bytes above which unwindowed command output waits.
BULK_LIMIT = 4 * 1024 * 1024
EXIT_STDIO_GRACE = 0.1
# How often a delivery waiting for the link to drain looks again.
UNSENT_POLL = 0.02

REQUEST = 1
RESULT = 2
ERROR = 3
EVENT = 4
CANCEL = 5
PING = 6

# ---------------------------------------------------------------------------
# Errors as Node reports them: a code such as ``ENOENT`` plus a diagnostic message.
# ---------------------------------------------------------------------------


class Failure(Exception):
    """A failure with a Node-style code; messages are diagnostic only."""

    def __init__(self, code: str, message: str, path: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.extra = extra or {}

    def with_extra(self, extra: Dict[str, Any]) -> "Failure":
        merged = dict(self.extra)
        merged.update(extra)
        return Failure(self.code, self.message, self.path, merged)

    def to_json(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            value["path"] = self.path
        value.update(self.extra)
        return value


class Synthetic(OSError):
    """A code libuv reports where the system call itself reported another one."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code_name = code


# libuv's ``uv_strerror`` texts, which Node uses on every system.
_DESCRIPTIONS = {
    "EACCES": "permission denied",
    "EAGAIN": "resource temporarily unavailable",
    "EBADF": "bad file descriptor",
    "EBUSY": "resource busy or locked",
    "EEXIST": "file already exists",
    "EFBIG": "file too large",
    "EINVAL": "invalid argument",
    "EIO": "i/o error",
    "EISDIR": "illegal operation on a directory",
    "ELOOP": "too many symbolic links encountered",
    "EMFILE": "too many open files",
    "EMLINK": "too many links",
    "ENAMETOOLONG": "name too long",
    "ENFILE": "file table overflow",
    "ENOENT": "no such file or directory",
    "ENOMEM": "not enough memory",
    "ENOSPC": "no space left on device",
    "ENOSYS": "function not implemented",
    "ENOTDIR": "not a directory",
    "ENOTEMPTY": "directory not empty",
    "ENOTSUP": "operation not supported on socket",
    "EPERM": "operation not permitted",
    "EROFS": "read-only file system",
    "ETXTBSY": "text file is busy",
    "EXDEV": "cross-device link not permitted",
}

# libuv's ``uv_translate_sys_error`` for the errors file and process operations produce on Windows.
_WIN_ERROR_NAMES = {
    1: "EISDIR",
    2: "ENOENT",
    3: "ENOENT",
    4: "EMFILE",
    5: "EPERM",
    6: "EBADF",
    8: "ENOMEM",
    14: "ENOMEM",
    17: "EXDEV",
    19: "EROFS",
    32: "EBUSY",
    33: "EBUSY",
    39: "ENOSPC",
    50: "ENOTSUP",
    80: "EEXIST",
    82: "ENOSPC",
    87: "EINVAL",
    112: "ENOSPC",
    122: "EINVAL",
    123: "ENOENT",
    126: "ENOENT",
    131: "EINVAL",
    145: "ENOTEMPTY",
    161: "ENOENT",
    183: "EEXIST",
    206: "ENAMETOOLONG",
    267: "ENOENT",
    998: "EFAULT",
    1314: "EPERM",
    1464: "EINVAL",
    4392: "ENOENT",
    740: "EACCES",
    1920: "EACCES",
    1921: "ELOOP",
}


def error_name(error: BaseException) -> str:
    """The libuv name of an error: a synthetic code, or the system's code translated per platform."""
    code = getattr(error, "code_name", None)
    if isinstance(code, str):
        return code
    if isinstance(error, OSError):
        if WINDOWS and error.winerror is not None:
            return _WIN_ERROR_NAMES.get(error.winerror, "UNKNOWN")
        if error.errno:
            return errno.errorcode.get(error.errno, "UNKNOWN")
    return "UNKNOWN"


def io_failure(error: BaseException, syscall: str, path: str) -> Failure:
    return io_between(error, syscall, path, None)


def io_between(error: BaseException, syscall: str, path: str, destination: Optional[str]) -> Failure:
    code = error_name(error)
    description = _DESCRIPTIONS.get(code)
    if description is None:
        text = str(error)
        text = text.split(" (os error")[0] if " (os error" in text else text
        description = text.lower()
    target = f"'{path}' -> '{destination}'" if destination is not None else f"'{path}'"
    return Failure(code, f"{code}: {description}, {syscall} {target}", path)


def synthetic(code: str) -> OSError:
    return Synthetic(code)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """Lone surrogates become U+FFFD, as Rust's ``to_string_lossy`` and Node's encoding do.

    File names on POSIX arrive surrogate-escaped, and a lone surrogate cannot be encoded as JSON.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    if isinstance(value, dict):
        return {
            json_safe(key) if isinstance(key, str) else key: json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def encode_frame(kind: int, request_id: int, json: Dict[str, Any], payload: bytes = b"") -> bytes:
    body = json_module.dumps(json_safe(json), ensure_ascii=False).encode("utf-8")
    header = struct.pack(">IBII", 9 + len(body) + len(payload), kind, request_id, len(body))
    return header + body + payload


# ---------------------------------------------------------------------------
# Output: the single writer of stdout. Control frames always go before bulk
# command output, so a slow link delays output but never a cancel's result.
# ---------------------------------------------------------------------------


class Output:
    """The single writer of stdout.

    Writing runs on a dedicated one-thread executor: file operations share the loop's default
    executor, and a blocked operation there (a FIFO open, for example) must not stall replies.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, stream: Any) -> None:
        self._loop = loop
        self._stream = stream
        # A writer of its own, like the Rust daemon's writer thread.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-env-output")
        self._control: Deque[bytes] = collections.deque()
        self._bulk: Deque[Tuple[bytes, Optional[_Counter]]] = collections.deque()
        self._bulk_bytes = 0
        self._closed = False
        self._wake: Optional[asyncio.Event] = None
        self._task = loop.create_task(self._run())

    def control(self, frame: bytes) -> None:
        self._control.append(frame)
        self._wake_writer()

    def bulk(self, frame: bytes, counter: Optional["_Counter"] = None) -> None:
        # Payload bytes are after the 13-byte header: length, type, id, json length.
        payload_length = struct.unpack_from(">I", frame, 0)[0] - 9 - struct.unpack_from(">I", frame, 9)[0]
        self._bulk_bytes += max(payload_length, 0)
        if counter is not None:
            counter.count += 1
        self._bulk.append((frame, counter))
        self._wake_writer()

    def _wake_writer(self) -> None:
        if self._wake is not None:
            self._wake.set()

    async def wait_for_room(self, stop: Callable[[], bool]) -> None:
        """Wait while more than the bulk limit is unsent, unless ``stop`` is set."""
        while self._bulk_bytes > BULK_LIMIT and not self._closed and not stop():
            await asyncio.sleep(0.02)

    def close(self) -> None:
        self._closed = True
        self._wake_writer()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if self._control:
                frame = self._control.popleft()
                counter = None
            elif self._bulk:
                frame, counter = self._bulk.popleft()
                payload_length = struct.unpack_from(">I", frame, 0)[0] - 9 - struct.unpack_from(">I", frame, 9)[0]
                self._bulk_bytes -= max(payload_length, 0)
            elif self._closed:
                return
            else:
                self._wake = asyncio.Event()
                await self._wake.wait()
                self._wake = None
                continue
            await loop.run_in_executor(self._executor, self._write, frame)
            if counter is not None:
                counter.count -= 1

    def shutdown(self) -> None:
        """Stop the writer thread; queued frames are dropped, as at the end of a connection."""
        self.close()
        self._task.cancel()
        self._executor.shutdown(wait=False)

    def _write(self, frame: bytes) -> None:
        """Runs on the executor thread: only plain fields are touched, no loop primitives."""
        try:
            self._stream.write(frame)
            self._stream.flush()
        except OSError:
            # The client is gone; stop writing. The loop exits on its next wake.
            self._closed = True


class _Counter:
    def __init__(self) -> None:
        self.count = 0


# ---------------------------------------------------------------------------
# Cancellation of one request, registered before the request runs so a cancel
# that arrives first is not lost.
# ---------------------------------------------------------------------------


class Control:
    def __init__(self) -> None:
        self.aborted = False
        self.killed = False
        self.event: Optional[asyncio.Event] = None  # ``watch`` waits on this
        self.sender: Optional[Callable[[bool], None]] = None  # ``exec`` messages

    def cancel(self, kill: bool) -> None:
        self.killed = self.killed or kill
        self.aborted = self.aborted or not kill
        if self.event is not None:
            self.event.set()
        if self.sender is not None:
            self.sender(kill)


# ---------------------------------------------------------------------------
# Handle table
# ---------------------------------------------------------------------------


class ClosableHandle:
    """A handle whose resource closes after the operations already running on it, as Rust's ``Arc`` keeps a file alive until its users finish."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._ops = 0
        self._closing = False
        self._ops_lock = threading.Lock()

    @contextlib.contextmanager
    def operating(self):
        with self._ops_lock:
            self._ops += 1
        try:
            yield
        finally:
            with self._ops_lock:
                self._ops -= 1
                release = self._closing and self._ops == 0
            if release:
                self.release()

    def close(self) -> None:
        with self._ops_lock:
            self._closing = True
            release = self._ops == 0
        if release:
            self.release()

    def release(self) -> None:
        """Close the underlying resource; subclasses override."""
        raise NotImplementedError


class FileHandle(ClosableHandle):
    def __init__(self, file: Any, path: str) -> None:
        super().__init__(path)
        self.file = file
        # A failed chunk write poisons a write handle, so later chunks cannot leave a gap.
        self.failed = False
        # Chunk writes run one at a time in arrival order, though the client sends several at once.
        self.lock: Optional[asyncio.Lock] = None
        # Positional reads on Windows (no ``os.pread``) serialize seek + read through this lock.
        self.thread_lock = threading.Lock()

    def release(self) -> None:
        try:
            self.file.close()
        except OSError:
            pass

    def read_at(self, offset: int, length: int) -> bytes:
        """Up to ``length`` bytes at ``offset``, as ``os.pread`` where it exists."""
        pread = getattr(os, "pread", None)
        if pread is not None:
            chunks: List[bytes] = []
            filled = 0
            while filled < length:
                read = pread(self.file.fileno(), length - filled, offset + filled)
                if not read:
                    break
                chunks.append(read)
                filled += len(read)
            return b"".join(chunks)
        with self.thread_lock:
            self.file.seek(offset)
            return self.file.read(length)


class DirHandle(ClosableHandle):
    def __init__(self, entries: Any, path: str) -> None:
        super().__init__(path)
        self.entries = entries
        # Listings of one directory run one at a time.
        self.lock: Optional[asyncio.Lock] = None

    def release(self) -> None:
        try:
            self.entries.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Server:
    def __init__(self, output: Output, tmpdir: str, groups: set) -> None:
        self.output = output
        self.tmpdir = tmpdir
        self.handles: Dict[int, Any] = {}
        self.next_handle = 1
        self.controls: Dict[int, Control] = {}
        self.groups = groups
        # Response tasks are kept here: a task no one references can be garbage-collected mid-run.
        self.background: set = set()
        # Spill files are written off the event loop; one worker keeps chunk order.
        self.spill_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-env-spill")

    def spawn(self, task: "asyncio.Task") -> None:
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    # -- handles ---------------------------------------------------------------

    def file_handle(self, json: Dict[str, Any]) -> FileHandle:
        handle = self.lookup(json, "file")
        if isinstance(handle, FileHandle):
            return handle
        raise Failure("EBADF", "not a file handle")

    def dir_handle(self, json: Dict[str, Any]) -> DirHandle:
        handle = self.lookup(json, "dir")
        if isinstance(handle, DirHandle):
            return handle
        raise Failure("EBADF", "not a directory handle")

    def lookup(self, json: Dict[str, Any], kind: str) -> Any:
        raw = json.get("handle")
        if not isinstance(raw, int):
            raise Failure("EINVAL", "missing or invalid field handle")
        found = self.handles.get(raw)
        if found is None:
            raise Failure("EBADF", "unknown handle")
        return found

    def insert(self, handle: Any) -> int:
        if len(self.handles) >= MAX_HANDLES:
            handle.close()
            raise Failure("EMFILE", "EMFILE: too many open files")
        handle_id = self.next_handle
        self.next_handle += 1
        self.handles[handle_id] = handle
        return handle_id

    # -- dispatch --------------------------------------------------------------

    def dispatch(self, json: Dict[str, Any], payload: bytes, control: Control) -> Tuple[Dict[str, Any], bytes]:
        """One file operation, run on the executor; returns the result JSON and payload."""
        if not isinstance(json, dict):
            raise Failure("EINVAL", "invalid JSON")
        op = json.get("op")
        empty = lambda value: (value, b"")  # noqa: E731
        if op == "hello":
            return empty(
                {
                    "protocol": PROTOCOL,
                    "version": VERSION,
                    "os": os_name(),
                    "arch": arch_name(),
                    "home": home_dir(),
                    "separator": "\\" if WINDOWS else "/",
                    "tmpdir": self.tmpdir,
                    "cwd": working_dir(),
                    "driveCwds": drive_cwds(),
                    "pid": os.getpid(),
                }
            )
        if op == "lstat":
            return empty(fs_lstat(field(json, "path")))
        if op == "realpath":
            return empty(fs_realpath(field(json, "path")))
        if op == "write":
            path = field(json, "path")
            parents_raw = json.get("parents")
            parents = parents_raw if isinstance(parents_raw, bool) else True
            opened = fs_write(path, bool(json.get("append")), parents, payload)
            if not json.get("keep"):
                opened.close()
                return empty({})
            return empty({"handle": self.insert(FileHandle(opened, path))})
        if op == "writeChunk":
            handle = self.file_handle(json)
            with handle.operating():
                if handle.failed:
                    raise Failure("EBADF", "an earlier write failed")
                try:
                    handle.file.write(payload)
                except OSError as error:
                    handle.failed = True
                    raise io_failure(error, "write", handle.path) from None
            return empty({})
        if op == "truncate":
            return empty(fs_truncate(field(json, "path"), number(json, "size")))
        if op == "fsync":
            return empty(fs_fsync(field(json, "path")))
        if op == "rename":
            return empty(fs_rename(field(json, "path"), field(json, "to")))
        if op == "mkdir":
            return empty(fs_mkdir(field(json, "path"), bool(json.get("recursive"))))
        if op == "rm":
            return empty(fs_rm(field(json, "path"), bool(json.get("recursive")), bool(json.get("force"))))
        if op == "mkdtemp":
            return empty(fs_mkdtemp(field(json, "path")))
        if op == "open":
            path = field(json, "path")
            if json.get("mode") == "read":
                # Node's ``open(path, "r")``: any kind of file, blocking; reads report what the file is.
                opened, stat_json, stat_error = sys_open_read(path)
                result: Dict[str, Any] = {}
                if stat_error is not None:
                    result["statError"] = stat_error.to_json()
                else:
                    result["stat"] = stat_json
                result["handle"] = self.insert(FileHandle(opened, path))
                return empty(result)
            opened, info = fs_open_reader(path, bool(json.get("noFollow")))
            return empty({"handle": self.insert(FileHandle(opened, path)), "info": info})
        if op == "pread":
            handle = self.file_handle(json)
            length = min(number(json, "length"), 16 * 1024 * 1024 - 9)
            offset = json.get("offset")
            with handle.operating():
                if offset is not None:
                    return ({}, fs_pread(handle, handle.path, int(offset), length))
                return ({}, fs_read(handle, handle.path, length))
        if op == "fstat":
            handle = self.file_handle(json)
            with handle.operating():
                try:
                    metadata = os.fstat(handle.file.fileno())
                except OSError as error:
                    raise io_failure(error, "fstat", handle.path) from None
            return empty(info_of(handle.path, metadata))
        if op == "scanLines":
            handle = self.file_handle(json)
            with handle.operating():
                return empty(scan_lines(handle, json, control))
        if op == "opendir":
            path = field(json, "path")
            try:
                entries = sys_read_dir(path)
            except OSError as error:
                raise io_failure(error, "opendir", path) from None
            return empty({"handle": self.insert(DirHandle(entries, path))})
        if op == "readdir":
            return empty(self.read_dir(json))
        if op == "close":
            raw = json.get("handle")
            if not isinstance(raw, int):
                raise Failure("EINVAL", "missing or invalid field handle")
            found = self.handles.pop(raw, None)
            if found is not None:
                # Closes now, or after the operations already reading or writing it finish.
                found.close()
            return empty({})
        raise Failure("EINVAL", f"unknown operation {op}")

    def read_dir(self, json: Dict[str, Any]) -> Dict[str, Any]:
        maximum = max(number(json, "max"), 1)
        handle = self.dir_handle(json)
        entries: List[Dict[str, Any]] = []
        done = False
        with handle.operating():
            while len(entries) < maximum:
                try:
                    entry = next(handle.entries)
                except StopIteration:
                    done = True
                    break
                except OSError as error:
                    raise io_failure(error, "readdir", handle.path) from None
                entries.append(dir_entry(handle.path, entry))
        return {"entries": entries, "done": done}


def field(json: Dict[str, Any], key: str) -> str:
    value = json.get(key)
    if not isinstance(value, str):
        raise Failure("EINVAL", f"missing field {key}")
    return value


def number(json: Dict[str, Any], key: str) -> int:
    value = json.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Failure("EINVAL", f"missing or invalid field {key}")
    return value


def os_name() -> str:
    return {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


def arch_name() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine


def home_dir() -> str:
    if WINDOWS:
        return os.environ.get("USERPROFILE", "")
    return os.environ.get("HOME", "")


def working_dir() -> str:
    try:
        return os.getcwd()
    except OSError:
        return ""


def node_tmpdir() -> str:
    """Node's ``os.tmpdir()``; Termux's Node falls back to ``$PREFIX/tmp``."""
    if WINDOWS:
        def variable(key: str) -> Optional[str]:
            value = os.environ.get(key)
            return value or None

        path = variable("TEMP") or variable("TMP")
        if path is None:
            root = variable("SystemRoot") or variable("windir") or ""
            path = os.path.join(root, "temp")
    else:
        path = None
        for key in ("TMPDIR", "TMP", "TEMP"):
            value = os.environ.get(key)
            if value:
                path = value
                break
        if path is None:
            if sys.platform == "linux" and os.environ.get("PREFIX", "").endswith("/com.termux/files/usr"):
                # Close to Termux: its sshd normally sets TMPDIR; fall back as Termux's Node does.
                path = os.path.join(os.environ["PREFIX"], "tmp")
            else:
                path = "/tmp"
    if len(path) > 1 and path[-1] in ("/", "\\") and not path.endswith(":\\"):
        path = path[:-1]
    return path


def drive_cwds() -> Dict[str, str]:
    """The per-drive working directories of the ``=C:`` variables (Windows)."""
    result: Dict[str, str] = {}
    for key, value in os.environ.items():
        if len(key) == 3 and key[0] == "=" and key[1].isalpha() and key[2] == ":":
            result[key[1:].upper()] = value
    return result


# ---------------------------------------------------------------------------
# File operations with Node's semantics (daemon ``fs.rs``).
# ---------------------------------------------------------------------------


def info_of(path: str, metadata: os.stat_result) -> Dict[str, Any]:
    if stat_module.S_ISREG(metadata.st_mode):
        kind = "file"
    elif stat_module.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat_module.S_ISLNK(metadata.st_mode):
        kind = "symlink"
    else:
        kind = "other"
    mtime_sec = metadata.st_mtime_ns // 1_000_000_000
    mtime_nsec = metadata.st_mtime_ns - mtime_sec * 1_000_000_000
    return {
        "name": os.path.basename(path),
        "kind": kind,
        "size": metadata.st_size,
        "mtimeSec": mtime_sec,
        "mtimeNsec": mtime_nsec,
        "dev": metadata.st_dev,
        "ino": metadata.st_ino,
    }


def fs_lstat(path: str) -> Dict[str, Any]:
    try:
        return info_of(path, os.lstat(path))
    except OSError as error:
        raise io_failure(error, "lstat", path) from None


def fs_realpath(path: str) -> Dict[str, Any]:
    try:
        # Rust's ``canonicalize`` requires the path to exist.
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        resolved = os.path.realpath(path)
        if WINDOWS:
            if resolved.startswith("\\\\?\\UNC\\"):
                resolved = "\\\\" + resolved[len("\\\\?\\UNC\\") :]
            elif resolved.startswith("\\\\?\\"):
                resolved = resolved[len("\\\\?\\") :]
        return {"path": resolved}
    except OSError as error:
        raise io_failure(error, "realpath", path) from None


def mkdir_recursive(path: str) -> None:
    """Node's recursive ``mkdir``: an existing directory is fine, an existing file is ``EEXIST``, a file in the way of a parent is ``ENOTDIR``."""
    try:
        os.mkdir(path)
        return
    except FileExistsError:
        if os.path.isdir(path):
            return
        raise
    except FileNotFoundError:
        parent = os.path.dirname(path)
        if not parent or parent == path:
            raise
        try:
            mkdir_recursive(parent)
        except FileExistsError:
            raise synthetic("ENOTDIR") from None
        try:
            os.mkdir(path)
        except FileExistsError:
            if os.path.isdir(path):
                return
            raise


def fs_mkdir(path: str, recursive: bool) -> Dict[str, Any]:
    try:
        if recursive:
            mkdir_recursive(path)
        else:
            os.mkdir(path)
        return {}
    except OSError as error:
        raise io_failure(error, "mkdir", path) from None


def fs_write(path: str, append: bool, parents: bool, content: bytes) -> Any:
    """``writeFile``/``appendFile``: create missing parents like Node's recursive ``mkdir``, then write."""
    if parents:
        parent = os.path.dirname(path)
        if parent:
            try:
                mkdir_recursive(parent)
            except OSError as error:
                raise io_failure(error, "mkdir", parent) from None
    try:
        opened = open(path, "ab" if append else "wb", buffering=0)
    except OSError as error:
        raise io_failure(error, "open", path) from None
    try:
        opened.write(content)
    except OSError as error:
        opened.close()
        raise io_failure(error, "write", path) from None
    return opened


def fs_truncate(path: str, size: int) -> Dict[str, Any]:
    try:
        with open(path, "r+b") as opened:
            opened.truncate(size)
        return {}
    except OSError as error:
        raise io_failure(error, "ftruncate", path) from None


def fs_fsync(path: str) -> Dict[str, Any]:
    try:
        # POSIX refuses to open a directory for writing; check first so Windows reports the same.
        if os.path.isdir(path):
            raise Failure("EISDIR", f"EISDIR: illegal operation on a directory, open '{path}'", path)
        with open(path, "r+b") as opened:
            os.fsync(opened.fileno())
        return {}
    except OSError as error:
        raise io_failure(error, "fsync", path) from None


def fs_rename(source: str, destination: str) -> Dict[str, Any]:
    try:
        os.replace(source, destination)
        return {}
    except OSError as error:
        raise io_between(error, "rename", source, destination) from None


def fs_rm(path: str, recursive: bool, force: bool) -> Dict[str, Any]:
    """Node's ``fs.rm``: missing paths fail unless ``force``; a directory without ``recursive`` reports ``EISDIR``, which the client maps to ``is_directory`` like ``LocalExecutionEnv`` does."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        if force:
            return {}
        raise io_failure(error, "lstat", path) from None
    except OSError as error:
        raise io_failure(error, "lstat", path) from None
    try:
        if stat_module.S_ISDIR(metadata.st_mode):
            if not recursive:
                # ``LocalExecutionEnv`` reports ``is_directory``; an ``EISDIR`` failure keeps the two
                # environments' ``FileError`` codes equal (Node itself says ``ERR_FS_EISDIR``).
                raise Failure("EISDIR", "Is a directory", path)
            shutil.rmtree(path)
        else:
            remove_file(path)
        return {}
    except OSError as error:
        raise io_failure(error, "rm", path) from None


def remove_file(path: str) -> None:
    if WINDOWS:
        # Node's ``rm`` on Windows: a symbolic link or junction to a directory is removed as a
        # directory entry, and a read-only file is retried after making it writable.
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            raise
        if stat_module.S_ISLNK(metadata.st_mode) and os.path.isdir(path):
            os.rmdir(path)
            return
        try:
            os.remove(path)
        except PermissionError:
            os.chmod(path, stat_module.S_IWRITE)
            os.remove(path)
        return
    os.remove(path)


def fs_mkdtemp(prefix: str) -> Dict[str, Any]:
    """A new directory named ``prefix`` plus six random characters, as libuv's ``mkdtemp`` creates it."""
    try:
        directory, name = os.path.split(prefix)
        # ``mkdtemp(3)`` creates the template in the current directory when the prefix has none.
        created = tempfile.mkdtemp(prefix=name, dir=directory or ".")
        return {"path": created}
    except OSError as error:
        raise io_failure(error, "mkdtemp", f"{prefix}XXXXXX") from None


def fs_open_reader(path: str, no_follow: bool) -> Tuple[Any, Dict[str, Any]]:
    """Open a regular file for positional reads; directories fail with ``EISDIR``, other non-regular files with ``NOT_REGULAR``, and with ``no_follow`` a final symbolic link with ``SYMLINK``."""
    opened = None
    try:
        if WINDOWS:
            # Windows cannot open a directory as a file and has no ``O_NOFOLLOW``: check the path
            # first, as ``NodeExecutionEnv`` does there (not race-free).
            if no_follow:
                metadata = os.lstat(path)
                if stat_module.S_ISLNK(metadata.st_mode):
                    raise Failure("SYMLINK", "Refusing to follow a symbolic link", path)
            if os.path.isdir(path):
                raise Failure("EISDIR", "EISDIR: illegal operation on a directory, read", path)
            # Full sharing, so a rename or removal of the open file works, as Rust's opens do.
            opened = open(_windows_shared_open(path), "rb", buffering=0, closefd=True)
        else:
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            if no_follow:
                flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            opened = open(fd, "rb", buffering=0, closefd=True)
        metadata = os.fstat(opened.fileno())
        if stat_module.S_ISDIR(metadata.st_mode):
            raise Failure("EISDIR", "EISDIR: illegal operation on a directory, read", path)
        if not stat_module.S_ISREG(metadata.st_mode):
            raise Failure("NOT_REGULAR", "Not a regular file", path)
        return opened, info_of(path, metadata)
    except Failure:
        if opened is not None:
            opened.close()
        raise
    except OSError as error:
        if opened is not None:
            opened.close()
        if no_follow and error.errno in (errno.ELOOP, errno.EMLINK):
            raise Failure("SYMLINK", "Refusing to follow a symbolic link", path) from None
        raise io_failure(error, "open", path) from None


def sys_open_read(path: str) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Failure]]:
    """Node's ``open(path, "r")``: any kind of file, blocking; fstat failure is reported, not fatal."""
    try:
        if WINDOWS:
            opened = open(_windows_shared_open(path), "rb", buffering=0, closefd=True)
        else:
            opened = open(path, "rb", buffering=0)
    except OSError as error:
        raise io_failure(error, "open", path) from None
    try:
        metadata = os.fstat(opened.fileno())
    except OSError as error:
        return opened, None, io_failure(error, "fstat", path)
    return opened, info_of(path, metadata), None


def fs_pread(handle: FileHandle, path: str, offset: int, length: int) -> bytes:
    """Up to ``length`` bytes at ``offset``; fewer only at the end of the file."""
    try:
        return handle.read_at(offset, length)
    except OSError as error:
        raise io_failure(error, "read", path) from None


def fs_read(handle: FileHandle, path: str, length: int) -> bytes:
    """Up to ``length`` bytes from the current position: one ``read`` call, as Node's ``readFile`` makes for files of unknown size (FIFOs, devices)."""
    try:
        return os.read(handle.file.fileno(), length)
    except OSError as error:
        raise io_failure(error, "read", path) from None


def scan_lines(handle: FileHandle, json: Dict[str, Any], control: Control) -> Dict[str, Any]:
    start_line = number(json, "startLine")
    end_line = json.get("endLine")
    if end_line is not None and (not isinstance(end_line, int) or isinstance(end_line, bool) or end_line <= start_line):
        raise Failure("EINVAL", "Invalid line range")
    try:
        scanner = LineScanner(start_line, end_line)
    except ValueError:
        raise Failure("EINVAL", "Invalid line range") from None
    path = handle.path
    position = 0
    try:
        while True:
            if control.aborted:
                raise Failure("aborted", "aborted")
            read = handle.read_at(position, SCAN_CHUNK)
            if not read:
                return scan_to_json(scanner.finish())
            scanner.push(read)
            position += len(read)
    except OSError as error:
        raise io_failure(error, "read", path) from None


def scan_to_json(scan: Any) -> Dict[str, Any]:
    return {
        "newlines": scan.newlines,
        "start": scan.start,
        "end": scan.end,
        "firstLineEnd": scan.first_line_end,
        "lastLineStart": scan.last_line_start,
        "selectedBytes": scan.selected_bytes,
        "firstLineBytes": scan.first_line_bytes,
    }


def sys_read_dir(path: str) -> Any:
    try:
        return os.scandir(path)
    except OSError as error:
        # libuv's ``scandir`` and ``opendir`` report a file as ``ENOTDIR`` on Windows.
        if WINDOWS and getattr(error, "winerror", None) == 267:
            raise synthetic("ENOTDIR") from None
        raise


def dir_entry(directory: str, entry: os.DirEntry) -> Dict[str, Any]:
    """One directory entry as Node lists it: the name decoded as UTF-8 (invalid bytes replaced), and lstat of the directory joined with that decoded name, so a name that is not valid UTF-8 reports ``ENOENT`` as in Node. ``raw`` carries the original bytes of such a name, which libuv sorts by."""
    raw = os.fsencode(entry.name) if not WINDOWS else None
    name = entry.name if WINDOWS else raw.decode("utf-8", "replace")
    entry_path = os.path.join(directory, name)
    try:
        metadata = os.lstat(entry_path)
        result: Dict[str, Any] = {"name": name, "info": info_of(entry_path, metadata)}
    except OSError as error:
        result = {"name": name, "error": io_failure(error, "lstat", entry_path).to_json()}
    if raw is not None and raw != name.encode("utf-8", "surrogatepass"):
        result["raw"] = list(raw)
    return result


# ---------------------------------------------------------------------------
# ``exec`` (daemon ``exec.rs`` + ``window.rs``): Durable's ``NodeExecutionEnv.exec``
# semantics, output streamed as ``output`` events.
# ---------------------------------------------------------------------------


class Window:
    def __init__(self, max_bytes: int, max_lines: int, min_interval: float, bytes_per_second: float) -> None:
        self.max_bytes = max_bytes
        self.max_lines = max_lines
        self.min_interval = min_interval
        self.bytes_per_second = bytes_per_second

    @staticmethod
    def from_json(json: Any) -> Optional["Window"]:
        if not isinstance(json, dict):
            return None
        try:
            max_bytes = json["maxBytes"]
            max_lines = json["maxLines"]
            if not _non_negative_int(max_bytes) or not _non_negative_int(max_lines):
                # Rust reads these as ``as_u64``; anything else ignores the window.
                return None
            return Window(
                max_bytes=max_bytes,
                max_lines=max_lines,
                min_interval=max(float(json["minIntervalMs"]), 0.0) / 1000.0,
                bytes_per_second=max(float(json["bytesPerSecond"]), 1.0),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class Skipped:
    def __init__(self) -> None:
        self.bytes = 0
        self.newlines = 0
        self.ends_with_newline = False


class WindowEvent:
    def __init__(self, stream: str, text: str, skipped: Optional[Dict[str, Any]]) -> None:
        self.stream = stream
        self.text = text
        self.skipped = skipped


class Pending:
    """Durable's output window: undelivered command output is held and coalesced; text followed by more than the window (by a byte or a line) can never be in the caller's tail and is replaced by counts. Delivery follows the caller's commit pace, and waits while an earlier frame is still unsent."""

    def __init__(self, window: Window) -> None:
        self.window = window
        self.segments: List[Tuple[str, str]] = []
        self.bytes = 0
        self.newlines = 0
        self.skipped: Optional[Skipped] = None
        self.next_send = time.monotonic()

    def is_empty(self) -> bool:
        return not self.segments

    def push(self, stream: str, text: str) -> None:
        if text == "":
            return
        size = len(text.encode("utf-8"))
        self.bytes += size
        self.newlines += text.count("\n")
        if self.segments and self.segments[-1][0] == stream:
            self.segments[-1] = (stream, self.segments[-1][1] + text)
        else:
            self.segments.append((stream, text))
        # Trimming costs a pass over the held text; do it once a few windows have accumulated.
        if self.bytes > 4 * (self.window.max_bytes + 1) or self.newlines > 4 * (self.window.max_lines + 1):
            self.trim()

    def _proves(self, total_bytes: int, total_newlines: int) -> bool:
        """Whether the text after a cut still proves the cut text is outside the window."""
        return total_bytes > self.window.max_bytes or total_newlines > self.window.max_lines

    def trim(self) -> None:
        """Replace the longest prefix whose remainder still proves exclusion by counts."""
        removed = ""
        # Whole segments first.
        while len(self.segments) > 1:
            text = self.segments[0][1]
            size = len(text.encode("utf-8"))
            newlines = text.count("\n")
            if not self._proves(self.bytes - size, self.newlines - newlines):
                break
            self.segments.pop(0)
            self.bytes -= size
            self.newlines -= newlines
            removed += text
        # Then a prefix of the first remaining segment, at a character boundary.
        first = self.segments[0][1]
        data = first.encode("utf-8")
        rest_bytes = self.bytes - len(data)
        rest_newlines = self.newlines - first.count("\n")
        cut = 0
        # By bytes: keep more than ``max_bytes`` overall.
        keep_bytes = max(self.window.max_bytes + 1 - rest_bytes, 0)
        if len(data) > keep_bytes:
            index = len(data) - keep_bytes
            while index > 0 and (data[index] & 0xC0) == 0x80:
                index -= 1
            cut = max(cut, index)
        # By lines: keep more than ``max_lines`` newlines overall, starting at a newline of this segment.
        keep_newlines = max(self.window.max_lines + 1 - rest_newlines, 0)
        if keep_newlines > 0:
            positions = [index for index, byte in enumerate(data) if byte == 0x0A]
            if len(positions) >= keep_newlines:
                cut = max(cut, positions[len(positions) - keep_newlines])
        else:
            cut = len(data)
        if cut > 0:
            prefix_bytes = data[:cut]
            prefix = prefix_bytes.decode("utf-8")
            rest = data[cut:].decode("utf-8")
            self.segments[0] = (self.segments[0][0], rest)
            self.bytes -= len(prefix_bytes)
            self.newlines -= prefix.count("\n")
            removed += prefix
            if rest == "":
                self.segments.pop(0)
        if removed:
            skipped = self.skipped
            if skipped is None:
                skipped = Skipped()
                self.skipped = skipped
            skipped.bytes += len(removed.encode("utf-8"))
            skipped.newlines += removed.count("\n")
            skipped.ends_with_newline = removed.endswith("\n")

    def take(self) -> List[WindowEvent]:
        """Deliver everything held: one event carrying the skip and all text after it, or one event per segment."""
        segments = self.segments
        self.segments = []
        delivered = self.bytes
        self.bytes = 0
        self.newlines = 0
        # The caller writes at most the window per sample, and paces itself by what it wrote.
        pace = min(delivered, self.window.max_bytes) / self.window.bytes_per_second
        self.next_send = time.monotonic() + max(self.window.min_interval, pace)
        skipped = self.skipped
        self.skipped = None
        if skipped is not None:
            stream = segments[-1][0] if segments else "stdout"
            text = "".join(text for _, text in segments)
            return [
                WindowEvent(
                    stream,
                    text,
                    {
                        "bytes": skipped.bytes,
                        "newlines": skipped.newlines,
                        "endsWithNewline": skipped.ends_with_newline,
                    },
                )
            ]
        return [WindowEvent(stream, text, None) for stream, text in segments]


class Spill:
    """Spill of the complete raw output, started once it crosses either threshold, like Node's."""

    def __init__(self, after_bytes: int, after_lines: int) -> None:
        self.after_bytes = after_bytes
        self.after_lines = after_lines
        self.seen_bytes = 0
        self.seen_newlines = 0
        self.prefix: List[bytes] = []
        self.file: Optional[Any] = None
        self.path: Optional[str] = None
        self.failed: Optional[str] = None

    def push(self, chunk: bytes, tmpdir: str) -> None:
        if self.failed is not None or not chunk:
            return
        if self.file is None:
            self.seen_bytes += len(chunk)
            self.seen_newlines += chunk.count(b"\n")
            lines = self.seen_newlines + (0 if chunk[-1:] == b"\n" else 1)
            if self.seen_bytes <= self.after_bytes and lines <= self.after_lines:
                self.prefix.append(chunk)
                return
            try:
                self.start(tmpdir)
            except OSError as error:
                self.failed = str(error)
                return
        try:
            self.file.write(chunk)
        except OSError as error:
            self.failed = str(error)

    def start(self, tmpdir: str) -> None:
        """``createTempFile({ prefix: "pi-output-", suffix: ".log" })``: a fresh ``tmp-`` directory holding the file."""
        directory, name = os.path.split(os.path.join(tmpdir, "tmp-"))
        created = tempfile.mkdtemp(prefix=name, dir=directory or None)
        path = os.path.join(created, f"pi-output-{uuid.uuid4()}.log")
        opened = open(path, "ab")
        try:
            for chunk in self.prefix:
                opened.write(chunk)
        except OSError:
            opened.close()
            raise
        self.prefix = []
        self.file = opened
        self.path = path


def resolve_program(program: str, cwd: str, env: Dict[str, Any]) -> str:
    """The program an argv command runs. On POSIX the system searches ``PATH`` like ``execvp``; on Windows the search follows libuv (working directory, then ``PATH``, adding ``.com`` and ``.exe``), and batch files are refused."""
    if not WINDOWS:
        return program
    failure_not_found = lambda: Failure("spawn_error", f"spawn {program} ENOENT")  # noqa: E731

    def candidates(base: str) -> List[str]:
        names = []
        if os.path.splitext(base)[1]:
            names.append(base)
        for extension in ("com", "exe"):
            names.append(base + "." + extension)
        return names

    if "\\" in program or "/" in program or ":" in program:
        found = next((c for c in candidates(os.path.join(cwd, program)) if os.path.isfile(c)), None)
    else:
        path_variable = None
        for key, value in env.items():
            if key.upper() == "PATH" and isinstance(value, str):
                path_variable = value
                break
        if path_variable is None:
            path_variable = os.environ.get("PATH", "")
        found = None
        for directory in [cwd, *path_variable.split(os.pathsep)]:
            if not directory:
                continue
            found = next((c for c in candidates(os.path.join(directory, program)) if os.path.isfile(c)), None)
            if found is not None:
                break
    if found is None:
        raise failure_not_found()
    if found.lower().endswith((".bat", ".cmd")):
        raise Failure("spawn_error", f"spawn {program} EINVAL")
    return found


def child_environment(env: Dict[str, Any], inherit_env: bool) -> List[Tuple[str, str]]:
    """The child environment as Node builds it on Windows: the variables in object order, sorted by UTF-16 code units, keeping the first of names that differ only in case; then libuv adds required variables the result lacks."""
    entries: List[Tuple[str, str]] = []

    def set_value(key: str, value: str) -> None:
        for index, (name, _) in enumerate(entries):
            if name == key:
                entries[index] = (name, value)
                return
        entries.append((key, value))

    if inherit_env:
        for key, value in os.environ.items():
            set_value(key, value)
    for key, value in env.items():
        if isinstance(value, str):
            set_value(key, value)
    entries.sort(key=lambda item: item[0].encode("utf-16-be", "surrogatepass"))
    seen = set()
    unique = []
    for name, value in entries:
        if name.upper() in seen:
            continue
        seen.add(name.upper())
        unique.append((name, value))
    entries = unique
    for required in [
        "HOMEDRIVE",
        "HOMEPATH",
        "LOGONSERVER",
        "PATH",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    ]:
        if not any(name.upper() == required for name, _ in entries) and required in os.environ:
            entries.append((required, os.environ[required]))
    return entries


def kill_tree(pid: int) -> None:
    """Kill the command's process group, or the process if the group is gone."""
    if WINDOWS:
        system_root = os.environ.get("SystemRoot", "C:\\Windows")
        try:
            subprocess_module.run(
                [os.path.join(system_root, "System32", "taskkill.exe"), "/F", "/T", "/PID", str(pid)],
                stdin=subprocess_module.DEVNULL,
                stdout=subprocess_module.DEVNULL,
                stderr=subprocess_module.DEVNULL,
                creationflags=getattr(subprocess_module, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except OSError:
            pass
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


if WINDOWS:
    # libuv puts every child it starts into one job that ends its processes when the daemon
    # exits. Best effort: a failed assignment only means the taskkill paths must do the work.
    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
    _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    _job: List[Any] = [None, False]  # [handle, created]

    def _windows_job() -> Optional[int]:
        if _job[1]:
            return _job[0]
        _job[1] = True
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.SetInformationJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None
            information = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            information.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_BREAKAWAY_OK
                | _JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
                | _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            configured = kernel32.SetInformationJobObject(
                job,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(information),
                ctypes.sizeof(information),
            )
            if not configured:
                return None
            _job[0] = job
            return job
        except (AttributeError, OSError):
            return None

    def assign_to_job(pid: int) -> None:
        job = _windows_job()
        if not job:
            return
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            if not process:
                return
            try:
                kernel32.AssignProcessToJobObject(job, process)
            finally:
                kernel32.CloseHandle(process)
        except (AttributeError, OSError):
            pass

else:

    def assign_to_job(pid: int) -> None:  # pragma: no cover - POSIX has process groups
        return None


NAMES = ("stdout", "stderr")


def output_frame(request_id: int, stream: str, text: str, skipped: Optional[Dict[str, Any]]) -> bytes:
    json: Dict[str, Any] = {"kind": "output", "stream": stream}
    if skipped is not None:
        json["skipped"] = skipped
    return encode_frame(EVENT, request_id, json, text.encode("utf-8"))


def close_stream(stream: Any) -> None:
    """Stop reading a pipe, so it closes even if a descendant still holds it."""
    if stream is None:
        return
    transport = getattr(stream, "transport", None) or getattr(stream, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


async def run_exec(
    request_id: int,
    json: Dict[str, Any],
    tmpdir: str,
    output: Output,
    control: Control,
    groups: set,
    loop: asyncio.AbstractEventLoop,
    spill_executor: concurrent.futures.Executor,
) -> Dict[str, Any]:
    command = json.get("command")
    argv = json.get("argv")
    cwd = json.get("cwd")
    if not isinstance(cwd, str):
        raise Failure("EINVAL", "exec needs cwd")
    env_raw = json.get("env")
    env = env_raw if isinstance(env_raw, dict) else {}
    inherit_env = json.get("inheritEnv", True)
    shell_path = json.get("shellPath")
    timeout = json.get("timeoutMs")
    deadline = time.monotonic() + timeout / 1000.0 if isinstance(timeout, (int, float)) else None
    spill_options = json.get("spill")
    window = Window.from_json(json.get("window"))

    messages: asyncio.Queue = asyncio.Queue()
    control.sender = lambda kill: messages.put_nowait(("cancel", kill, None))
    if control.aborted:
        raise Failure("aborted", "aborted")

    # A string runs through the shell (as its last argument, or on stdin for legacy WSL bash); argv runs directly.
    stdin_command: Optional[str] = None
    if isinstance(command, str):
        config = await get_shell_config(shell_path)
        if not config.ok:
            raise Failure("shell_unavailable", str(config.error))
        shell: Any = config.value
        args = list(shell.args)
        if shell.command_transport == "stdin":
            program, args, stdin_command = shell.shell, args, command
        else:
            program = shell.shell
            args.append(command)
    elif isinstance(argv, list) and all(isinstance(item, str) for item in argv):
        if not argv:
            raise Failure("spawn_error", "Empty argv: no program to run")
        program = resolve_program(argv[0], cwd, env)
        args = list(argv[1:])
    else:
        raise Failure("EINVAL", "exec needs command or argv")

    if not os.path.exists(cwd):
        raise Failure(
            "spawn_error",
            f"Working directory does not exist: {cwd}\nCannot execute bash commands.",
        )

    try:
        if WINDOWS:
            environment = child_environment(env, inherit_env)
            process = await asyncio.create_subprocess_exec(
                program,
                *args,
                cwd=cwd,
                env=dict(environment),
                stdin=asyncio.subprocess.PIPE if stdin_command is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=getattr(subprocess_module, "CREATE_NO_WINDOW", 0),
            )
        else:
            if inherit_env:
                environment = dict(os.environ)
                environment.update({key: value for key, value in env.items() if isinstance(value, str)})
            else:
                environment = {key: value for key, value in env.items() if isinstance(value, str)}
            process = await asyncio.create_subprocess_exec(
                program,
                *args,
                cwd=cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE if stdin_command is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
    except (OSError, ValueError) as error:
        raise Failure("spawn_error", str(error)) from None

    if stdin_command is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin_command.encode("utf-8", "surrogatepass"))
            await process.stdin.drain()
        except (OSError, RuntimeError):
            pass
        finally:
            try:
                process.stdin.close()
            except (OSError, RuntimeError):
                pass

    pid = process.pid
    groups.add(pid)
    # Windows: put the child into the daemon's job, so it dies with the daemon even if cleanup
    # paths are skipped (libuv does the same). Best effort; failures leave the taskkill paths.
    assign_to_job(pid)
    stop = {"stopped": False}
    unsent = _Counter()

    async def pump(index: int, stream: Any) -> None:
        try:
            while True:
                chunk = await stream.read(64 * 1024)
                if not chunk:
                    return
                await messages.put(("data", index, chunk))
                # Without an output window, reading waits while too much output is unsent.
                if window is None:
                    await output.wait_for_room(lambda: stop["stopped"])
        finally:
            await messages.put(("eof", index, None))

    async def waiter() -> None:
        code = await process.wait()
        await messages.put(("exit", code, None))

    pump_tasks = [
        loop.create_task(pump(0, process.stdout)),
        loop.create_task(pump(1, process.stderr)),
    ]
    wait_task = loop.create_task(waiter())

    decoders = [StreamDecoder(), StreamDecoder()]
    spill = None
    if isinstance(spill_options, dict) and isinstance(spill_options.get("afterBytes"), int) and isinstance(
        spill_options.get("afterLines"), int
    ):
        spill = Spill(spill_options["afterBytes"], spill_options["afterLines"])
    pending = Pending(window) if window is not None else None

    def emit(stream_index: int, text: str) -> None:
        if text == "":
            return
        if pending is not None:
            pending.push(NAMES[stream_index], text)
        else:
            output.bulk(output_frame(request_id, NAMES[stream_index], text, None))

    def deliver() -> None:
        if pending is None:
            return
        for event in pending.take():
            output.bulk(output_frame(request_id, event.stream, event.text, event.skipped), unsent)

    ended = [False, False]
    status: Optional[int] = None
    idle_until: Optional[float] = None
    timed_out = False
    aborted = False
    killed = False

    def kill() -> None:
        nonlocal killed
        if not killed:
            killed = True
            kill_tree(pid)

    if control.killed:
        kill()

    try:
        while True:
            now = time.monotonic()
            # Node's timer stays armed until the command settles, including the grace period after exit.
            if not timed_out and deadline is not None and now >= deadline:
                timed_out = True
                kill()
            if status is not None and ended[0] and ended[1]:
                break
            if idle_until is not None and now >= idle_until:
                # A descendant may hold the pipes open after the process exited; stop waiting for it.
                break
            if pending is not None and not pending.is_empty() and now >= pending.next_send and unsent.count == 0:
                deliver()
            wait = 3600.0
            if deadline is not None and not timed_out:
                wait = min(wait, max(deadline - now, 0.0))
            if idle_until is not None:
                wait = min(wait, max(idle_until - now, 0.0))
            if pending is not None and not pending.is_empty():
                wait = min(wait, UNSENT_POLL if unsent.count > 0 else max(pending.next_send - now, 0.0))
            try:
                kind, first, second = await asyncio.wait_for(messages.get(), timeout=wait)
            except asyncio.TimeoutError:
                continue
            if kind == "data":
                emit(first, decoders[first].decode(second))
                if spill is not None:
                    await loop.run_in_executor(spill_executor, spill.push, second, tmpdir)
                    if spill.failed is not None:
                        kill()
                if status is not None:
                    idle_until = time.monotonic() + EXIT_STDIO_GRACE
            elif kind == "eof":
                ended[first] = True
            elif kind == "exit":
                status = first
                idle_until = time.monotonic() + EXIT_STDIO_GRACE
            elif kind == "cancel":
                if first:
                    kill()
                else:
                    aborted = True
                    kill()
    finally:
        groups.discard(pid)
        # Like Node destroying its streams: stop reading, so the pipes close even if a descendant
        # still holds them.
        stop["stopped"] = True
        for task in pump_tasks:
            task.cancel()
        await asyncio.gather(*pump_tasks, return_exceptions=True)
        wait_task.cancel()
        await asyncio.gather(wait_task, return_exceptions=True)
        close_stream(process.stdout)
        close_stream(process.stderr)
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if spill is not None and spill.file is not None:
            try:
                spill.file.close()
            except OSError:
                pass

    emit(0, decoders[0].decode())
    emit(1, decoders[1].decode())
    if pending is not None and not pending.is_empty():
        deliver()

    with_spill = lambda failure: failure.with_extra({"spillPath": spill.path}) if spill is not None and spill.path else failure  # noqa: E731

    if timed_out:
        raise with_spill(Failure("timeout", "timeout"))
    if aborted:
        raise with_spill(Failure("aborted", "aborted"))
    if spill is not None and spill.failed is not None:
        raise Failure("unknown", f"Failed to preserve complete shell output: {spill.failed}")
    if status is None:
        raise Failure("spawn_error", "process did not report its exit")
    # A process killed by a signal reports 128 + the signal, as the shell does (Python's returncode is negative then).
    result: Dict[str, Any] = {"exitCode": 128 - status if status < 0 else status}
    if spill is not None and spill.path is not None:
        result["spillPath"] = spill.path
    return result


# ---------------------------------------------------------------------------
# ``watch``: Durable's ``NodeFileWatcher`` (``LocalFileWatcher``) running next to
# the files, so a scan costs no round trips.
# ---------------------------------------------------------------------------


def utf16_key(path: str) -> bytes:
    """JavaScript's default sort: by UTF-16 code units."""
    return path.encode("utf-16-be", "surrogatepass")


def parse_targets(json: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = json.get("targets")
    if not isinstance(targets, list):
        raise Failure("EINVAL", "watch needs targets")
    parsed: List[Dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict) or not isinstance(target.get("path"), str):
            raise Failure("EINVAL", "watch needs targets")
        exclude = target.get("exclude") or {}
        parsed.append(
            {
                "path": target["path"],
                "recursive": target.get("recursive") is True,
                "hidden": exclude.get("hidden") is True,
                "names": [name for name in (exclude.get("names") or []) if isinstance(name, str)],
            }
        )
    return parsed


async def run_watch(request_id: int, json: Dict[str, Any], output: Output, control: Control) -> Dict[str, Any]:
    targets = parse_targets(json)
    mode = json.get("mode")
    if mode not in ("native", "polling"):
        # Windows refuses to rename a directory while a directory below it is open; the Python
        # watcher handles that natively, so it chooses like a local watcher would.
        mode = "auto"

    state = {"closed": False}
    finished = asyncio.Event()

    def send_change(change: Dict[str, Any]) -> None:
        if state["closed"]:
            return
        if "overflow" in change:
            output.control(
                encode_frame(EVENT, request_id, {"kind": "change", "overflow": True, "mode": "polling"})
            )
        elif "paths" in change:
            paths = sorted(change["paths"], key=utf16_key)
            output.control(encode_frame(EVENT, request_id, {"kind": "change", "paths": paths}))
        elif "error" in change:
            error = change["error"]
            output.control(
                encode_frame(
                    EVENT,
                    request_id,
                    {"kind": "error", "code": _node_error_code(error.code), "message": str(error)},
                )
            )
            # Nothing follows an error event, and the request finishes, as in the Rust daemon.
            state["closed"] = True
            finished.set()

    poll_interval = json.get("pollIntervalMs")
    max_directories = json.get("maxDirectories")
    try:
        watcher = await LocalFileWatcher.open(
            [WatchTarget(path=t["path"], recursive=t["recursive"], exclude={"hidden": t["hidden"], "names": t["names"]}) for t in targets],
            resolve_path=lambda path: path,
            on_change=send_change,
            mode=mode,
            poll_interval_ms=poll_interval if isinstance(poll_interval, int) else 2000,
            max_directories=max_directories if isinstance(max_directories, int) else 10_000,
        )
    except FileError as error:
        raise Failure(_node_error_code(error.code), str(error)) from None
    except OSError as error:
        raise io_failure(error, "watch", targets[0]["path"] if targets else "") from None
    try:
        output.control(encode_frame(EVENT, request_id, {"kind": "ready", "mode": watcher.mode}))
        if control.event is None:
            control.event = asyncio.Event()
        if control.aborted:
            return {}
        cancel_wait = asyncio.ensure_future(control.event.wait())
        finish_wait = asyncio.ensure_future(finished.wait())
        try:
            await asyncio.wait({cancel_wait, finish_wait}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (cancel_wait, finish_wait):
                task.cancel()
            await asyncio.gather(cancel_wait, finish_wait, return_exceptions=True)
        return {}
    finally:
        state["closed"] = True
        await watcher.close(None)


def _node_error_code(code: str) -> str:
    """A ``FileError`` code as the Node-style code the protocol and the client expect."""
    return {
        "permission_denied": "EACCES",
        "invalid": "EINVAL",
        "not_found": "ENOENT",
        "not_directory": "ENOTDIR",
        "is_directory": "EISDIR",
    }.get(code, code)


# ---------------------------------------------------------------------------
# Serving frames
# ---------------------------------------------------------------------------


def now_ms() -> float:
    return time.time() * 1000


def kill_all(groups: set) -> None:
    for pid in list(groups):
        kill_tree(pid)


async def serve_frames(server: Server, frames: "asyncio.Queue", loop: asyncio.AbstractEventLoop) -> None:
    while True:
        request = await frames.get()
        if request is None:
            return
        kind, request_id, json, payload = request
        if kind == REQUEST:
            # Registered before the request runs, so a cancel right behind it is not lost.
            control = Control()
            server.controls[request_id] = control
            op = json.get("op") if isinstance(json, dict) else None
            # Commands and watchers last; they run as tasks of their own (watchers recurse through trees).
            if op in ("exec", "watch"):
                server.spawn(loop.create_task(respond_long(server, request_id, op, json, control)))
            else:
                server.spawn(loop.create_task(respond_file(server, request_id, json, payload, control)))
        elif kind == CANCEL:
            control = server.controls.get(request_id)
            if control is not None:
                control.cancel(isinstance(json, dict) and json.get("mode") == "kill")
        # Pings and other kinds are ignored.


async def respond_file(server: Server, request_id: int, json: Any, payload: bytes, control: Control) -> None:
    loop = asyncio.get_running_loop()
    try:
        if not isinstance(json, dict):
            raise Failure("EINVAL", "invalid JSON")
        op = json.get("op")
        if op == "writeChunk" or op == "readdir":
            # Serialize per handle: chunk writes run one at a time in arrival order.
            if op == "writeChunk":
                handle = server.file_handle(json)
                if handle.lock is None:
                    handle.lock = asyncio.Lock()
                lock = handle.lock
            else:
                dir_handle = server.dir_handle(json)
                if dir_handle.lock is None:
                    dir_handle.lock = asyncio.Lock()
                lock = dir_handle.lock
            async with lock:
                result_json, result_payload = await loop.run_in_executor(
                    None, server.dispatch, json, payload, control
                )
        else:
            result_json, result_payload = await loop.run_in_executor(
                None, server.dispatch, json, payload, control
            )
        reply = encode_frame(RESULT, request_id, result_json, result_payload)
        if len(result_payload) > CONTROL_PAYLOAD:
            server.output.bulk(reply)
        else:
            server.output.control(reply)
    except Failure as failure:
        server.controls.pop(request_id, None)
        server.output.control(encode_frame(ERROR, request_id, failure.to_json()))
        return
    except asyncio.CancelledError:
        server.controls.pop(request_id, None)
        raise
    except BaseException as error:  # noqa: BLE001
        server.controls.pop(request_id, None)
        server.output.control(
            encode_frame(ERROR, request_id, Failure("unknown", str(error)).to_json())
        )
        return
    server.controls.pop(request_id, None)


async def respond_long(server: Server, request_id: int, op: str, json: Any, control: Control) -> None:
    try:
        if not isinstance(json, dict):
            raise Failure("EINVAL", "invalid JSON")
        if op == "exec":
            result = await run_exec(
                request_id,
                json,
                server.tmpdir,
                server.output,
                control,
                server.groups,
                asyncio.get_running_loop(),
                server.spill_executor,
            )
            # A command's result follows its output.
            server.output.bulk(encode_frame(RESULT, request_id, result))
        else:
            result = await run_watch(request_id, json, server.output, control)
            # Watch replies are small and must not queue behind command output.
            server.output.control(encode_frame(RESULT, request_id, result))
    except Failure as failure:
        if op == "exec":
            server.output.bulk(encode_frame(ERROR, request_id, failure.to_json()))
        else:
            server.output.control(encode_frame(ERROR, request_id, failure.to_json()))
    except asyncio.CancelledError:
        raise
    except BaseException as error:  # noqa: BLE001
        server.output.control(
            encode_frame(ERROR, request_id, Failure("unknown", str(error)).to_json())
        )
    finally:
        server.controls.pop(request_id, None)


def parse_frame(buffer: bytearray) -> Optional[Tuple[int, int, Any, bytes]]:
    """Parse one frame from ``buffer``; ``None`` when more bytes are needed. Raises ``ValueError`` on corrupt framing."""
    if len(buffer) < 4:
        return None
    length = int.from_bytes(buffer[:4], "big")
    if length < 9 or length > 16 * 1024 * 1024:
        raise ValueError("Corrupt frame")
    if len(buffer) < 4 + length:
        return None
    # A view over the buffer; released before the buffer is resized.
    view = memoryview(buffer)
    body = view[4 : 4 + length]
    try:
        kind = body[0]
        request_id = int.from_bytes(body[1:5], "big")
        json_length = int.from_bytes(body[5:9], "big")
        if 9 + json_length > length:
            raise ValueError("Corrupt frame")
        try:
            json = json_module.loads(bytes(body[9 : 9 + json_length]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Corrupt frame") from None
        payload = bytes(body[9 + json_length :])
    finally:
        body.release()
        view.release()
    del buffer[: 4 + length]
    return kind, request_id, json, payload


async def serve(token: str) -> None:
    stream = sys.stdout.buffer
    stream.write(f"PI-ENV {token}\n".encode("ascii"))
    stream.flush()

    loop = asyncio.get_running_loop()
    output = Output(loop, stream)
    groups: set = set()
    server = Server(output, node_tmpdir(), groups)
    last_seen = [now_ms()]

    def stdin_reader() -> None:
        raw = sys.stdin.buffer
        try:
            while True:
                # ``read1`` returns after one system read; ``read`` would wait for the full size.
                chunk = raw.read1(65536)
                if not chunk:
                    break
                last_seen[0] = now_ms()
                _post(frames_in, chunk)
        except (OSError, ValueError):
            pass
        _post(frames_in, None)

    def _post(queue: asyncio.Queue, item: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass  # The loop is gone; the reader thread is about to end with the process.

    frames_in: "asyncio.Queue" = asyncio.Queue()
    frames: "asyncio.Queue" = asyncio.Queue()

    async def parse_frames() -> None:
        buffer = bytearray()
        while True:
            chunk = await frames_in.get()
            if chunk is None:
                break
            buffer.extend(chunk)
            try:
                while True:
                    parsed = parse_frame(buffer)
                    if parsed is None:
                        break
                    frames.put_nowait(parsed)
            except ValueError:
                frames.put_nowait(None)
                return
        frames.put_nowait(None)

    ping_task = loop.create_task(ping_loop(output, last_seen, groups))
    parser_task = loop.create_task(parse_frames())
    threading.Thread(target=stdin_reader, daemon=True).start()

    try:
        await serve_frames(server, frames, loop)
    finally:
        # However input ended, the client is gone: stop everything it started.
        kill_all(groups)
        parser_task.cancel()
        ping_task.cancel()
        server.spill_executor.shutdown(wait=False)
        output.shutdown()


async def ping_loop(output: Output, last_seen: List[float], groups: set) -> None:
    while True:
        await asyncio.sleep(PING_INTERVAL)
        output.control(encode_frame(PING, 0, {}))
        # A client that went silent (a phone that lost its network) leaves nothing running.
        if now_ms() - last_seen[0] > SILENCE_LIMIT_MS:
            kill_all(groups)
            os._exit(0)


def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == "serve":
        token = None
        for index, arg in enumerate(args):
            if arg == "--token" and index + 1 < len(args):
                token = args[index + 1]
                break
        if token is None:
            print("usage: pi-env serve --token <hex>", file=sys.stderr)
            sys.exit(2)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(serve(token))
        except KeyboardInterrupt:
            pass
        finally:
            try:
                loop.close()
            except Exception:
                pass
        # Request threads may still be blocked (a FIFO open); nothing is left to answer them.
        # ``asyncio.run`` would wait for the default executor on newer Pythons; exit instead.
        os._exit(0)
    if len(args) >= 1 and args[0] == "--version":
        print(f"pi-env {VERSION}")
        return
    print("usage: pi-env serve --token <hex>", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
