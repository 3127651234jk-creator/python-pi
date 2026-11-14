"""Native filesystem watch backends behind ``LocalFileWatcher``.

One backend per platform, all standard library:
- Windows: ``ReadDirectoryChangesW`` with one async recursive handle per watched
  directory, each serviced by a waiting thread.
- Linux: ``inotify`` with one watch per directory (inotify has no recursion)
  and one reader thread.
- macOS: FSEvents with one stream over the watched paths and a run-loop thread.

Backends only report "something changed at or under this path": the watcher
rescans the watched trees and diffs snapshots, so no report ever depends on
which events an operating system chooses to deliver.

``on_event(directory, filename)`` may fire from backend threads:

- both arguments: ``filename`` names an entry (possibly with intermediate
  components) relative to the watched ``directory``;
- ``filename`` only: ``directory`` is the full path of what changed, unnamed;
- ``directory`` only: a pure wake-up, nothing is known.

Callers must keep it cheap and thread-safe.
"""

from __future__ import annotations

import errno
import os
import queue
import select
import struct
import sys
import threading
from typing import Callable, Dict, List, Optional, Set

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
DARWIN = sys.platform == "darwin"

EventCallback = Callable[[Optional[str], Optional[str]], None]


def ancestors_of(path: str) -> List[str]:
    """Ancestors of ``path``, nearest first, up to the root."""
    result: List[str] = []
    current = os.path.dirname(path)
    while True:
        result.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            return result
        current = parent


class WindowsWatchBackend:  # pragma: no cover - exercised on Windows only
    """One async recursive ``ReadDirectoryChangesW`` handle per directory."""

    def __init__(self, on_event: EventCallback) -> None:
        self._on_event = on_event
        self._closed = False
        self._watches: Dict[str, "_WindowsWatch"] = {}

    def watch(self, directory: str) -> None:
        if self._closed:
            return
        handle = _open_directory_handle(directory)
        overlapped = None
        stop = threading.Event()
        try:
            buffer = ctypes.create_string_buffer(_BUFFER_SIZE)
            overlapped = _Overlapped()
            overlapped.hEvent = _kernel32.CreateEventW(None, True, False, None)
            if not overlapped.hEvent:
                raise OSError(errno.EIO, f"CreateEventW failed ({ctypes.GetLastError()})")
            self._issue(handle, buffer, overlapped)
        except BaseException:
            _kernel32.CloseHandle(handle)
            if overlapped is not None and overlapped.hEvent:
                _kernel32.CloseHandle(overlapped.hEvent)
            raise
        watch = _WindowsWatch(
            thread=threading.Thread(
                target=self._run, args=(directory, handle, buffer, overlapped, stop), daemon=True
            ),
            handle=handle,
            buffer=buffer,
            overlapped=overlapped,
            stop=stop,
        )
        self._watches[directory] = watch
        watch.thread.start()

    def unwatch(self, directory: str) -> None:
        watch = self._watches.pop(directory, None)
        if watch is None:
            return
        self._stop(watch)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
        for watch in list(self._watches.values()):
            self._stop(watch)
        self._watches.clear()

    def _stop(self, watch: "_WindowsWatch") -> None:
        """Cancel the pending read and release both handles.

        ``CancelIoEx``/``CloseHandle`` on the directory alone leaves the
        overlapped event handle allocated, which accumulates for the lifetime
        of a long-running worker. Wake the waiting thread first so it can
        observe the cancellation, then close the event it was waiting on.
        """
        watch.stop.set()
        _kernel32.CancelIoEx(watch.handle, None)
        event = watch.overlapped.hEvent if watch.overlapped is not None else None
        if event:
            _kernel32.SetEvent(event)
        watch.thread.join(timeout=1.0)
        _kernel32.CloseHandle(watch.handle)
        if event:
            _kernel32.CloseHandle(event)
            watch.overlapped.hEvent = None

    def _issue(self, handle: int, buffer, overlapped: "_Overlapped") -> None:
        _kernel32.ResetEvent(overlapped.hEvent)
        if not _kernel32.ReadDirectoryChangesW(
            handle, buffer, _BUFFER_SIZE, True, _NOTIFY_FILTERS, None, ctypes.byref(overlapped), None
        ):
            winerror = ctypes.GetLastError()
            raise OSError(_WIN32_ERRNOS.get(winerror, errno.EIO), f"ReadDirectoryChangesW failed ({winerror})")

    def _run(self, directory: str, handle: int, buffer, overlapped: "_Overlapped",
             stop: threading.Event) -> None:
        while not self._closed and not stop.is_set():
            _kernel32.WaitForSingleObject(overlapped.hEvent, _INFINITE)
            if self._closed or stop.is_set():
                break
            nbytes = wintypes.DWORD()
            if not _kernel32.GetOverlappedResult(
                handle, ctypes.byref(overlapped), ctypes.byref(nbytes), False
            ):
                error = ctypes.GetLastError()
                if error != _ERROR_MORE_DATA:
                    # Aborted by close, or the directory is gone: one wake-up so the
                    # watcher's rescan reconciles coverage.
                    if not self._closed and not stop.is_set():
                        self._on_event(directory, None)
                    break
            if self._closed or stop.is_set():
                break
            for name in _parse_change_names(buffer, nbytes.value):
                self._on_event(directory, name)
            if self._closed or stop.is_set():
                break
            try:
                self._issue(handle, buffer, overlapped)
            except OSError:
                if not self._closed and not stop.is_set():
                    self._on_event(directory, None)
                break


class _WindowsWatch:
    def __init__(self, thread: threading.Thread, handle: int, buffer, overlapped,
                 stop: threading.Event) -> None:
        self.thread = thread
        self.handle = handle
        self.buffer = buffer
        self.overlapped = overlapped
        self.stop = stop


if WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32

    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_SHARE_ALL = 0x1 | 0x2 | 0x4  # read | write | delete
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OVERLAPPED = 0x40000000
    _INFINITE = 0xFFFFFFFF
    _ERROR_MORE_DATA = 234
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    _FILE_NOTIFY_CHANGE_FILE_NAME = 0x1
    _FILE_NOTIFY_CHANGE_DIR_NAME = 0x2
    _FILE_NOTIFY_CHANGE_ATTRIBUTES = 0x4
    _FILE_NOTIFY_CHANGE_SIZE = 0x8
    _FILE_NOTIFY_CHANGE_LAST_WRITE = 0x10
    _NOTIFY_FILTERS = (
        _FILE_NOTIFY_CHANGE_FILE_NAME
        | _FILE_NOTIFY_CHANGE_DIR_NAME
        | _FILE_NOTIFY_CHANGE_ATTRIBUTES
        | _FILE_NOTIFY_CHANGE_SIZE
        | _FILE_NOTIFY_CHANGE_LAST_WRITE
    )

    _WIN32_ERRNOS = {
        2: errno.ENOENT,  # ERROR_FILE_NOT_FOUND
        3: errno.ENOENT,  # ERROR_PATH_NOT_FOUND
        5: errno.EACCES,  # ERROR_ACCESS_DENIED
        32: errno.EACCES,  # ERROR_SHARING_VIOLATION
    }

    _BUFFER_SIZE = 64 * 1024

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.POINTER(ctypes.c_ulong)),
            ("InternalHigh", ctypes.POINTER(ctypes.c_ulong)),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32.CreateFileW.restype = ctypes.c_void_p
    _kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    _kernel32.CreateEventW.restype = ctypes.c_void_p
    _kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, ctypes.c_wchar_p]
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _kernel32.ResetEvent.argtypes = [ctypes.c_void_p]
    _kernel32.CancelIoEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Overlapped)]
    _kernel32.GetOverlappedResult.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Overlapped),
        ctypes.POINTER(wintypes.DWORD),
        wintypes.BOOL,
    ]
    _kernel32.ReadDirectoryChangesW.restype = wintypes.BOOL
    _kernel32.ReadDirectoryChangesW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(_Overlapped),
        ctypes.c_void_p,
    ]

    def _open_directory_handle(directory: str) -> int:
        # Full sharing, so the watched directory itself stays renameable and removable.
        handle = _kernel32.CreateFileW(
            directory,
            _FILE_LIST_DIRECTORY,
            _FILE_SHARE_ALL,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OVERLAPPED,
            None,
        )
        if handle in (None, _INVALID_HANDLE_VALUE):
            winerror = ctypes.GetLastError()
            raise OSError(_WIN32_ERRNOS.get(winerror, errno.EIO), f"Cannot watch {directory} ({winerror})")
        return handle

    def _parse_change_names(buffer, nbytes: int) -> List[str]:
        """File names in one ``FILE_NOTIFY_INFORMATION`` batch; actions are ignored."""
        data = bytes(memoryview(buffer)[:nbytes])
        names: List[str] = []
        offset = 0
        while offset + 12 <= len(data):
            next_offset = int.from_bytes(data[offset : offset + 4], "little")
            name_length = int.from_bytes(data[offset + 8 : offset + 12], "little")
            if name_length:
                raw = data[offset + 12 : offset + 12 + name_length]
                try:
                    names.append(raw.decode("utf-16-le"))
                except UnicodeDecodeError:
                    break  # A torn record: the rescan catches what this one missed.
            if next_offset == 0:
                break
            offset += next_offset
        return names


if LINUX:
    import ctypes

    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.inotify_init1.restype = ctypes.c_int
    _libc.inotify_init1.argtypes = [ctypes.c_int]
    _libc.inotify_add_watch.restype = ctypes.c_int
    _libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    _libc.inotify_rm_watch.restype = ctypes.c_int
    _libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]

    _IN_NONBLOCK = 0o4000
    _IN_CLOEXEC = 0o2000000

    _IN_MODIFY = 0x2
    _IN_ATTRIB = 0x4
    _IN_CLOSE_WRITE = 0x8
    _IN_MOVED_FROM = 0x40
    _IN_MOVED_TO = 0x80
    _IN_CREATE = 0x100
    _IN_DELETE = 0x200
    _IN_DELETE_SELF = 0x400
    _IN_MOVE_SELF = 0x800
    _IN_UNMOUNT = 0x2000
    _IN_Q_OVERFLOW = 0x4000
    _IN_IGNORED = 0x8000
    _WATCH_MASK = (
        _IN_MODIFY
        | _IN_ATTRIB
        | _IN_CLOSE_WRITE
        | _IN_MOVED_FROM
        | _IN_MOVED_TO
        | _IN_CREATE
        | _IN_DELETE
        | _IN_DELETE_SELF
        | _IN_MOVE_SELF
    )

    _EVENT_HEADER = struct.Struct("=iIII")

    _statfs = _libc.statfs
    _statfs.restype = ctypes.c_int
    _statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]

    def _statfs_type(path: str) -> Optional[int]:
        """``f_type`` of the file system holding ``path``, or ``None`` if unknown."""
        buffer = (ctypes.c_long * 1)()
        if _statfs(os.fsencode(path), buffer) != 0:
            return None
        return buffer[0]

    # Linux statfs magic numbers of file systems that accept watches but do not report
    # changes made elsewhere (same list as the TS implementation).
    UNRELIABLE_FILE_SYSTEMS = frozenset(
        {
            0x6969,  # NFS
            0x517B,  # SMB
            0xFF534D42,  # CIFS
            0xFE534D42,  # SMB2
            0x65735546,  # FUSE (sshfs, Android shared storage)
            0x01021997,  # 9P (WSL2 Windows drives)
            0x0BD00BD0,  # Lustre
            0x47504653,  # GPFS
            0x00C36400,  # Ceph
            0x5346414F,  # OpenAFS
            0x6B414653,  # kAFS
            0x5DCA2DF5,  # sdcardfs
        }
    )

    def any_unreliable_file_system(paths: List[str]) -> bool:
        """Whether any path, or its nearest existing ancestor, sits on a file system
        that does not report remote changes."""
        for path in paths:
            for candidate in [path, *ancestors_of(path)]:
                fs_type = _statfs_type(candidate)
                if fs_type is None:
                    continue
                if fs_type in UNRELIABLE_FILE_SYSTEMS:
                    return True
                break
        return False


else:

    def any_unreliable_file_system(paths: List[str]) -> bool:
        """Off Linux every file system that native watching supports reports changes."""
        return False


class InotifyWatchBackend:  # pragma: no cover - exercised on Linux only
    """One inotify watch per directory, one reader thread for all of them."""

    def __init__(self, on_event: EventCallback) -> None:
        self._on_event = on_event
        self._fd = _libc.inotify_init1(_IN_CLOEXEC | _IN_NONBLOCK)
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_r, False)
        os.set_blocking(self._wake_w, False)
        self._dirs: Dict[int, str] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def watch(self, directory: str) -> None:
        if self._closed:
            return
        wd = _libc.inotify_add_watch(self._fd, os.fsencode(directory), _WATCH_MASK)
        if wd < 0:
            raise OSError(ctypes.get_errno(), f"Cannot watch {directory}")
        self._dirs[wd] = directory

    def unwatch(self, directory: str) -> None:
        for wd, path in list(self._dirs.items()):
            if path == directory:
                del self._dirs[wd]
                _libc.inotify_rm_watch(self._fd, wd)  # The IN_IGNORED is not mapped any more.

    def commit(self) -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.write(self._wake_w, b"x")
        except OSError:
            pass
        self._thread.join(timeout=2.0)
        for fd in (self._fd, self._wake_r, self._wake_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def _run(self) -> None:
        while not self._closed:
            try:
                ready, _, _ = select.select([self._fd, self._wake_r], [], [], 1.0)
            except (OSError, ValueError):
                break
            if self._closed:
                break
            if self._wake_r in ready:
                try:
                    os.read(self._wake_r, 4096)
                except OSError:
                    pass
                continue
            if self._fd not in ready:
                continue
            try:
                data = os.read(self._fd, 64 * 1024)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                break
            self._consume(data)

    def _consume(self, data: bytes) -> None:
        offset = 0
        while offset + _EVENT_HEADER.size <= len(data):
            wd, mask, _cookie, name_length = _EVENT_HEADER.unpack_from(data, offset)
            name_raw = data[offset + _EVENT_HEADER.size : offset + _EVENT_HEADER.size + name_length]
            offset += _EVENT_HEADER.size + name_length
            if mask & _IN_Q_OVERFLOW:
                self._on_event(None, None)  # Lost events: wake the rescan.
                continue
            directory = self._dirs.get(wd)
            if directory is None:
                continue
            if mask & _IN_IGNORED:
                self._on_event(directory, None)
                continue
            if mask & (_IN_DELETE_SELF | _IN_MOVE_SELF | _IN_UNMOUNT):
                self._on_event(directory, None)
                continue
            name: Optional[str] = None
            if name_length:
                name = name_raw.split(b"\0")[0].decode("utf-8", "replace")
            self._on_event(directory, name or None)


if DARWIN:  # pragma: no cover - exercised on macOS only
    import ctypes

    _cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    _fs_events = ctypes.CDLL(
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/FSEvents.framework/FSEvents"
    )

    _cf.CFArrayCreate.restype = ctypes.c_void_p
    _cf.CFArrayCreate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_long,
        ctypes.c_void_p,
    ]
    _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    _cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    _cf.CFRunLoopGetCurrent.restype = ctypes.c_void_p
    _cf.CFRunLoopRunInMode.restype = ctypes.c_int
    _cf.CFRunLoopRunInMode.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
    _cf.CFRunLoopStop.argtypes = [ctypes.c_void_p]
    _kCFRunLoopDefaultMode = ctypes.c_void_p.in_dll(_cf, "kCFRunLoopDefaultMode")
    _kCFTypeArrayCallBacks = ctypes.c_void_p.in_dll(_cf, "kCFTypeArrayCallBacks")

    _fs_events.FSEventStreamCreate.restype = ctypes.c_void_p
    _fs_events.FSEventStreamCreate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_uint32,
    ]
    _fs_events.FSEventStreamScheduleWithRunLoop.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _fs_events.FSEventStreamStart.restype = ctypes.c_bool
    _fs_events.FSEventStreamStart.argtypes = [ctypes.c_void_p]
    _fs_events.FSEventStreamStop.argtypes = [ctypes.c_void_p]
    _fs_events.FSEventStreamInvalidate.argtypes = [ctypes.c_void_p]
    _fs_events.FSEventStreamRelease.argtypes = [ctypes.c_void_p]

    _K_FSEVENT_STREAM_CREATE_FLAG_NO_DEFER = 0x8
    _K_FSEVENT_STREAM_CREATE_FLAG_WATCH_ROOT = 0x4
    _K_FSEVENT_STREAM_CREATE_FLAG_FILE_EVENTS = 0x10
    _K_FSEVENT_STREAM_EVENT_ID_SINCE_NOW = (1 << 64) - 1

    _FSEventStreamCallback = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # client info
        ctypes.c_ulong,  # num events
        ctypes.POINTER(ctypes.c_char_p),  # event paths
        ctypes.POINTER(ctypes.c_uint32),  # event flags
        ctypes.POINTER(ctypes.c_uint64),  # event ids
    )

    def _cf_string(text: str) -> int:
        pointer = _cf.CFStringCreateWithCString(None, text.encode("utf-8"), 0x08000100)  # UTF-8
        if not pointer:
            raise OSError("CFStringCreateWithCString failed")
        return pointer

    def _cf_array(paths: Set[str]) -> int:
        strings = [_cf_string(path) for path in sorted(paths)]
        values = (ctypes.c_void_p * len(strings))(*strings)
        array = _cf.CFArrayCreate(None, values, len(strings), _kCFTypeArrayCallBacks)
        if not array:
            raise OSError("CFArrayCreate failed")
        return array


class FSEventsWatchBackend:  # pragma: no cover - exercised on macOS only
    """One FSEvents stream over the watched paths; membership changes restart it."""

    def __init__(self, on_event: EventCallback) -> None:
        self._on_event = on_event
        self._paths: Set[str] = set()
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._closed = False
        self._stream: Optional[int] = None
        self._runloop: Optional[int] = None
        self._callback = _FSEventStreamCallback(self._dispatch)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def watch(self, directory: str) -> None:
        self._paths.add(directory)
        self._commands.put("restart")

    def unwatch(self, directory: str) -> None:
        self._paths.discard(directory)
        self._commands.put("restart")

    def commit(self) -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._commands.put("stop")
        self._thread.join(timeout=2.0)

    def _dispatch(self, _stream, _info, count, paths, _flags, _ids) -> None:
        try:
            for index in range(count):
                raw = paths[index]
                if raw:
                    self._on_event(raw.decode("utf-8", "replace"), None)
        except Exception:
            pass  # A throwing callback must not stop the stream.

    def _run(self) -> None:
        self._runloop = _cf.CFRunLoopGetCurrent()
        while not self._closed:
            try:
                self._serve()
            except Exception:
                # The stream is dead: one wake-up, and the watcher's rescan timer
                # keeps reporting until it re-arms watches.
                self._on_event(None, None)
                break
        if self._stream is not None:
            self._teardown_stream()
        if self._runloop is not None:
            _cf.CFRunLoopStop(self._runloop)

    def _serve(self) -> None:
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            command = None
        if command == "restart":
            self._restart()
        elif command == "stop":
            self._closed = True
            return
        if self._closed:
            return
        _cf.CFRunLoopRunInMode(_kCFRunLoopDefaultMode, 0.1, False)

    def _restart(self) -> None:
        if self._stream is not None:
            self._teardown_stream()
        if not self._paths:
            return
        array = _cf_array(self._paths)
        self._stream = _fs_events.FSEventStreamCreate(
            None,
            self._callback,
            None,
            array,
            _K_FSEVENT_STREAM_EVENT_ID_SINCE_NOW,
            0.05,
            _K_FSEVENT_STREAM_CREATE_FLAG_NO_DEFER
            | _K_FSEVENT_STREAM_CREATE_FLAG_WATCH_ROOT
            | _K_FSEVENT_STREAM_CREATE_FLAG_FILE_EVENTS,
        )
        if not self._stream:
            raise OSError("FSEventStreamCreate failed")
        _fs_events.FSEventStreamScheduleWithRunLoop(self._stream, self._runloop, _kCFRunLoopDefaultMode)
        if not _fs_events.FSEventStreamStart(self._stream):
            self._teardown_stream()
            raise OSError("FSEventStreamStart failed")

    def _teardown_stream(self) -> None:
        stream, self._stream = self._stream, None
        _fs_events.FSEventStreamStop(stream)
        _fs_events.FSEventStreamInvalidate(stream)
        _fs_events.FSEventStreamRelease(stream)


def default_backend() -> Optional[type]:
    """The backend class for this platform, or ``None`` when native watching is unsupported."""
    if WINDOWS:
        return WindowsWatchBackend
    if LINUX:
        return InotifyWatchBackend
    if DARWIN:
        return FSEventsWatchBackend
    return None


def read_shared(path: str) -> Optional[bytes]:
    """Whole-file read that tolerates writers holding the file open on Windows.

    Snapshot comparison hashes files like this, so a rewrite the file system's
    timestamp granularity hides is still seen even while the writer keeps the
    file open. Returns ``None`` when the file cannot be read at all.
    """
    try:
        if WINDOWS:
            handle = _kernel32.CreateFileW(
                path,
                0x80000000,  # GENERIC_READ
                _FILE_SHARE_ALL,
                None,
                _OPEN_EXISTING,
                0x80,  # FILE_ATTRIBUTE_NORMAL
                None,
            )
            if handle in (None, _INVALID_HANDLE_VALUE):
                return None
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
                with open(descriptor, "rb", closefd=True) as file_handle:
                    return file_handle.read()
            except OSError:
                _kernel32.CloseHandle(handle)
                return None
        with open(path, "rb") as file_handle:
            return file_handle.read()
    except OSError:
        return None
