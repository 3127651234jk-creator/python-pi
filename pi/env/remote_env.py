"""A Durable ``ExecutionEnv`` on another machine, reached through a pi-env daemon
(port of ``packages/env/src/remote-env.ts``).

Results match ``NodeExecutionEnv`` running on the remote machine: the daemon
performs the system calls, and this class applies Node's path, error, and
result rules on the client. Path resolution (``~``, ``file://``, relative to
``cwd``, Windows drive-relative paths), abort checkpoints, and error code
mapping happen here; system calls, processes, spill files, and line scans
happen in the daemon (see the daemon's ``semantics`` notes).

Porting deviations from the TS implementation:
- ``Context`` carries an ``asyncio.Event`` abort signal;
- aborts cancel in-flight ``pread``/``writeChunk`` tasks explicitly, where TS
  leaves their promises to be garbage-collected;
- ``file://`` URLs are resolved with a small local helper (malformed URLs stay
  ordinary paths, as in TS).
"""

from __future__ import annotations

import asyncio
import math
import ntpath
import os
import posixpath
import re
import uuid as uuid_module
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from ..durable.env import (
    Context,
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
    WatchChange,
    WatchTarget,
    err,
    ok,
    range_decoder,
)
from .connection import Connection, Json, RemoteError
from .errors import abort_result, to_file_error
from .watch import RemoteWatchOptions, RemoteWatcher

MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000
# Bytes per ``pread`` request of a binary reader; several are in flight at once.
READ_CHUNK = 256 * 1024
READ_DEPTH = 8
# Node's ``writeFile`` writes at most this much per call and checks for an abort before each.
WRITE_CHUNK = 512 * 1024
WRITE_DEPTH = 8
# Node's ``readFile``: chunk of known-size reads, chunk of unknown-size reads, and the largest file it reads.
READ_FILE_CHUNK = 512 * 1024
READ_FILE_UNKNOWN_CHUNK = 64 * 1024
READ_FILE_MAX = 2**31 - 1
# ``NodeTextLineReader``'s read size.
LINE_CHUNK = 64 * 1024


def _well_formed(text: str) -> str:
    """Lone surrogates become U+FFFD, as Node encodes strings for system calls."""
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


class RemotePaths:
    """Node's path rules of the remote system (``node:path`` posix or win32)."""

    def __init__(self, windows: bool) -> None:
        self.windows = windows
        self._mod = ntpath if windows else posixpath

    def join(self, *parts: str) -> str:
        return self._mod.join(*parts)

    def basename(self, path: str) -> str:
        return self._mod.basename(path)

    def is_absolute(self, path: str) -> bool:
        return self._mod.isabs(path)

    def resolve_absolute(self, path: str) -> str:
        if self.windows:
            # Node resolves a path rooted without a drive ("\\x") against the drive of the
            # resolving process's working directory, which here is the client's.
            if path[:1] in ("/", "\\") and not (len(path) >= 2 and path[1] in ("/", "\\")):
                drive = ntpath.splitdrive(os.getcwd())[0]
                if drive:
                    path = drive + path
            return ntpath.normpath(path)
        return posixpath.normpath(path)

    def resolve_relative(self, base: str, path: str) -> str:
        if self.windows:
            return ntpath.normpath(ntpath.join(base, path))
        return posixpath.normpath(posixpath.join(base, path))


async def remote_path(connection: Connection) -> RemotePaths:
    """Node's path rules of the remote system."""
    return RemotePaths((await connection.info()).os == "windows")


def _file_url_to_path(url: str, windows: bool) -> Optional[str]:
    """``fileURLToPath`` for the remote system; ``None`` keeps the URL an ordinary path."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "file":
            return None
        host = parsed.hostname or ""
        path = unquote(parsed.path)
        if windows:
            if host and host != "localhost":
                return "\\\\" + host + path.replace("/", "\\")
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return path.replace("/", "\\")
        if host and host != "localhost":
            return "//" + host + path
        return path
    except ValueError:
        return None


def _to_info(path: str, remote: Json, paths: RemotePaths) -> Result:
    kind = remote.get("kind")
    if kind == "other":
        return err(FileError("invalid", "Unsupported file type", path))
    return ok(
        FileInfo(
            name=paths.basename(path),
            path=path,
            kind=kind,
            size=remote.get("size", 0),
            mtime_ms=remote.get("mtimeSec", 0) * 1000 + remote.get("mtimeNsec", 0) / 1e6,
        )
    )


def _concat(chunks: List[bytes]) -> bytes:
    if len(chunks) == 1:
        return chunks[0]
    return b"".join(chunks)


async def _cancel_pending(tasks: List["asyncio.Task"]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def read_range(
    read: Callable[[int, int], Awaitable[bytes]],
    offset: int,
    length: int,
    chunk: int,
    aborted: Callable[[], bool],
) -> Optional[Tuple[List[bytes], int]]:
    """
    Read ``length`` bytes from ``offset`` as consecutive reads of at most ``chunk`` bytes, several in
    flight. Ends early at the end of the file. A short read continues from where it ended, as
    sequential reads would. ``aborted`` is checked after each read; ``None`` means aborted.
    """
    chunks: List[bytes] = []
    total = 0
    next_offset = offset
    in_flight: List[Tuple[int, "asyncio.Task[bytes]"]] = []
    try:
        while total < length:
            while len(in_flight) < READ_DEPTH and next_offset < offset + length:
                size = min(offset + length - next_offset, chunk)
                task = asyncio.ensure_future(read(next_offset, size))
                in_flight.append((size, task))
                next_offset += size
            head_size, head = in_flight.pop(0)
            data = await head
            if aborted():
                await _cancel_pending([task for _, task in in_flight])
                return None
            if len(data) == 0:
                break
            chunks.append(data)
            total += len(data)
            if len(data) < head_size:
                # Reads already in flight assumed a full read; continue right after this one.
                await _cancel_pending([task for _, task in in_flight])
                in_flight = []
                next_offset = offset + total
        return chunks, total
    except BaseException:
        await _cancel_pending([task for _, task in in_flight])
        raise


class _Handle:
    """A daemon handle: valid only in the session that opened it."""

    def __init__(self, env: "RemoteExecutionEnv", handle_id: int, session: int, path: str) -> None:
        self.env = env
        self.id = handle_id
        self.session = session
        self.path = path

    async def request(
        self,
        op: str,
        json: Optional[Json] = None,
        payload: Optional[bytes] = None,
        abort_signal: Optional[asyncio.Event] = None,
    ) -> Any:
        return await self.env.connection.request(
            op,
            {**(json or {}), "handle": self.id},
            payload=payload,
            abort_signal=abort_signal,
            session=self.session,
        )

    async def pread(self, offset: Optional[int], length: int) -> bytes:
        json = {"length": length}
        if offset is not None:
            json["offset"] = offset
        return (await self.request("pread", json)).payload

    async def close(self) -> None:
        try:
            await self.request("close")
        except RemoteError:
            return


class RemoteBinaryReader:
    """Positional reads from one opened regular file, served by ``pread`` requests."""

    def __init__(self, handle: _Handle) -> None:
        self._handle = handle
        self._closed = False

    @property
    def _path(self) -> str:
        return self._handle.path

    def _closed_result(self) -> Optional[Result]:
        if self._closed:
            return err(FileError("invalid", "Binary reader is closed", self._path))
        return None

    async def info(self, context: Context) -> Result:
        early = abort_result(context, self._path) or self._closed_result()
        if early is not None:
            return early
        try:
            reply = await self._handle.request("fstat")
            paths = await remote_path(self._handle.env.connection)
            return _to_info(self._path, reply.json, paths)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, self._path))

    async def read(self, offset: int, length: int, context: Context) -> Result:
        early = abort_result(context, self._path) or self._closed_result()
        if early is not None:
            return early
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or not isinstance(length, int)
            or isinstance(length, bool)
            or offset < 0
            or length < 0
        ):
            return err(FileError("invalid", "Offset and length must be non-negative safe integers", self._path))
        try:
            read = await read_range(
                lambda at, size: self._handle.pread(at, size),
                offset,
                length,
                READ_CHUNK,
                lambda: context.abort_signal is not None and context.abort_signal.is_set(),
            )
            if read is None:
                return err(FileError("aborted", "aborted", self._path))
            return ok(_concat(read[0]))
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, self._path))

    async def scan_lines(self, options: Dict[str, int], context: Context) -> Result:
        early = abort_result(context, self._path) or self._closed_result()
        if early is not None:
            return early
        start_line = options.get("start_line")
        end_line = options.get("end_line")
        if (
            not isinstance(start_line, int)
            or isinstance(start_line, bool)
            or start_line < 0
            or (end_line is not None and (not isinstance(end_line, int) or isinstance(end_line, bool) or end_line <= start_line))
        ):
            return err(FileError("invalid", "Invalid line range", self._path))
        json: Json = {"startLine": start_line}
        if end_line is not None:
            json["endLine"] = end_line
        try:
            reply = await self._handle.request(
                "scanLines", json, abort_signal=context.abort_signal
            )
            scan = reply.json
            return ok(
                LineScan(
                    newlines=scan["newlines"],
                    start=scan["start"],
                    end=scan["end"],
                    first_line_end=scan["firstLineEnd"],
                    last_line_start=scan["lastLineStart"],
                    selected_bytes=scan["selectedBytes"],
                    first_line_bytes=scan["firstLineBytes"],
                )
            )
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, self._path))

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        await self._handle.close()


class RemoteDirReader:
    """Pages of one directory's entries, in the order the daemon reports."""

    def __init__(self, handle: _Handle) -> None:
        self._handle = handle
        self._done = False
        self._closed = False

    async def next(self, max_entries: int, context: Context) -> Result:
        path = self._handle.path
        early = abort_result(context, path)
        if early is not None:
            return early
        if self._closed:
            return err(FileError("invalid", "Directory reader is closed", path))
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries <= 0:
            return err(FileError("invalid", "maxEntries must be a positive safe integer", path))
        entries: List[FileInfo] = []
        try:
            paths = await remote_path(self._handle.env.connection)
            # Like ``NodeDirReader``, count only entries that are part of the listing.
            while not self._done and len(entries) < max_entries:
                reply = await self._handle.request("readdir", {"max": max_entries - len(entries)})
                loop_abort = abort_result(context, path)
                if loop_abort is not None:
                    return loop_abort
                for entry in reply.json.get("entries") or []:
                    entry_path = paths.resolve_relative(path, entry["name"])
                    failure = entry.get("error")
                    if failure is not None:
                        # Removed between enumeration and lstat: not part of the listing any more.
                        if failure.get("code") == "ENOENT":
                            continue
                        return err(to_file_error(RemoteError(failure), entry_path))
                    info = _to_info(entry_path, entry["info"], paths)
                    if info.ok:
                        entries.append(info.value)
                self._done = reply.json.get("done") is True
            return ok({"entries": entries, "done": self._done})
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, path))

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        await self._handle.close()


class RemoteTextLineReader:
    """``NodeTextLineReader``: positional reads of a file opened like ``open(path, "r")``."""

    def __init__(self, handle: _Handle) -> None:
        self._handle = handle
        self._decoder = StreamDecoder()
        self._offset = 0
        self._buffered = ""
        self._ended = False
        self._closed = False

    async def read_line(self, context: Context) -> Result:
        path = self._handle.path
        early = abort_result(context, path)
        if early is not None:
            return early
        if self._closed:
            return err(FileError("invalid", "Text line reader is closed", path))
        try:
            while True:
                newline = self._buffered.find("\n")
                if newline != -1:
                    text = self._buffered[:newline]
                    self._buffered = self._buffered[newline + 1 :]
                    return ok(TextLine(text=text, terminated=True))
                if self._ended:
                    if len(self._buffered) == 0:
                        return ok(None)
                    text = self._buffered
                    self._buffered = ""
                    return ok(TextLine(text=text, terminated=False))
                data = await self._handle.pread(self._offset, LINE_CHUNK)
                after = abort_result(context, path)
                if after is not None:
                    return after
                self._offset += len(data)
                if len(data) == 0:
                    self._buffered += self._decoder.decode()
                    self._ended = True
                else:
                    self._buffered += self._decoder.decode(data)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, path))

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffered = ""
        await self._handle.close()


class RemoteExecutionEnv:
    """
    An ``ExecutionEnv`` on another machine, reached through a pi-env daemon. Results match
    ``NodeExecutionEnv`` running on that machine: the daemon performs the system calls, and this
    class applies Node's path, error, and result rules.
    """

    def __init__(
        self,
        connection: Connection,
        id: str,
        cwd: str,
        shell_path: Optional[str] = None,
        shell_env: Optional[Dict[str, str]] = None,
        watch: Optional[RemoteWatchOptions] = None,
    ) -> None:
        self.id = id
        self.cwd = cwd
        self.connection = connection
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._watch_options = watch or RemoteWatchOptions()
        # Running commands this environment started, for ``cleanup()``.
        self._running: Dict[int, int] = {}
        self._cleanup_generation = 0

    # -- paths -----------------------------------------------------------------

    async def _resolve(self, path: str) -> str:
        """Node's ``resolvePath`` on the remote system, with its home directory and working directories."""
        info = await self.connection.info()
        windows = info.os == "windows"
        paths = RemotePaths(windows)
        normalized = path
        if normalized == "~":
            normalized = info.home
        elif normalized.startswith("~/") or (windows and normalized.startswith("~\\")):
            normalized = paths.join(info.home, normalized[2:])
        elif normalized.startswith("file://"):
            converted = _file_url_to_path(normalized, windows)
            if converted is not None:
                normalized = converted
        if paths.is_absolute(normalized):
            return paths.resolve_absolute(normalized)
        if windows:
            # A drive-relative path on another drive: Node on Windows resolves it against that
            # drive's working directory (``=D:``), else its own working directory if on that drive,
            # else the drive's root.
            match = re.match(r"^([a-zA-Z]:)(?![\\/])", normalized)
            if match is not None:
                drive = match.group(1)
                if drive.lower() != self.cwd[:2].lower():
                    base = info.drive_cwds.get(drive.upper(), info.cwd)
                    if base[:2].lower() != drive.lower() and base[2:3] == "\\":
                        base = drive + "\\"
                    return paths.resolve_relative(base, normalized)
        return paths.resolve_relative(self.cwd, normalized)

    async def _file_op(
        self,
        path: str,
        context: Context,
        run: Callable[[str], Awaitable[Any]],
        check: str = "before",
    ) -> Result:
        try:
            resolved = await self._resolve(path)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, path))
        aborted = abort_result(context, resolved)
        if aborted is not None:
            return aborted
        try:
            value = await run(resolved)
            if check == "both":
                after = abort_result(context, resolved)
                if after is not None:
                    return after
            return ok(value)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, resolved))

    async def _open(self, resolved: str, json: Json) -> Tuple[_Handle, Json]:
        reply = await self.connection.request("open", {"path": resolved, **json})
        return _Handle(self, reply.json["handle"], reply.session, resolved), reply.json

    async def _open_dir(self, resolved: str) -> _Handle:
        reply = await self.connection.request("opendir", {"path": resolved})
        return _Handle(self, reply.json["handle"], reply.session, resolved)

    # -- filesystem ------------------------------------------------------------

    async def absolute_path(self, path: str, context: Context) -> Result:
        try:
            return ok(await self._resolve(path))
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, path))

    async def join_path(self, parts: Sequence[str], context: Context) -> Result:
        try:
            return ok((await remote_path(self.connection)).join(*parts))
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error))

    async def _read_file(self, path: str, encoding: bool, context: Context) -> Result:
        signal = context.abort_signal

        def aborted() -> bool:
            return signal is not None and signal.is_set()

        async def run(resolved: str) -> Any:
            handle, json = await self._open(resolved, {"mode": "read"})
            abort = lambda: FileError("aborted", "The operation was aborted", resolved)  # noqa: E731
            try:
                if json.get("statError") is not None:
                    raise RemoteError(json["statError"])
                if aborted():
                    raise abort()
                stat = json["stat"]
                size = stat.get("size", 0) if stat.get("kind") == "file" else 0
                if size > READ_FILE_MAX:
                    raise FileError("unknown", f"File size ({size}) is greater than 2 GiB", resolved)
                if size == 0:
                    # Unknown size: sequential reads until the end, as for FIFOs and devices.
                    decoder = range_decoder()
                    chunks: List[bytes] = []
                    total = 0
                    text = ""
                    while True:
                        if aborted():
                            raise abort()
                        data = await handle.pread(None, READ_FILE_UNKNOWN_CHUNK)
                        if len(data) == 0:
                            break
                        total += len(data)
                        chunks.append(data)
                        if encoding:
                            text += decoder.decode(data)
                    if not encoding:
                        return _concat(chunks)
                    return "" if total == 0 else text + decoder.decode()
                if not encoding:
                    # A single read of the whole size, or reads of 512 KiB until the size or the end of the file.
                    if size <= READ_FILE_CHUNK:
                        return await handle.pread(0, size)
                    read = await read_range(
                        lambda at, length: handle.pread(at, length), 0, size, READ_FILE_CHUNK, aborted
                    )
                    if read is None:
                        raise abort()
                    return _concat(read[0])
                # Decode exactly the size observed at open. ``read_range`` bounds the final read,
                # so a file that grows concurrently cannot extend this operation indefinitely.
                decoder = range_decoder()
                read = await read_range(
                    lambda at, length: handle.pread(at, length),
                    0,
                    size,
                    READ_FILE_CHUNK,
                    aborted,
                )
                if read is None:
                    raise abort()
                return "".join(decoder.decode(chunk) for chunk in read[0]) + decoder.decode()
            finally:
                await handle.close()

        return await self._file_op(path, context, run)

    async def read_text_file(self, path: str, context: Context) -> Result:
        return await self._read_file(path, True, context)

    async def read_binary_file(self, path: str, context: Context) -> Result:
        return await self._read_file(path, False, context)

    async def open_binary_reader(
        self, path: str, options: Optional[Dict[str, Any]], context: Context
    ) -> Result:
        no_follow = options is not None and options.get("no_follow") is True
        opened = await self._file_op(
            path, context, lambda resolved: self._open_handle(resolved, {"noFollow": no_follow})
        )
        if not opened.ok:
            return opened
        aborted = abort_result(context, opened.value.path)
        if aborted is not None:
            await opened.value.close()
            return aborted
        return ok(RemoteBinaryReader(opened.value))

    async def _open_handle(self, resolved: str, json: Json) -> _Handle:
        handle, _json = await self._open(resolved, json)
        return handle

    async def open_text_line_reader(self, path: str, context: Context) -> Result:
        opened = await self._file_op(path, context, lambda resolved: self._open_handle(resolved, {"mode": "read"}))
        if not opened.ok:
            return opened
        aborted = abort_result(context, opened.value.path)
        if aborted is not None:
            await opened.value.close()
            return aborted
        return ok(RemoteTextLineReader(opened.value))

    async def read_text_lines(self, path: str, options: Optional[Dict[str, int]], context: Context) -> Result:
        max_lines = None if options is None else options.get("max_lines")
        if max_lines is not None and max_lines <= 0:
            return ok([])
        opened = await self.open_text_line_reader(path, context)
        if not opened.ok:
            return opened
        reader: RemoteTextLineReader = opened.value
        lines: List[str] = []
        try:
            while max_lines is None or len(lines) < max_lines:
                line = await reader.read_line(context)
                if not line.ok:
                    return line
                if line.value is None:
                    break
                lines.append(line.value.text)
            return ok(lines)
        finally:
            await reader.close(context)

    async def _write(self, path: str, content: Any, append: bool, context: Context) -> Result:
        if isinstance(content, str):
            data = _well_formed(content).encode("utf-8")
        else:
            data = bytes(content)
        signal = None if append else context.abort_signal

        async def run(resolved: str) -> None:
            first = data[:WRITE_CHUNK]
            keep = len(data) > WRITE_CHUNK
            reply = await self.connection.request(
                "write", {"path": resolved, "append": append, "keep": keep}, payload=first
            )
            if not keep:
                return
            handle = _Handle(self, reply.json["handle"], reply.session, resolved)
            in_flight: List["asyncio.Task"] = []
            try:
                for offset in range(WRITE_CHUNK, len(data), WRITE_CHUNK):
                    if signal is not None and signal.is_set():
                        # A chunk that failed before this check fails the write, as it would have in sequence.
                        results = await asyncio.gather(*in_flight, return_exceptions=True)
                        for result in results:
                            if isinstance(result, BaseException):
                                raise result
                        raise FileError("aborted", "The operation was aborted", resolved)
                    written = asyncio.ensure_future(
                        handle.request("writeChunk", {}, payload=data[offset : offset + WRITE_CHUNK])
                    )
                    in_flight.append(written)
                    if len(in_flight) >= WRITE_DEPTH:
                        await in_flight.pop(0)
                # The first failure; later chunks are refused by the daemon, so the file has no gaps.
                for written in in_flight:
                    await written
            finally:
                await asyncio.gather(*in_flight, return_exceptions=True)
                await handle.close()

        return await self._file_op(path, context, run, "both" if append else "before")

    async def write_file(self, path: str, content: Any, context: Context) -> Result:
        return await self._write(path, content, False, context)

    async def append_file(self, path: str, content: Any, context: Context) -> Result:
        return await self._write(path, content, True, context)

    async def truncate_file(self, path: str, size: int, context: Context) -> Result:
        async def run(resolved: str) -> None:
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise FileError("invalid", "File size must be a non-negative safe integer", resolved)
            await self.connection.request("truncate", {"path": resolved, "size": size})

        return await self._file_op(path, context, run, "both")

    async def flush_file(self, path: str, context: Context) -> Result:
        async def run(resolved: str) -> None:
            await self.connection.request("fsync", {"path": resolved})

        return await self._file_op(path, context, run, "both")

    async def rename_file(self, source_path: str, destination_path: str, context: Context) -> Result:
        try:
            source = await self._resolve(source_path)
            destination = await self._resolve(destination_path)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, source_path))
        aborted = abort_result(context, destination)
        if aborted is not None:
            return aborted
        try:
            await self.connection.request("rename", {"path": source, "to": destination})
            return ok(None)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, source))

    async def file_info(self, path: str, context: Context) -> Result:
        async def run(resolved: str) -> Result:
            reply = await self.connection.request("lstat", {"path": resolved})
            paths = await remote_path(self.connection)
            return _to_info(resolved, reply.json, paths)

        result = await self._file_op(path, context, run)
        return result.value if result.ok else result

    async def list_dir(self, path: str, context: Context) -> Result:
        async def run(resolved: str) -> List[FileInfo]:
            handle = await self._open_dir(resolved)
            entries: List[Json] = []
            try:
                done = False
                while not done:
                    reply = await handle.request("readdir", {"max": 1000})
                    entries.extend(reply.json.get("entries") or [])
                    done = reply.json.get("done") is True
            finally:
                await handle.close()
            # Node's ``readdir`` fails on any entry it cannot lstat. libuv sorts names by their bytes
            # on POSIX; on Windows it keeps the file system's order, which the daemon reports.
            paths = await remote_path(self.connection)
            if not paths.windows:
                def key(entry: Json) -> bytes:
                    raw = entry.get("raw")
                    if isinstance(raw, list):
                        return bytes(raw)
                    return entry["name"].encode("utf-8", "surrogatepass")

                entries.sort(key=key)
            infos: List[FileInfo] = []
            for entry in entries:
                if context.abort_signal is not None and context.abort_signal.is_set():
                    raise FileError("aborted", "aborted", resolved)
                entry_path = paths.resolve_relative(resolved, entry["name"])
                failure = entry.get("error")
                if failure is not None:
                    raise to_file_error(RemoteError(failure), entry_path)
                info = _to_info(entry_path, entry["info"], paths)
                if info.ok:
                    infos.append(info.value)
            return infos

        return await self._file_op(path, context, run)

    async def open_dir_reader(self, path: str, context: Context) -> Result:
        opened = await self._file_op(path, context, self._open_dir)
        if not opened.ok:
            return opened
        aborted = abort_result(context, opened.value.path)
        if aborted is not None:
            await opened.value.close()
            return aborted
        return ok(RemoteDirReader(opened.value))

    async def canonical_path(self, path: str, context: Context) -> Result:
        async def run(resolved: str) -> str:
            reply = await self.connection.request("realpath", {"path": resolved})
            return reply.json["path"]

        return await self._file_op(path, context, run)

    async def exists(self, path: str, context: Context) -> Result:
        result = await self.file_info(path, context)
        if result.ok:
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    async def create_dir(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:
        async def run(resolved: str) -> None:
            recursive = True if options is None else bool(options.get("recursive", True))
            await self.connection.request("mkdir", {"path": resolved, "recursive": recursive})

        return await self._file_op(path, context, run)

    async def remove(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:
        async def run(resolved: str) -> None:
            recursive = False if options is None else bool(options.get("recursive", False))
            force = False if options is None else bool(options.get("force", False))
            await self.connection.request(
                "rm", {"path": resolved, "recursive": recursive, "force": force}
            )

        return await self._file_op(path, context, run)

    async def create_temp_dir(self, prefix: Optional[str], context: Context) -> Result:
        early = abort_result(context)
        if early is not None:
            return early
        try:
            info = await self.connection.info()
            paths = await remote_path(self.connection)
            reply = await self.connection.request("mkdtemp", {"path": paths.join(info.tmpdir, prefix or "tmp-")})
            return ok(reply.json["path"])
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error))

    async def create_temp_file(self, options: Optional[Dict[str, str]], context: Context) -> Result:
        directory = await self.create_temp_dir("tmp-", context)
        if not directory.ok:
            return directory
        prefix = (options or {}).get("prefix", "")
        suffix = (options or {}).get("suffix", "")
        file_path = ""
        try:
            paths = await remote_path(self.connection)
            file_path = paths.join(directory.value, f"{prefix}{uuid_module.uuid4()}{suffix}")
            # Node's ``writeFile(filePath, "")``, which does not create parents.
            await self.connection.request(
                "write", {"path": file_path, "append": False, "parents": False}, payload=b""
            )
            return ok(file_path)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error, file_path))

    async def watch(
        self, targets: Sequence[WatchTarget], on_change: Callable[[WatchChange], None], context: Context
    ) -> Result:
        early = abort_result(context)
        if early is not None:
            return early
        try:
            resolved = [
                WatchTarget(path=await self._resolve(target.path), recursive=target.recursive, exclude=target.exclude)
                for target in targets
            ]
            after_resolve = abort_result(context)
            if after_resolve is not None:
                return after_resolve
            watcher = await RemoteWatcher.open(self.connection, resolved, on_change, self._watch_options)
            after_open = abort_result(context)
            if after_open is not None:
                await watcher.close(context)
                return after_open
            return ok(watcher)
        except Exception as error:  # noqa: BLE001
            return err(to_file_error(error))

    # -- shell -----------------------------------------------------------------

    async def exec(
        self, command: Any, options: Optional[ShellExecOptions], context: Context
    ) -> Result:
        cleanup_generation = self._cleanup_generation
        signal = context.abort_signal
        if signal is not None and signal.is_set():
            return err(ExecutionError("aborted", "aborted"))
        timeout = options.timeout if options is not None else None
        if timeout is not None:
            if (
                not isinstance(timeout, (int, float))
                or isinstance(timeout, bool)
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                return err(ExecutionError("timeout", "Invalid timeout: must be a finite number of seconds"))
            if timeout * 1000 > MAX_TIMEOUT_MS:
                return err(ExecutionError("timeout", f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"))
        inherit_env = True if options is None or options.inherit_env is None else options.inherit_env
        try:
            cwd = await self._resolve(options.cwd) if options is not None and options.cwd else self.cwd
        except Exception as error:  # noqa: BLE001
            return err(ExecutionError("unknown", str(error)))
        if signal is not None and signal.is_set():
            return err(ExecutionError("aborted", "aborted"))
        env: Dict[str, str] = {}
        extra_env = (options.env if options is not None else None) or {}
        source: Dict[str, str] = (
            {**(self._shell_env or {}), **extra_env} if inherit_env else dict(extra_env)
        )
        for key, value in source.items():
            if value is not None:
                env[_well_formed(key)] = value

        callback_error: Optional[ExecutionError] = None
        settled = False
        running: Optional[int] = None
        local_abort = asyncio.Event()

        async def _relay_abort() -> None:
            if signal is not None:
                await signal.wait()
                local_abort.set()

        relay_task = asyncio.ensure_future(_relay_abort())

        def on_event(event: Json, payload: bytes) -> None:
            nonlocal callback_error
            if settled or callback_error is not None or event.get("kind") != "output":
                return
            text = payload.decode("utf-8", "replace")
            if text == "" or options is None or options.on_output is None:
                return
            info = ShellOutputInfo(stream="stderr" if event.get("stream") == "stderr" else "stdout")
            skipped = event.get("skipped")
            if isinstance(skipped, dict):
                info.skipped = ShellOutputSkip(
                    bytes=skipped.get("bytes", 0),
                    newlines=skipped.get("newlines", 0),
                    ends_with_newline=bool(skipped.get("endsWithNewline")),
                )
            try:
                options.on_output(text, context, info)
            except Exception as error:  # noqa: BLE001 - a failing output callback fails the command
                cause = error if isinstance(error, Exception) else RuntimeError(str(error))
                callback_error = ExecutionError("callback_error", str(cause), cause)
                local_abort.set()

        def on_start(request_id: int, session_id: int) -> None:
            nonlocal running
            running = request_id
            self._running[request_id] = session_id
            if cleanup_generation != self._cleanup_generation:
                self.connection.kill(request_id, session_id)

        try:
            reply = await self.connection.request(
                "exec",
                {
                    **({"command": command} if isinstance(command, str) else {"argv": list(command)}),
                    "cwd": cwd,
                    "env": env,
                    "inheritEnv": inherit_env,
                    **({} if self._shell_path is None else {"shellPath": self._shell_path}),
                    **({} if timeout is None else {"timeoutMs": timeout * 1000}),
                    **(
                        {}
                        if options is None or options.spill is None
                        else {"spill": {"afterBytes": options.spill.after_bytes, "afterLines": options.spill.after_lines}}
                    ),
                    **(
                        {}
                        if options is None or options.window is None
                        else {
                            "window": {
                                "maxBytes": options.window.max_bytes,
                                "maxLines": options.window.max_lines,
                                "minIntervalMs": options.window.min_interval_ms,
                                "bytesPerSecond": options.window.bytes_per_second,
                            }
                        }
                    ),
                },
                abort_signal=local_abort,
                on_event=on_event,
                on_start=on_start,
            )
            if callback_error is not None:
                return err(callback_error)
            spill_path = reply.json.get("spillPath")
            return ok(
                ShellExecResult(
                    exit_code=reply.json.get("exitCode", 0),
                    spill_path=spill_path if isinstance(spill_path, str) else None,
                )
            )
        except RemoteError as error:
            if callback_error is not None:
                return err(callback_error)
            if error.code == "timeout":
                failure = ExecutionError("timeout", f"timeout:{timeout}")
            elif error.code == "aborted":
                failure = ExecutionError("aborted", "aborted")
            elif error.code in ("shell_unavailable", "spawn_error"):
                failure = ExecutionError(error.code, str(error))
            else:
                failure = ExecutionError("unknown", str(error))
            spill_path = error.fields.get("spillPath")
            if isinstance(spill_path, str):
                failure.spill_path = spill_path
            return err(failure)
        except Exception as error:  # noqa: BLE001
            if callback_error is not None:
                return err(callback_error)
            return err(ExecutionError("unknown", str(error)))
        finally:
            settled = True
            relay_task.cancel()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
            if running is not None:
                self._running.pop(running, None)

    async def cleanup(self, context: Context) -> None:
        # Kill without aborting, so the commands settle with their killed status, as ``NodeExecutionEnv`` does.
        self._cleanup_generation += 1
        for request_id, session_id in list(self._running.items()):
            self.connection.kill(request_id, session_id)
        self._running.clear()
