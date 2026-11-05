"""Portable execution environment: filesystem, shell, and watch contracts.

Python port of ``packages/durable/src/env`` (runtime parts of index.ts plus
decode.ts). Operations return ``Result`` values instead of raising expected
failures, so tools compose them without try/except chains.

Porting deviations from the TS env:
- ``Context`` is a plain dataclass carrying an optional ``asyncio.Event`` abort
  signal instead of the chord context object.
- Interfaces are ``typing.Protocol`` classes; the local implementation is
  ``local.LocalExecutionEnv`` (asyncio + os) instead of Node fs promises.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Union

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """Result of a fallible operation. Expected failures are returned, not raised."""

    ok: bool
    value: Any = None
    error: Any = None


def ok(value: Any) -> Result:
    return Result(ok=True, value=value)


def err(error: Any) -> Result:
    return Result(ok=False, error=error)


def get_or_throw(result: Result) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def get_or_undefined(result: Result) -> Any:
    return result.value if result.ok else None


def to_error(error: Any) -> Exception:
    if isinstance(error, Exception):
        return error
    return RuntimeError(str(error))


# ---------------------------------------------------------------------------
# Context and abort
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Cancellation scope: an ``asyncio.Event`` set means "stop working"."""

    abort_signal: Optional[asyncio.Event] = None

    @property
    def aborted(self) -> bool:
        return self.abort_signal is not None and self.abort_signal.is_set()


BACKGROUND_CONTEXT = Context()


def with_abort_signal(signal: asyncio.Event, context: Context = BACKGROUND_CONTEXT) -> Context:
    return Context(abort_signal=signal)


def aborted_context() -> Context:
    signal = asyncio.Event()
    signal.set()
    return Context(abort_signal=signal)


# ---------------------------------------------------------------------------
# Filesystem contracts
# ---------------------------------------------------------------------------

FileKind = str  # "file" | "directory" | "symlink"


class FileError(Exception):
    """A filesystem operation failure with a stable code."""

    def __init__(
        self, code: str, message: str, path: Optional[str] = None, cause: Optional[BaseException] = None
    ) -> None:
        super().__init__(message)
        self.name = "FileError"
        self.code = code
        self.path = path
        self.__cause__ = cause


class ExecutionError(Exception):
    """A shell execution failure with a stable code."""

    def __init__(self, code: str, message: str, cause: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.name = "ExecutionError"
        self.code = code
        self.spill_path: Optional[str] = None
        self.__cause__ = cause


@dataclass
class FileInfo:
    name: str
    path: str
    kind: FileKind
    size: int
    mtime_ms: float


@dataclass
class TextLine:
    text: str
    terminated: bool


class TextLineReader(Protocol):
    async def read_line(self, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def close(self, context: Context) -> None:  # pragma: no cover - contract
        ...


class BinaryReader(Protocol):
    """Positional reads from one opened regular file; all calls see the same file."""

    async def info(self, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def read(self, offset: int, length: int, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def scan_lines(self, options: Dict[str, int], context: Context) -> Result:  # pragma: no cover - contract
        """Locate lines ``[start_line, end_line)`` (missing ``end_line``: to the end), 0-based."""

    async def close(self, context: Context) -> None:  # pragma: no cover - contract
        ...


@dataclass
class LineScan:
    """Where lines of a file are, as ``BinaryReader.scan_lines`` found them."""

    newlines: int
    start: int
    end: int
    first_line_end: int
    last_line_start: int
    selected_bytes: int
    first_line_bytes: int


@dataclass
class WatchTarget:
    """A file or directory to watch. It may be missing; creating it is a change."""

    path: str
    recursive: Optional[bool] = None
    exclude: Optional[Dict[str, Any]] = None  # {"hidden": bool, "names": [str]}


WatchChange = Dict[str, Any]  # {"paths": [str]} | {"overflow": True} | {"error": FileError}


class FileWatcher(Protocol):
    mode: str  # "native" | "polling"

    async def close(self, context: Context) -> None:  # pragma: no cover - contract
        ...


class DirReader(Protocol):
    async def next(self, max_entries: int, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def close(self, context: Context) -> None:  # pragma: no cover - contract
        ...


class FileSystem(Protocol):
    """Portable filesystem capability. Operations return failures rather than raising."""

    id: str
    cwd: str

    async def absolute_path(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def join_path(self, parts: Sequence[str], context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def read_text_file(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def open_text_line_reader(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def read_text_lines(
        self, path: str, options: Optional[Dict[str, int]], context: Context
    ) -> Result:  # pragma: no cover - contract
        ...

    async def read_binary_file(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def open_binary_reader(
        self, path: str, options: Optional[Dict[str, Any]], context: Context
    ) -> Result:  # pragma: no cover - contract
        ...

    async def write_file(self, path: str, content: Union[str, bytes], context: Context) -> Result:  # pragma: no cover
        ...

    async def append_file(self, path: str, content: Union[str, bytes], context: Context) -> Result:  # pragma: no cover
        ...

    async def truncate_file(self, path: str, size: int, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def flush_file(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def rename_file(self, source_path: str, destination_path: str, context: Context) -> Result:  # pragma: no cover
        ...

    async def file_info(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def list_dir(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def open_dir_reader(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def watch(
        self, targets: Sequence[WatchTarget], on_change: Callable[[WatchChange], None], context: Context
    ) -> Result:  # pragma: no cover - contract
        ...

    async def canonical_path(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def exists(self, path: str, context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def create_dir(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:  # pragma: no cover
        ...

    async def remove(self, path: str, options: Optional[Dict[str, Any]], context: Context) -> Result:  # pragma: no cover
        ...

    async def create_temp_dir(self, prefix: Optional[str], context: Context) -> Result:  # pragma: no cover - contract
        ...

    async def create_temp_file(self, options: Optional[Dict[str, str]], context: Context) -> Result:  # pragma: no cover
        ...

    async def cleanup(self, context: Context) -> None:  # pragma: no cover - contract
        ...


# ---------------------------------------------------------------------------
# Shell contracts
# ---------------------------------------------------------------------------


@dataclass
class ShellSpillOptions:
    """Spill the complete output to a temporary file once it exceeds either threshold."""

    after_bytes: int
    after_lines: int  # complete or partial lines


@dataclass
class ShellExecResult:
    exit_code: int
    spill_path: Optional[str] = None


@dataclass
class ShellOutputWindow:
    """The tail of the combined output a caller keeps, and how often it samples it."""

    max_bytes: int
    max_lines: int
    min_interval_ms: int
    bytes_per_second: int


@dataclass
class ShellOutputSkip:
    """Output an environment omitted, measured on the decoded text ``on_output`` would have received."""

    bytes: int
    newlines: int
    ends_with_newline: bool


@dataclass
class ShellOutputInfo:
    stream: str  # "stdout" | "stderr"
    skipped: Optional[ShellOutputSkip] = None


@dataclass
class ShellExecOptions:
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    inherit_env: Optional[bool] = None
    timeout: Optional[float] = None
    on_output: Optional[Callable[[str, Context, ShellOutputInfo], None]] = None
    spill: Optional[ShellSpillOptions] = None
    window: Optional[ShellOutputWindow] = None


class Shell(Protocol):
    async def exec(
        self, command: Union[str, Sequence[str]], options: Optional[ShellExecOptions], context: Context
    ) -> Result:  # pragma: no cover - contract
        ...

    async def cleanup(self, context: Context) -> None:  # pragma: no cover - contract
        ...


class ExecutionEnv(FileSystem, Shell):
    """A filesystem and a shell of one environment (host, container, or remote)."""


# ---------------------------------------------------------------------------
# Decoding (port of decode.ts)
# ---------------------------------------------------------------------------


def range_decoder() -> "_RangeDecoder":
    """A streaming decoder for a byte range; BOM handling is off."""
    return _RangeDecoder()


class _RangeDecoder:
    """Streaming UTF-8 decode with WHATWG replacement semantics; BOM handling is off."""

    def __init__(self) -> None:
        self._pending = b""

    def decode(self, data: Optional[bytes] = None) -> str:
        if data is None:
            text, self._pending = self._pending.decode("utf-8", "replace"), b""
            return text
        buffered = self._pending + data
        consumed = _safe_prefix_length(buffered)
        text = buffered[:consumed].decode("utf-8", "replace")
        self._pending = buffered[consumed:]
        return text


def _safe_prefix_length(buffered: bytes) -> int:
    """Length of the prefix that decodes without seeing future bytes.

    An incomplete trailing sequence is held back; everything else decodes now,
    with invalid sequences replaced like the WHATWG decoder does.
    """
    n = len(buffered)
    continuation = 0
    while continuation < n and (buffered[n - 1 - continuation] & 0xC0) == 0x80:
        continuation += 1
    if continuation == 0:
        return n
    if continuation == n:
        # Only continuation bytes: no lead byte to complete; they are all invalid.
        return n
    lead = buffered[n - 1 - continuation]
    if 0xC2 <= lead <= 0xDF:
        needed = 2
    elif 0xE0 <= lead <= 0xEF:
        needed = 3
    elif 0xF0 <= lead <= 0xF4:
        needed = 4
    else:
        return n  # invalid lead byte: replaced immediately, nothing held back
    if continuation < needed:
        return n - 1 - continuation  # hold the incomplete sequence
    return n


def starts_with_bom(first_bytes: bytes) -> bool:
    """Whether decoding the whole input drops its first three bytes as a byte-order mark."""
    return len(first_bytes) >= 3 and first_bytes[0] == 0xEF and first_bytes[1] == 0xBB and first_bytes[2] == 0xBF


class StreamDecoder:
    """Decodes one stream chunk by chunk exactly like decoding all of it at once."""

    def __init__(self) -> None:
        self._decoder = range_decoder()
        self._started = False

    def decode(self, data: Optional[bytes] = None) -> str:
        text = self._decoder.decode(data)
        if self._started or text == "":
            return text
        self._started = True
        # U+FEFF encodes only as EF BB BF, so a leading U+FEFF is a byte-order mark at the stream's start.
        return text[1:] if text.startswith("\ufeff") else text


from .line_scan import LineScanner  # noqa: E402  (circular-free re-export at the bottom)
from .local import LocalExecutionEnv  # noqa: E402

__all__ = [
    "BACKGROUND_CONTEXT",
    "BinaryReader",
    "Context",
    "DirReader",
    "ExecutionEnv",
    "ExecutionError",
    "FileError",
    "FileInfo",
    "FileKind",
    "FileSystem",
    "FileWatcher",
    "LineScan",
    "LineScanner",
    "LocalExecutionEnv",
    "Result",
    "Shell",
    "ShellExecOptions",
    "ShellExecResult",
    "ShellOutputInfo",
    "ShellOutputSkip",
    "ShellOutputWindow",
    "ShellSpillOptions",
    "StreamDecoder",
    "TextLine",
    "TextLineReader",
    "WatchChange",
    "WatchTarget",
    "aborted_context",
    "err",
    "get_or_throw",
    "get_or_undefined",
    "ok",
    "range_decoder",
    "starts_with_bom",
    "to_error",
    "with_abort_signal",
]
