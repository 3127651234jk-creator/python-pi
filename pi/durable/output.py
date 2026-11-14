"""Tool output bounding and adaptive progress commits (port of ``harness/output.ts``).

Retention limits of one tool's output, exact slicing within the limits, a
bounded running output buffer, and the ``Progress`` throttle that commits
partial answers and running tool output.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Union

from .env import ShellOutputSkip, range_decoder
from .truncate import utf8_byte_length

NEWLINE = 0x0A
INVALID_OUTPUT = re.compile(r"[\x00-\x08\x0b-\x1f\ufff9-\ufffb]")


@dataclass
class OutputLimits:
    max_bytes: int
    max_lines: int
    retain: str  # "head" | "tail"


@dataclass
class OutputSlice:
    """An exact slice of the input within the limits, and what it left out."""

    text: str
    bytes: int
    dropped_bytes: int
    dropped_lines: int


@dataclass
class BoundedOutput:
    text: str
    dropped_bytes: int
    dropped_lines: int


def sanitize_output(text: str) -> str:
    """Remove control characters that break display and transcripts; tabs and newlines stay."""
    return INVALID_OUTPUT.sub("", text)


def character_end(data: bytes, index: int) -> int:
    """The last character boundary at or before ``index``."""
    end = index
    while end > 0 and ((data[end] if end < len(data) else 0) & 0xC0) == 0x80:
        end -= 1
    return end


def character_start(data: bytes, index: int) -> int:
    """The first character boundary at or after ``index``."""
    start = index
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return start


def _line_count(data: bytes) -> int:
    if len(data) == 0:
        return 0
    newlines = data.count(b"\n")
    return newlines + (0 if data[-1] == NEWLINE else 1)


def bound_output(text: str, limits: OutputLimits) -> OutputSlice:
    """Bound ``text`` to whole lines within the limits: first lines for head, last lines for tail."""
    data = text.encode("utf-8")
    start, end = _head_range(data, limits) if limits.retain == "head" else _tail_range(data, limits)
    kept = data[start:end]
    return OutputSlice(
        text=text if len(kept) == len(data) else kept.decode("utf-8", "replace"),
        bytes=len(kept),
        dropped_bytes=len(data) - len(kept),
        dropped_lines=_line_count(data) - _line_count(kept),
    )


def _head_range(data: bytes, limits: OutputLimits) -> "tuple[int, int]":
    if limits.max_lines == 0 or limits.max_bytes == 0:
        return 0, 0
    end = len(data)
    lines = 0
    index = data.find(b"\n")
    while index != -1:
        lines += 1
        if lines == limits.max_lines:
            end = index + 1
            break
        index = data.find(b"\n", index + 1)
    if end > limits.max_bytes:
        newline = data.rfind(b"\n", 0, limits.max_bytes)
        end = newline + 1 if newline != -1 else character_end(data, limits.max_bytes)
    return 0, end


def _tail_range(data: bytes, limits: OutputLimits) -> "tuple[int, int]":
    if limits.max_lines == 0 or limits.max_bytes == 0:
        return len(data), len(data)
    # A trailing newline ends the last line rather than starting another.
    last = len(data) - 2 if len(data) > 0 and data[-1] == NEWLINE else len(data) - 1
    start = 0
    lines = 1
    index = data.rfind(b"\n", 0, last + 1) if last >= 0 else -1
    while index != -1:
        if lines == limits.max_lines:
            start = index + 1
            break
        lines += 1
        index = -1 if index == 0 else data.rfind(b"\n", 0, index)
    if len(data) - start > limits.max_bytes:
        window_start = len(data) - limits.max_bytes
        newline = data.find(b"\n", max(0, window_start - 1))
        # The first line starting inside the byte window, or a cut of the last line when it alone is too long.
        if newline != -1 and newline + 1 < len(data):
            start = newline + 1
        else:
            start = character_start(data, window_start)
    return start, len(data)


def _lines(newlines: int, terminated: bool) -> int:
    return newlines + (0 if terminated else 1)


def _count_newlines(text: str) -> int:
    return text.count("\n")


def _tail_margin(text: str, limits: OutputLimits) -> str:
    """The shortest suffix with more than ``max_bytes`` bytes or more than ``max_lines`` newlines, or all of it."""
    data = text.encode("utf-8")
    byte_start = character_end(data, len(data) - limits.max_bytes - 1) if len(data) > limits.max_bytes else 0
    line_start = 0
    newlines = 0
    index = data.rfind(b"\n")
    while index != -1:
        newlines += 1
        if newlines > limits.max_lines:
            line_start = index
            break
        if index == 0:
            break
        index = data.rfind(b"\n", 0, index)
    start = max(byte_start, line_start)
    return text if start == 0 else data[start:].decode("utf-8", "replace")


class OutputBuffer:
    """Bounded running output of one tool call.

    Accepting a chunk costs time proportional to the chunk: head retention stops
    storing once the window is full, and tail retention drops stored text the
    window no longer needs when it snapshots. Counts of the whole stream are kept
    so the dropped totals stay exact.
    """

    def __init__(self, limits: OutputLimits) -> None:
        self._limits = limits
        # Decoder of byte chunks: slices decode exactly, so only a byte-order mark at the very start of the
        # output is dropped, never a U+FEFF later in it.
        self._decoder = range_decoder()
        self._started = False
        self._chunks: List[dict] = []
        self._stored_bytes = 0
        self._stored_newlines = 0
        self._full = False
        self._total_bytes = 0
        self._total_newlines = 0
        self._ends_with_newline = True

    @property
    def stored_bytes(self) -> int:
        return self._stored_bytes

    def push(self, chunk: Union[str, bytes], skipped: Optional[ShellOutputSkip] = None) -> bool:
        """Accept a chunk; ``skipped`` is output omitted right before it (tail retention only).

        A string chunk or a skip ends an incomplete character of earlier bytes (it becomes U+FFFD); the
        next byte chunk then starts a new character."""
        # Bytes of an incomplete character from an earlier byte chunk come first.
        pending = self._decoder.decode() if isinstance(chunk, str) or skipped is not None else ""
        text = chunk if isinstance(chunk, str) else self._decoder.decode(chunk)
        first = not self._started and pending == "" and skipped is None
        if pending != "" or text != "" or skipped is not None:
            self._started = True
        if first and not isinstance(chunk, str) and text.startswith("\ufeff"):
            text = text[1:]
        if skipped is None:
            return self._accept(pending + text)
        if self._limits.retain != "tail":
            raise ValueError("Skipped output requires tail retention")
        self._accept(pending)
        self._skip(skipped)
        self._accept(text)
        return True

    def _skip(self, skipped: ShellOutputSkip) -> None:
        """Count omitted output; nothing stored before it can be in the window once new text arrives."""
        if skipped.bytes == 0:
            return
        self._total_bytes += skipped.bytes
        self._total_newlines += skipped.newlines
        self._ends_with_newline = skipped.ends_with_newline
        self._chunks = []
        self._stored_bytes = 0
        self._stored_newlines = 0

    def end(self) -> None:
        """Flush an incomplete trailing character as a replacement character; call when the stream ends."""
        self._accept(self._decoder.decode())

    def _accept(self, text: str) -> bool:
        if len(text) == 0:
            return False
        chunk_bytes = utf8_byte_length(text)
        chunk_newlines = _count_newlines(text)
        self._total_bytes += chunk_bytes
        self._total_newlines += chunk_newlines
        self._ends_with_newline = text.endswith("\n")
        if self._full:
            return True
        self._chunks.append({"text": text, "bytes": chunk_bytes, "newlines": chunk_newlines})
        self._stored_bytes += chunk_bytes
        self._stored_newlines += chunk_newlines
        if self._limits.retain == "head":
            # Nothing past a full window is ever needed.
            self._full = self._stored_bytes > self._limits.max_bytes or self._stored_newlines >= self._limits.max_lines
            return True
        # Drop leading chunks while the rest still holds more than a window, so the window's line start can
        # still be found later. Each chunk is dropped once.
        while len(self._chunks) > 1:
            first = self._chunks[0]
            bytes_after = self._stored_bytes - first["bytes"]
            newlines_after = self._stored_newlines - first["newlines"]
            if bytes_after <= self._limits.max_bytes + 1 and newlines_after <= self._limits.max_lines + 1:
                break
            self._chunks.pop(0)
            self._stored_bytes = bytes_after
            self._stored_newlines = newlines_after
        return True

    def snapshot(self) -> BoundedOutput:
        """Retained, sanitized output and what the limits dropped from the whole stream."""
        stored = "".join(chunk["text"] for chunk in self._chunks)
        return self._snapshot_stored(stored)

    def _snapshot_stored(self, stored: str) -> BoundedOutput:
        kept = bound_output(stored, self._limits)
        stored_lines = _lines(self._stored_newlines, stored == "" or stored.endswith("\n"))
        kept_lines = stored_lines - kept.dropped_lines
        # Tail windows never reach back before this one, but finding a later window's first line needs what
        # precedes it: keep the shortest suffix longer than the window by a byte or a line.
        if self._limits.retain == "tail" or len(self._chunks) > 1:
            text = _tail_margin(stored, self._limits) if self._limits.retain == "tail" else stored
            byte_count = utf8_byte_length(text) if self._limits.retain == "tail" else self._stored_bytes
            self._chunks = [{"text": text, "bytes": byte_count, "newlines": _count_newlines(text)}] if text else []
            self._stored_bytes = byte_count
            self._stored_newlines = self._chunks[0]["newlines"] if self._chunks else 0
        return BoundedOutput(
            text=sanitize_output(kept.text),
            dropped_bytes=self._total_bytes - kept.bytes,
            dropped_lines=_lines(self._total_newlines, self._ends_with_newline) - kept_lines,
        )


PROGRESS_BYTES_PER_SECOND = 100 * 1024


class Progress:
    """Adaptive progress commits (port of the TS ``Progress``).

    The first change after an idle period commits at once. Each commit then
    delays the next by at least ``min_interval_ms`` and by the bytes it wrote at
    100 KiB/s. At most one commit is in flight; changes made meanwhile coalesce
    into the next one. ``stop()`` cancels the timer, waits for the in-flight
    commit, and returns the waiters the final transaction must settle.
    """

    def __init__(
        self,
        write: Callable[[], Awaitable[int]],
        on_error: Callable[[BaseException], None],
        min_interval_ms: float,
        clock: Callable[[], float] = time.monotonic,
        schedule: Optional[Callable[[float, Callable[[], None]], Any]] = None,
    ) -> None:
        self._write = write
        self._on_error = on_error
        self._min_interval_ms = float(min_interval_ms)
        self._clock = clock
        self._schedule_fn = schedule or self._default_schedule
        self._waiters: List[asyncio.Future] = []
        self._timer: Any = None
        self._in_flight: Optional[asyncio.Task] = None
        self._next_at = 0.0
        self._dirty = False
        self._stopped = False

    @staticmethod
    def _default_schedule(delay: float, callback: Callable[[], None]) -> Any:
        return asyncio.get_running_loop().call_later(max(0.0, delay), callback)

    def mark(self) -> None:
        self._dirty = True
        self._schedule()

    def mark_and_wait(self) -> "asyncio.Future":
        """Schedule a commit; the future settles with the commit including this change."""

        future = asyncio.get_running_loop().create_future()
        self._waiters.append(future)
        self.mark()
        return future

    async def stop(self) -> List["asyncio.Future"]:
        """Stop committing, await the in-flight commit, and return pending waiters."""

        self._stopped = True
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:  # noqa: BLE001 - injected schedulers may not cancel
                pass
            self._timer = None
        if self._in_flight is not None:
            await asyncio.shield(self._in_flight)
        waiters, self._waiters = self._waiters, []
        return waiters

    def _schedule(self) -> None:
        if self._stopped or self._timer is not None or self._in_flight is not None:
            return
        wait = self._next_at - self._clock()
        if wait <= 0:
            self._flush()
        else:
            self._timer = self._schedule_fn(wait, self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        self._flush()

    def _flush(self) -> None:
        if self._stopped or not self._dirty:
            return
        self._dirty = False
        waiters, self._waiters = self._waiters, []
        started = self._clock()

        async def run() -> None:
            try:
                written = await self._write()
                written = int(written or 0)
            except BaseException as exc:  # noqa: BLE001 - delivered through waiters and on_error
                self._next_at = started + self._min_interval_ms / 1000.0
                for waiter in waiters:
                    if not waiter.done():
                        waiter.set_exception(exc)
                try:
                    self._on_error(exc)
                except Exception:  # noqa: BLE001 - reporting must not break the throttle
                    pass
                return
            delay_ms = max(self._min_interval_ms, written * 1000.0 / PROGRESS_BYTES_PER_SECOND)
            self._next_at = started + delay_ms / 1000.0
            for waiter in waiters:
                if not waiter.done():
                    waiter.set_result(None)

        task = asyncio.ensure_future(run())
        self._in_flight = task

        def finished(_task: asyncio.Task) -> None:
            self._in_flight = None
            if self._dirty and not self._stopped:
                self._schedule()

        task.add_done_callback(finished)
