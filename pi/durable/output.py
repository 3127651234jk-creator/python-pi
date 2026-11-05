"""Tool output bounding (port of ``harness/output.ts``, runtime parts).

Retention limits of one tool's output, exact slicing within the limits, and a
bounded running output buffer. Progress commits (``Progress``) are not ported;
the Python harness has no output-commit surface yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union

from .env import ShellOutputSkip
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
        self._decoder = _StreamUtf8()
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
        """Accept a chunk; ``skipped`` is output omitted right before it (tail retention only)."""
        pending = self._decoder.decode() if isinstance(chunk, str) or skipped is not None else ""
        text = chunk if isinstance(chunk, str) else self._decoder.decode(chunk)
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


class _StreamUtf8:
    """Incremental bytes-to-text decoder holding back incomplete sequences."""

    def __init__(self) -> None:
        self._pending = b""

    def decode(self, data: Optional[bytes] = None) -> str:
        if data is None:
            text, self._pending = self._pending.decode("utf-8", "replace"), b""
            return text
        buffered = self._pending + data
        consumed = _safe_length(buffered)
        text = buffered[:consumed].decode("utf-8", "replace")
        self._pending = buffered[consumed:]
        return text


def _safe_length(buffered: bytes) -> int:
    n = len(buffered)
    continuation = 0
    while continuation < n and (buffered[n - 1 - continuation] & 0xC0) == 0x80:
        continuation += 1
    if continuation == 0 or continuation == n:
        return n
    lead = buffered[n - 1 - continuation]
    needed = 2 if 0xC2 <= lead <= 0xDF else 3 if 0xE0 <= lead <= 0xEF else 4 if 0xF0 <= lead <= 0xF4 else 0
    if needed == 0 or continuation >= needed:
        return n
    return n - 1 - continuation
