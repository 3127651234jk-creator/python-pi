"""Bounded-memory line scanning (port of ``env/line-scan.ts``).

``LineScanner`` computes a ``LineScan`` from a file's bytes fed in order, so an
environment can scan any size in bounded memory. Decoded sizes use the same
streaming decoder as decoding the whole file at once.
"""

from __future__ import annotations

from typing import List, Optional

from . import LineScan, range_decoder, starts_with_bom

NEWLINE = 0x0A


def _decoded_bytes(text: str) -> int:
    return len(text.encode("utf-8")) if text else 0


class LineScanner:
    """`start_line` and `end_line` must be non-negative integers with `end_line > start_line`."""

    def __init__(self, start_line: int, end_line: Optional[int] = None) -> None:
        if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 0:
            raise ValueError("Invalid line range")
        if end_line is None:
            end_line = _INFINITE
        elif not (isinstance(end_line, int) and not isinstance(end_line, bool)):
            raise ValueError("Invalid line range")
        if not (end_line > start_line):
            raise ValueError("Invalid line range")
        self._start_line = start_line
        self._end_line = end_line
        self._position = 0
        self._newlines = 0
        self._line_start = 0
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._first_line_end: Optional[int] = None
        self._last_line_start: Optional[int] = None
        self._selected_bytes = 0
        self._first_line_bytes = 0
        self._selection = None
        self._first_line = None
        # The first bytes, held until it is known whether they are a byte-order mark.
        self._head: Optional[bytearray] = bytearray()
        self._bom = False
        if start_line == 0:
            self._begin(0)

    def push(self, chunk: bytes) -> None:
        if self._head is not None:
            take = min(3 - len(self._head), len(chunk))
            self._head.extend(chunk[:take])
            if len(self._head) < 3:
                return
            chunk = self._release_head(chunk, take)
        self._process(chunk)

    def _release_head(self, chunk: bytes, taken: int) -> bytes:
        head = bytes(self._head)
        self._head = None
        self._bom = starts_with_bom(head)
        self._process(head)
        return chunk[taken:]

    def _process(self, chunk: bytes) -> None:
        base = self._position
        start = 0
        index = chunk.find(NEWLINE)
        while index != -1:
            # The newline ends line `self._newlines`. It belongs to the selection between selected lines only.
            self._feed(chunk, base, start, index)
            line = self._newlines
            position = base + index
            if line == self._start_line:
                self._end_first_line(position)
            if line == self._end_line - 1:
                self._end_selection(position)
            self._feed(chunk, base, index, index + 1)
            start = index + 1
            self._newlines += 1
            self._line_start = position + 1
            if self._newlines == self._start_line:
                self._begin(self._line_start)
            if self._newlines == self._end_line - 1:
                self._last_line_start = self._line_start
            index = chunk.find(NEWLINE, start)
        self._feed(chunk, base, start, len(chunk))
        self._position += len(chunk)

    def finish(self) -> LineScan:
        if self._head is not None:
            self._release_head(b"", 0)
        size = self._position
        if self._start is None:
            return LineScan(
                newlines=self._newlines,
                start=size,
                end=size,
                first_line_end=size,
                last_line_start=size,
                selected_bytes=0,
                first_line_bytes=0,
            )
        if self._first_line_end is None:
            self._end_first_line(size)
        if self._end is None:
            self._end_selection(size)
        return LineScan(
            newlines=self._newlines,
            start=self._start,
            end=size if self._end is None else self._end,
            first_line_end=size if self._first_line_end is None else self._first_line_end,
            # A selection that reaches past the last line ends with the last line.
            last_line_start=self._line_start if self._last_line_start is None else self._last_line_start,
            selected_bytes=self._selected_bytes,
            first_line_bytes=self._first_line_bytes,
        )

    def _begin(self, start: int) -> None:
        self._start = start
        if self._start_line == self._end_line - 1:
            self._last_line_start = start
        self._selection = range_decoder()
        self._first_line = range_decoder()

    def _end_first_line(self, position: int) -> None:
        self._first_line_end = position
        if self._first_line is not None:
            self._first_line_bytes += _decoded_bytes(self._first_line.decode())
        self._first_line = None

    def _end_selection(self, position: int) -> None:
        self._end = position
        if self._selection is not None:
            self._selected_bytes += _decoded_bytes(self._selection.decode())
        self._selection = None

    def _feed(self, chunk: bytes, base: int, start: int, end: int) -> None:
        # Decoding the whole file drops a leading byte-order mark.
        if self._bom and base + start < 3:
            start = min(end, 3 - base)
        if end <= start:
            return
        data = chunk[start:end]
        if self._selection is not None:
            self._selected_bytes += _decoded_bytes(self._selection.decode(data))
        if self._first_line is not None:
            self._first_line_bytes += _decoded_bytes(self._first_line.decode(data))


class _Infinite:
    """Sentinel for "to the end"; like ``Number.POSITIVE_INFINITY``."""

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __sub__(self, other: object) -> "_Infinite":
        return self  # infinity minus anything is infinity

    def __repr__(self) -> str:
        return "INFINITE"


_INFINITE = _Infinite()
