"""Shared runtime helpers: abort signal, async event stream, ANSI handling.

Ports of the AbortSignal contract and ``EventStream`` from
``packages/ai/src/utils/event-stream.ts``, and of ``splitIncompleteAnsiSuffix``
/ ``stripAnsi`` from ``packages/coding-agent/src/utils/ansi.ts`` (itself
derived from chalk's ansi-regex / strip-ansi, MIT), adapted to asyncio and
``re``.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator, Callable, Generic, List, Optional, Tuple, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


# ---------------------------------------------------------------------------
# ANSI escape handling (port of packages/coding-agent/src/utils/ansi.ts)
# ---------------------------------------------------------------------------

# Valid string terminator sequences are BEL, ESC\, and 0x9c.
_ST = r"(?:\x07|\x1b\\|\x9c)"
# OSC sequences: ESC ] ... ST.
_OSC_START = r"\x1b\]"
# CSI and related: ESC/C1, optional intermediates, optional params (supports ;
# and :), then final byte.
_CSI_START = r"[\x1b\x9b][\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?"
_CSI_FINAL = r"[\dA-PR-TZcf-nq-uy=><~]"

# Complete sequences; OSC is non-greedy until the first ST.
_ANSI_PATTERN = re.compile(rf"(?:{_OSC_START}[\s\S]*?{_ST})|{_CSI_START}{_CSI_FINAL}")
# Unfinished sequence at the very end of the text: OSC without its ST (a
# trailing ESC may start ESC\), or CSI without its final byte.
_UNFINISHED_ANSI_AT_END = re.compile(
    rf"(?:{_OSC_START}(?:[^\x07\x9c\x1b]|\x1b(?!\\))*|{_CSI_START})\Z")

# Longest unfinished sequence held back while streaming. Longer ones are
# processed as-is.
MAX_PENDING_ANSI_LENGTH = 256


def strip_ansi(value: str) -> str:
    """Remove ANSI escape sequences (SGR, OSC hyperlinks/titles, CSI) from text."""
    if "\x1b" not in value and "\x9b" not in value:
        return value
    return _ANSI_PATTERN.sub("", value)


def split_incomplete_ansi_suffix(value: str) -> Tuple[str, str]:
    """Split streamed text into the part safe to strip now and a trailing
    unfinished escape sequence to prepend to the next chunk."""
    if "\x1b" not in value and "\x9b" not in value:
        return value, ""
    window_start = max(0, len(value) - MAX_PENDING_ANSI_LENGTH)
    match = _UNFINISHED_ANSI_AT_END.search(value, window_start)
    if match is None:
        return value, ""
    return value[:match.start()], value[match.start():]


class AbortError(Exception):
    """Raised when an operation is aborted via AbortSignal."""


class AbortSignal:
    """Asyncio-friendly replacement for the DOM AbortSignal used by pi."""

    def __init__(self) -> None:
        # Created on first async use: on Python 3.9 asyncio.Event binds to the
        # loop current at construction, and signals are often built before the
        # loop that will wait on them exists.
        self._event: Optional[asyncio.Event] = None
        self._aborted = False

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self) -> None:
        self._aborted = True
        if self._event is not None:
            self._event.set()

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise AbortError("Operation aborted")

    async def wait(self) -> None:
        if self._aborted:
            return
        if self._event is None:
            self._event = asyncio.Event()
        await self._event.wait()

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` once when aborted (immediately if already aborted)."""

        if self.aborted:
            listener()
            return lambda: None

        async def waiter() -> None:
            await self.wait()
            listener()

        task = asyncio.get_running_loop().create_task(waiter())

        def remove() -> None:
            task.cancel()

        return remove


class EventStream(Generic[T, U]):
    """Push-based async stream that terminates with a final result value.

    ``end()`` or a terminal event closes the stream; ``result()`` waits for
    completion without consuming events. Consumers iterate with ``async for``.
    """

    def __init__(self, is_done: Callable[[T], bool], get_result: Callable[[T], U]) -> None:
        # Queue/Event are created on first use: on Python 3.9 they bind to the
        # loop current at construction, and streams are often built before the
        # loop that will consume them. Events pushed before that are buffered.
        self._queue: Optional["asyncio.Queue[object]"] = None
        self._pending: List[object] = []
        self._sentinel = object()
        self._is_done = is_done
        self._get_result = get_result
        self._result: Optional[U] = None
        self._ended = False
        self._completion: Optional[asyncio.Event] = None

    def push(self, event: T) -> None:
        if self._ended:
            raise RuntimeError("EventStream already ended")
        if self._queue is None:
            self._pending.append(event)
        else:
            self._queue.put_nowait(event)
        if self._is_done(event):
            self.end(self._get_result(event))

    def end(self, result: U) -> None:
        if self._ended:
            return
        self._result = result
        self._ended = True
        if self._completion is not None:
            self._completion.set()
        if self._queue is not None:
            self._queue.put_nowait(self._sentinel)

    async def result(self) -> U:
        if not self._ended:
            if self._completion is None:
                self._completion = asyncio.Event()
            await self._completion.wait()
        return cast(U, self._result)

    def _ensure_queue(self) -> "asyncio.Queue[object]":
        if self._queue is None:
            # Created inside the consumer's loop; replay what was pushed before.
            self._queue = asyncio.Queue()
            for event in self._pending:
                self._queue.put_nowait(event)
            self._pending.clear()
            if self._ended:
                self._queue.put_nowait(self._sentinel)
        return self._queue

    def __aiter__(self) -> "EventStream[T, U]":
        return self

    async def __anext__(self) -> T:
        queue = self._ensure_queue()
        event = await queue.get()
        if event is self._sentinel:
            # Keep completion visible to later and concurrent consumers.
            queue.put_nowait(self._sentinel)
            raise StopAsyncIteration
        return cast(T, event)


class AsyncEventStream(Generic[T]):
    """Simple pull-based wrapper that turns an async generator into a stream."""

    def __init__(self, iterator: AsyncIterator[T]) -> None:
        self._iterator = iterator

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iterator

    async def __anext__(self) -> T:
        return await self._iterator.__anext__()
