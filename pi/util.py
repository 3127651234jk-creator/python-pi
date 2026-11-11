"""Shared runtime helpers: abort signal and async event stream.

Ports of the AbortSignal contract and ``EventStream`` from
``packages/ai/src/utils/event-stream.ts``, adapted to asyncio.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Generic, Optional, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


class AbortError(Exception):
    """Raised when an operation is aborted via AbortSignal."""


class AbortSignal:
    """Asyncio-friendly replacement for the DOM AbortSignal used by pi."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def abort(self) -> None:
        self._event.set()

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise AbortError("Operation aborted")

    async def wait(self) -> None:
        await self._event.wait()

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` once when aborted (immediately if already aborted)."""

        if self.aborted:
            listener()
            return lambda: None

        async def waiter() -> None:
            await self._event.wait()
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
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()
        self._sentinel = object()
        self._is_done = is_done
        self._get_result = get_result
        self._result: Optional[U] = None
        self._ended = False
        self._completion = asyncio.Event()

    def push(self, event: T) -> None:
        if self._ended:
            raise RuntimeError("EventStream already ended")
        self._queue.put_nowait(event)
        if self._is_done(event):
            self.end(self._get_result(event))

    def end(self, result: U) -> None:
        if self._ended:
            return
        self._result = result
        self._ended = True
        self._completion.set()
        self._queue.put_nowait(self._sentinel)

    async def result(self) -> U:
        await self._completion.wait()
        return cast(U, self._result)

    def __aiter__(self) -> "EventStream[T, U]":
        return self

    async def __anext__(self) -> T:
        event = await self._queue.get()
        if event is self._sentinel:
            # Keep completion visible to later and concurrent consumers.
            self._queue.put_nowait(self._sentinel)
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
