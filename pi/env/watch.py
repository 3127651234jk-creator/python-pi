"""A watcher running in the daemon (port of ``packages/env/src/watch.ts``).

The daemon runs Durable's ``NodeFileWatcher`` next to the files. When the
connection is lost, the watcher is opened again once the connection can be
started again, and reports ``overflow``: changes in between were not seen.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..durable.env import Context, FileError, WatchChange, WatchTarget
from .connection import Connection, Json, RemoteError, is_connection_lost
from .errors import to_file_error

RECONNECT_FIRST_MS = 1000
RECONNECT_MAX_MS = 30_000


@dataclass
class RemoteWatchOptions:
    """How the daemon watches, like ``NodeExecutionEnv``'s ``NodeWatchOptions``."""

    # Force a mode; by default Windows and file systems that do not report remote changes poll.
    mode: Optional[str] = None  # "native" | "polling"
    # Interval between snapshots in ``polling`` mode; default 2000 ms.
    poll_interval_ms: Optional[int] = None
    # Most directories one watcher covers; default 10,000.
    max_directories: Optional[int] = None

    def to_json(self) -> Json:
        json: Json = {}
        if self.mode is not None:
            json["mode"] = self.mode
        if self.poll_interval_ms is not None:
            json["pollIntervalMs"] = self.poll_interval_ms
        if self.max_directories is not None:
            json["maxDirectories"] = self.max_directories
        return json


class RemoteWatcher:
    """
    Watches resolved targets through the daemon; open with ``open()``. When the connection is lost,
    it is opened again once the connection can be started again, and reports ``overflow``.
    """

    def __init__(
        self,
        connection: Connection,
        targets: Sequence[WatchTarget],
        on_change: Callable[[WatchChange], None],
        options: RemoteWatchOptions,
    ) -> None:
        self._connection = connection
        self._targets = list(targets)
        self._on_change = on_change
        self._options = options
        self._mode = "native"  # "native" | "polling"
        self._closed = False
        self._controller: Optional[asyncio.Event] = None
        self._request: Optional["asyncio.Task"] = None
        self._reconnect_task: Optional["asyncio.Task"] = None
        self._wake: Optional[asyncio.Event] = None

    @classmethod
    async def open(
        cls,
        connection: Connection,
        targets: Sequence[WatchTarget],
        on_change: Callable[[WatchChange], None],
        options: Optional[RemoteWatchOptions] = None,
    ) -> "RemoteWatcher":
        """Watch ``targets`` (resolved paths); fails like ``NodeFileWatcher.open`` if coverage cannot be established."""
        watcher = cls(connection, targets, on_change, options or RemoteWatchOptions())
        await watcher._start()
        return watcher

    @property
    def mode(self) -> str:
        return self._mode

    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        if self._wake is not None:
            self._wake.set()
        if self._controller is not None:
            self._controller.set()
        request = self._request
        if request is not None:
            try:
                await request
            except asyncio.CancelledError:
                pass

    def _deliver(self, change: WatchChange) -> None:
        if self._closed:
            return
        try:
            self._on_change(change)
        except Exception:
            pass  # A throwing callback must not stop watching.

    async def _start(self) -> None:
        """Open the daemon's watcher; raises if coverage cannot be established."""
        loop = asyncio.get_running_loop()
        ready_future: "asyncio.Future" = loop.create_future()
        ready = False
        controller = asyncio.Event()
        self._controller = controller

        def on_event(event: Json, payload: bytes) -> None:
            nonlocal ready
            kind = event.get("kind")
            if kind == "ready":
                ready = True
                self._mode = "polling" if event.get("mode") == "polling" else "native"
                if not ready_future.done():
                    ready_future.set_result(None)
            elif kind == "change":
                if event.get("mode") == "polling":
                    self._mode = "polling"
                if event.get("overflow") is True:
                    self._deliver({"overflow": True})
                else:
                    self._deliver({"paths": event.get("paths") or []})
            elif kind == "error":
                code = "permission_denied" if event.get("code") == "permission_denied" else "invalid"
                self._deliver({"error": FileError(code, str(event.get("message")))})
                self._closed = True

        async def run() -> None:
            try:
                await self._connection.request(
                    "watch",
                    {
                        "targets": [
                            {
                                "path": target.path,
                                "recursive": target.recursive is True,
                                "exclude": {
                                    "hidden": bool((target.exclude or {}).get("hidden", False)),
                                    "names": list((target.exclude or {}).get("names") or []),
                                },
                            }
                            for target in self._targets
                        ],
                        **self._options.to_json(),
                    },
                    abort_signal=controller,
                    on_event=on_event,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                if not ready:
                    if not ready_future.done():
                        ready_future.set_exception(error)
                    return
                if self._closed:
                    return
                if is_connection_lost(error):
                    # Stored: a task no one references can be garbage-collected mid-run.
                    self._reconnect_task = asyncio.ensure_future(self._reconnect())
                    return
                self._deliver({"error": to_file_error(error)})
                self._closed = True
            finally:
                if not ready_future.done():
                    # Closed or lost before it was ready: ``open()`` must not wait forever.
                    ready_future.set_exception(RemoteError({"code": "unknown", "message": "Watcher closed"}))

        self._request = asyncio.ensure_future(run())
        try:
            await ready_future
        except asyncio.CancelledError:
            # Nobody will use the watcher: cancel the request instead of leaking it.
            self._closed = True
            controller.set()
            raise

    async def _reconnect(self) -> None:
        delay = RECONNECT_FIRST_MS / 1000
        while not self._closed:
            wake = asyncio.Event()
            self._wake = wake
            try:
                await asyncio.wait_for(wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._wake = None
            if self._closed:
                return
            try:
                await self._start()
                # Changes while disconnected were not seen.
                self._deliver({"overflow": True})
                return
            except Exception as error:  # noqa: BLE001
                if is_connection_lost(error):
                    delay = min(delay * 2, RECONNECT_MAX_MS / 1000)
                    continue
                self._deliver({"error": to_file_error(error)})
                self._closed = True
                return
