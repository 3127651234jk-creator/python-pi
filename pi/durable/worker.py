"""Durable session worker: one process owning one session's storage.

Compact port of the TS session worker and its services
(``packages/coding-agent/src/experimental/session-worker.ts`` and
``services/worker.ts``). The worker serves the routed protocol over its stdio;
the server process connects with ``Client.connect_streams`` and forwards
session service calls to it. It owns the session's storage exclusively, so its
death is the failure signal: the manager observes the process exit, resolves
the session handle's ``terminated`` future, and the next open replays from
durable state (running tasks reset to pending on reopen).

Deviations from the TS worker: no coordinator process (the routed stdio
connection is the only channel), no idle retirement (the worker lives until a
``shutdown`` member or the parent's death), and no plugin packages.

stdout carries protocol frames only; everything else logs to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from typing import Any, Dict, Optional

from ..ai.json_io import to_json as ai_to_json
from ..ai.models import ModelError, resolve_model
from ..coding_agent.tools import create_default_tools
from ..server import Server
from .cli import open_storage
from .harness import Harness
from .session import create_session
from .tasks import register_builtin_tasks
from .types import record_to_dict

SESSION_SERVICE_ID = "pi.agent-controller"
SCHEDULER_TICK_SECONDS = 1.0

GENERATION_TASK = "pi.generation"
ACTIVE_STATUSES = ("pending", "running")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi durable-worker", description="One durable session worker (internal)")
    parser.add_argument("--server-id", required=True, help="Routed server identity the manager expects")
    parser.add_argument("--storage", required=True, help="Session storage path (sqlite or jsonl)")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for the session's tools")
    parser.add_argument("--provider", help="anthropic | openai | faux | ...")
    parser.add_argument("--model", help="Model id")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="Provider base URL override")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent task executions")
    return parser


class _StdoutWriter:
    """Minimal async writer over the real stdout buffer (drain = flush)."""

    def __init__(self, buffer) -> None:
        self._buffer = buffer

    def write(self, data: bytes) -> None:
        self._buffer.write(data)

    async def drain(self) -> None:
        self._buffer.flush()

    def close(self) -> None:
        try:
            self._buffer.flush()
        except (OSError, ValueError):
            pass

    async def wait_closed(self) -> None:
        return None


async def _stdio_streams():
    """The routed connection over our own stdio pipes.

    A daemon thread pumps stdin into an asyncio ``StreamReader``: on Windows the
    proactor loop cannot register the inherited stdin pipe handle with its IO
    completion port, so a threaded blocking read is the portable bridge.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=8 * 1024 * 1024, loop=loop)

    def pump() -> None:
        # os.read on the raw fd: a blocked daemon thread must not hold the
        # BufferedReader lock, or interpreter shutdown deadlocks and crashes.
        fd = sys.stdin.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                loop.call_soon_threadsafe(reader.feed_eof)
                return
            loop.call_soon_threadsafe(reader.feed_data, chunk)

    threading.Thread(target=pump, name="stdin-pump", daemon=True).start()
    return reader, _StdoutWriter(sys.stdout.buffer)


def _entry_to_json(entry) -> Dict[str, Any]:
    data: Dict[str, Any] = {"id": entry.id, "kind": entry.kind}
    if entry.head is not None:
        data["head"] = entry.head
    data["messages"] = [ai_to_json(message) for message in (entry.model or [])]
    return data


class SessionServices:
    """State shared by the worker's endpoints: the harness, its conversation, and the stop event."""

    def __init__(self, harness: Harness, conversation_id: int, stop: asyncio.Event) -> None:
        self.harness = harness
        self.conversation_id = conversation_id
        self.stop = stop

    async def abortable_task(self, task_id: Optional[int]) -> Optional[int]:
        if task_id is not None:
            return task_id
        records = await self.harness.list_tasks(self.conversation_id)
        for record in reversed(records):
            if record.kind == GENERATION_TASK and record.state.status in ACTIVE_STATUSES:
                return record.id
        return None


class SessionEndpoint:
    """Serves ``pi.agent-controller`` for one routed connection to the worker."""

    def __init__(self, services: SessionServices) -> None:
        self.services = services

    async def invoke_service(self, call: dict, publish, cancelled) -> Any:
        member = call.get("member")
        args = call.get("args") or []
        first = args[0] if args else None
        harness = self.services.harness
        if member == "prompt":
            text = first.get("text") if isinstance(first, dict) else first
            task_id = await harness.submit_input(self.services.conversation_id, str(text or ""))
            return {"taskId": task_id}
        if member == "abort":
            requested = first.get("taskId") if isinstance(first, dict) else first
            task_id = await self.services.abortable_task(requested)
            if task_id is None:
                return {"aborted": False}
            await harness.abort_task(task_id)
            return {"aborted": True, "taskId": task_id}
        if member == "transcript":
            entries = await harness.transcript(self.services.conversation_id)
            return {"entries": [_entry_to_json(entry) for entry in entries]}
        if member == "tasks":
            records = await harness.list_tasks(self.services.conversation_id)
            return {"tasks": [record_to_dict(record) for record in records]}
        if member == "compact":
            task_id = await harness.submit_compact_task(self.services.conversation_id)
            return {"taskId": task_id}
        if member == "shutdown":
            self.services.stop.set()
            return None
        raise ValueError(f"Unknown member {member!r}")

    async def release(self) -> None:
        return None


class SessionWorkerHost:
    """Routed host of the worker: one server-scoped service, no session routes."""

    def __init__(self, services: SessionServices) -> None:
        self.services = services

    async def attach_client(self, presentation) -> SessionEndpoint:
        return SessionEndpoint(self.services)

    async def resolve_session(self, session_id):
        raise RoutingRejected("The worker serves no session routes")

    async def open_session(self, metadata):
        raise RoutingRejected("The worker serves no session routes")


class RoutingRejected(RuntimeError):
    code = "invalid_target"

    def __init__(self, message: str) -> None:
        super().__init__(message)


async def run_scheduler(harness: Harness, session, stop: asyncio.Event, concurrency: int) -> None:
    """Drive the pull-only scheduler; commits (submissions, settlements, aborts) wake it."""
    wake = asyncio.Event()
    session.subscribe_commits(lambda publication: wake.set())
    while not stop.is_set():
        try:
            await asyncio.wait_for(wake.wait(), SCHEDULER_TICK_SECONDS)
        except asyncio.TimeoutError:
            pass
        wake.clear()
        try:
            await harness.run_pending_tasks(max(1, concurrency))
        except Exception as exc:  # noqa: BLE001 - a failed task is recorded durably; keep scheduling
            print(f"session worker: scheduler error: {type(exc).__name__}: {exc}", file=sys.stderr)


async def root_conversation(harness: Harness) -> int:
    """The storage's first conversation, or a fresh one on first open (TS ``Harness.root``)."""

    async def change(tx):
        page = await tx.scan_conversations({}, 1)
        if page.items:
            return page.items[0].id
        record = await tx.create_conversation({"kind": "ownerless"})
        return record.id

    return await harness.session.commit(change)


async def async_main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    model = resolve_model(args.provider, args.model, args.api_key, args.base_url)
    storage = open_storage(args.storage)
    session = create_session(storage)
    harness = Harness(session, model)
    register_builtin_tasks(harness, tools={tool.name: tool for tool in create_default_tools(args.cwd)})
    conversation_id = await root_conversation(harness)

    stop = asyncio.Event()
    services = SessionServices(harness, conversation_id, stop)
    scheduler = asyncio.create_task(run_scheduler(harness, session, stop, args.concurrency))

    server = Server(SessionWorkerHost(services), server_id=args.server_id)
    reader, writer = await _stdio_streams()
    serve_task = await server.start_streams(reader, writer)
    stopped = asyncio.create_task(stop.wait())
    await asyncio.wait({serve_task, stopped}, return_when=asyncio.FIRST_COMPLETED)

    stopped.cancel()
    scheduler.cancel()
    # The client keeps the stream open while it waits for this process to
    # exit, so serve_task cannot finish on its own. Give a pending shutdown
    # response a moment to flush, then stop serving; otherwise close_session
    # waits out WORKER_STOP_TIMEOUT and kills the process before
    # server.close()/storage.close() run.
    try:
        await asyncio.wait_for(asyncio.shield(serve_task), 0.5)
    except asyncio.TimeoutError:
        serve_task.cancel()
    await asyncio.gather(scheduler, stopped, serve_task, return_exceptions=True)
    await server.close()
    await storage.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
