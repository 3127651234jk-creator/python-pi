"""Durable coding-agent service host: session catalog, worker manager, routed host, CLI.

Compact port of ``packages/coding-agent/src/experimental/{server,session-catalog,
session-worker-manager}.ts``. A session is a directory under the session root
holding ``meta.json`` and worker-owned ``session.sqlite``; one worker process
per session serves ``pi.agent-controller`` over the routed protocol on its stdio
(``pi/durable/worker.py``). The server process holds no storage handles: it
resolves session metadata from the catalog plus tracked workers, and observes
worker exits to resolve each handle's ``terminated`` future.

Server-scoped services are chord contracts (``pi/chord``): ``pi.session-directory``
publishes its replicated state to subscribers and ``pi.session-management``
exposes ``create``/``attach``/``detach``/``remove`` with TS ``SessionSummary``
results.

Deviations from the TS stack: no coordinator (workers connect over stdio), no
lockfile/server profile (one server per session directory, identity by
``--server-id``), no plugin packages, no relay/auth, and the transport is TCP
loopback or a Unix socket instead of Unix-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from ..ai.models import ModelError
from ..chord import RemoteServiceEndpoint, RemoteServiceProvider, ReplicatedState
from ..client import Client, ServerError
from ..server import Server
from .worker import SESSION_SERVICE_ID

SESSION_DIRECTORY_SERVICE = "pi.session-directory"
SESSION_MANAGEMENT_SERVICE = "pi.session-management"
WORKER_STOP_TIMEOUT = 5.0
WORKER_CONNECT_TIMEOUT = 10.0
REQUEST_TIMEOUT = 120.0


class SessionNotFoundError(RuntimeError):
    def __init__(self, session_id) -> None:
        super().__init__(f"Session not found: {session_id}")
        self.code = "session_not_found"


# ---------------------------------------------------------------------------
# Session catalog (compact port of session-catalog.ts)
# ---------------------------------------------------------------------------


def _session_path(root: str, session_id: str) -> str:
    # Session ids name directories; anything else is treated as absent, not traversed.
    if not session_id or os.path.basename(session_id) != session_id or session_id in (".", ".."):
        raise SessionNotFoundError(session_id)
    return os.path.join(root, session_id)


def create_session_directory(root: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    path = _session_path(root, session_id)
    os.makedirs(path)
    metadata = {
        "id": session_id,
        "createdMs": int(time.time() * 1000),
        "cwd": os.path.abspath(cwd) if cwd else os.getcwd(),
        "storagePath": os.path.join(path, "session.sqlite"),
    }
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    return metadata


def _read_meta(path: str) -> Dict[str, Any]:
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as handle:
        return json.load(handle)


def list_session_directories(root: str) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return sessions
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "meta.json")):
            try:
                sessions.append(_read_meta(path))
            except (OSError, ValueError):
                continue
    return sessions


def read_session_directory(root: str, session_id: str) -> Dict[str, Any]:
    path = _session_path(root, session_id)
    if not os.path.isfile(os.path.join(path, "meta.json")):
        raise SessionNotFoundError(session_id)
    return _read_meta(path)


def delete_session_directory(metadata: Dict[str, Any]) -> None:
    shutil.rmtree(os.path.dirname(metadata["storagePath"]), ignore_errors=True)


# ---------------------------------------------------------------------------
# Worker manager (compact port of session-worker-manager.ts)
# ---------------------------------------------------------------------------


def _package_root() -> str:
    # pi/durable/server.py -> pi/durable -> pi -> the importable repo root.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SessionWorkerHandle:
    """Routed session handle owning one worker process; shared by all attachments."""

    def __init__(self, manager: "SessionWorkerManager", metadata: Dict[str, Any],
                 server_id: str, process, client: Client) -> None:
        self.manager = manager
        self.metadata = metadata
        self.server_id = server_id
        self.process = process
        self.client = client
        # None after an expected close; an Exception instance on unexpected termination.
        self.terminated = asyncio.get_running_loop().create_future()
        self.publishers = set()
        self.pump = asyncio.create_task(self._pump_updates())

    async def attach_client(self) -> "WorkerServiceEndpoint":
        return WorkerServiceEndpoint(self)

    async def close(self) -> None:
        await self.manager.close_session(self.metadata["id"])

    def cancel_pump(self) -> None:
        self.pump.cancel()

    async def _pump_updates(self) -> None:
        """Forward worker service updates to every attachment's in-flight publishers."""
        while True:
            message = await self.client.updates.get()
            for publish in list(self.publishers):
                try:
                    await publish(message["subscriptionId"], message["update"])
                except (ConnectionError, OSError):
                    self.publishers.discard(publish)


class WorkerServiceEndpoint:
    """Forwards outer session service calls to the worker's routed connection."""

    def __init__(self, handle: SessionWorkerHandle) -> None:
        self.handle = handle

    async def invoke_service(self, call: dict, publish, cancelled) -> Any:
        self.handle.publishers.add(publish)
        try:
            # Cancellation only stops the caller's waiting; admitted durable work runs on.
            return await self.handle.client.request(
                {"serverId": self.handle.server_id}, call, timeout=REQUEST_TIMEOUT)
        finally:
            self.handle.publishers.discard(publish)

    async def release(self) -> None:
        return None


class SessionWorkerManager:
    """Spawns and tracks one worker process per session; monitors their exits."""

    def __init__(self, session_root: str, model_args: Optional[List[str]] = None, concurrency: int = 1) -> None:
        self.session_root = session_root
        self.model_args = list(model_args or [])
        self.concurrency = max(1, concurrency)
        self.workers: Dict[str, SessionWorkerHandle] = {}
        self.lock = asyncio.Lock()

    async def open_session(self, metadata: Dict[str, Any]) -> SessionWorkerHandle:
        async with self.lock:
            existing = self.workers.get(metadata["id"])
            if existing is not None:
                return existing
            handle = await self._launch(metadata)
            self.workers[metadata["id"]] = handle
            return handle

    async def _launch(self, metadata: Dict[str, Any]) -> SessionWorkerHandle:
        server_id = str(uuid.uuid4())
        argv = [
            sys.executable, "-m", "pi", "durable-worker",
            "--server-id", server_id,
            "--storage", metadata["storagePath"],
            "--cwd", metadata["cwd"],
            "--concurrency", str(self.concurrency),
            *self.model_args,
        ]
        env = dict(os.environ)
        package_root = _package_root()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = package_root if not existing else package_root + os.pathsep + existing
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                cwd=metadata["cwd"], env=env)
        except OSError as exc:
            raise RuntimeError(f"Failed to launch session worker: {exc}") from exc

        client = Client(server_id)
        try:
            await client.connect_streams(process.stdout, process.stdin, timeout=WORKER_CONNECT_TIMEOUT)
        except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
            process.kill()
            await process.wait()
            raise RuntimeError("Session worker exited before readiness") from exc

        handle = SessionWorkerHandle(self, metadata, server_id, process, client)
        asyncio.create_task(self._monitor(handle))
        return handle

    async def _monitor(self, handle: SessionWorkerHandle) -> None:
        code = await handle.process.wait()
        await self._remove_worker(
            handle, RuntimeError(f"Session worker {handle.metadata['id']} exited unexpectedly (code {code})"))

    async def _remove_worker(self, handle: SessionWorkerHandle, error: Optional[Exception]) -> None:
        async with self.lock:
            if self.workers.get(handle.metadata["id"]) is handle:
                del self.workers[handle.metadata["id"]]
        if not handle.terminated.done():
            handle.terminated.set_result(error)
        await handle.client.close()
        handle.cancel_pump()

    async def close_session(self, session_id: str) -> None:
        async with self.lock:
            handle = self.workers.pop(session_id, None)
        if handle is None:
            return
        if not handle.terminated.done():
            handle.terminated.set_result(None)
        try:
            await handle.client.request(
                {"serverId": handle.server_id},
                {"serviceId": SESSION_SERVICE_ID, "member": "shutdown", "args": []}, timeout=WORKER_STOP_TIMEOUT)
        except (ConnectionError, ServerError, asyncio.TimeoutError, OSError):
            pass
        await self._stop_process(handle)
        await handle.client.close()
        handle.cancel_pump()

    async def _stop_process(self, handle: SessionWorkerHandle) -> None:
        try:
            await asyncio.wait_for(handle.process.wait(), WORKER_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            handle.process.terminate()
            try:
                await asyncio.wait_for(handle.process.wait(), WORKER_STOP_TIMEOUT)
            except asyncio.TimeoutError:
                handle.process.kill()
                await handle.process.wait()

    async def shutdown(self) -> None:
        handles = list(self.workers.values())
        await asyncio.gather(*(self.close_session(handle.metadata["id"]) for handle in handles),
                             return_exceptions=True)


# ---------------------------------------------------------------------------
# Routed host (compact port of experimental/server.ts startServerBackend)
# ---------------------------------------------------------------------------


def _session_id_arg(first: Any) -> Optional[str]:
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        value = first.get("sessionId")
        return str(value) if value is not None else None
    return None


class SessionDirectoryState:
    """Shared ``pi.session-directory`` replicated state in the TS ``SessionDirectoryState``
    shape: ``{revision, sessions: [{serverId, sessionId, createdAt}]}``. One instance
    serves every connection; ``refresh`` publishes a root replacement on changes."""

    def __init__(self, host: "DurableServerHost") -> None:
        self.host = host
        self.revision = 0
        self.state = ReplicatedState(self._value())

    def _summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {"serverId": self.host.server_id, "sessionId": metadata["id"],
                "createdAt": metadata["createdMs"]}

    def _value(self) -> Dict[str, Any]:
        return {"revision": self.revision,
                "sessions": [self._summary(item) for item in list_session_directories(self.host.session_root)]}

    def refresh(self) -> None:
        """Re-read the catalog and publish one revision."""
        self.revision += 1
        self.state.replace(self._value())


class _DirectoryService:
    """Provider view of the shared session directory (member: ``state``)."""

    def __init__(self, directory: SessionDirectoryState) -> None:
        self.state = directory.state


class _ManagementService:
    """``pi.session-management`` contract: ``create``/``attach``/``detach``/``remove``."""

    def __init__(self, host: "DurableServerHost", presentation) -> None:
        self._host = host
        self._presentation = presentation

    async def create(self, options: Any = None, context: Any = None) -> Dict[str, Any]:
        cwd = options.get("cwd") if isinstance(options, dict) else None
        metadata = create_session_directory(self._host.session_root, cwd)
        self._host.directory.refresh()
        return {"serverId": self._host.server_id, "sessionId": metadata["id"],
                "createdAt": metadata["createdMs"]}

    async def attach(self, session_id: Any, context: Any = None) -> None:
        resolved = _session_id_arg(session_id)
        if resolved is None:
            raise SessionNotFoundError(session_id)
        await self._presentation.attach_session(resolved)

    async def detach(self, context: Any = None) -> None:
        await self._presentation.detach_session()

    async def remove(self, session_id: Any, context: Any = None) -> None:
        resolved = _session_id_arg(session_id)
        if resolved is None:
            raise SessionNotFoundError(session_id)
        metadata = read_session_directory(self._host.session_root, resolved)
        # Release routed attachments and close the worker before deleting durable metadata.
        await self._presentation.prepare_session_removal(resolved)
        delete_session_directory(metadata)
        self._host.directory.refresh()


class ServerServicesEndpoint:
    """Per-connection server services served by one chord provider endpoint."""

    def __init__(self, host: "DurableServerHost", presentation) -> None:
        self.host = host
        provider = RemoteServiceProvider([{"id": SESSION_DIRECTORY_SERVICE},
                                          {"id": SESSION_MANAGEMENT_SERVICE}])
        provider.provide(SESSION_DIRECTORY_SERVICE, _DirectoryService(host.directory))
        provider.provide(SESSION_MANAGEMENT_SERVICE, _ManagementService(host, presentation))
        self.provider = provider
        self.endpoint = RemoteServiceEndpoint(provider)

    async def invoke_service(self, call: dict, publish, cancelled) -> Any:
        return await self.endpoint.invoke_service(call, publish, cancelled)

    async def release(self) -> None:
        self.endpoint.dispose()
        self.provider.dispose()


class DurableServerHost:
    """Binds the session catalog and worker manager to the routed ``Server``."""

    def __init__(self, manager: SessionWorkerManager) -> None:
        self.manager = manager
        self.server_id: Optional[str] = None
        self.directory = SessionDirectoryState(self)

    @property
    def session_root(self) -> str:
        return self.manager.session_root

    async def attach_client(self, presentation) -> ServerServicesEndpoint:
        return ServerServicesEndpoint(self, presentation)

    async def resolve_session(self, session_id: str) -> Dict[str, Any]:
        handle = self.manager.workers.get(session_id)
        if handle is not None:
            return handle.metadata
        return read_session_directory(self.session_root, session_id)

    async def open_session(self, metadata: Dict[str, Any]) -> SessionWorkerHandle:
        return await self.manager.open_session(metadata)


# ---------------------------------------------------------------------------
# CLI entrypoint (compact port of experimental/commands.ts runServerCommand)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi server", description="Durable coding-agent routed server")
    parser.add_argument("--session-dir", default="durable-sessions", help="Directory holding session subdirectories")
    parser.add_argument("--server-id", default=None, help="Routed server identity (default: generated)")
    parser.add_argument("--host", default="127.0.0.1", help="TCP bind host")
    parser.add_argument("--port", type=int, default=0, help="TCP port (0: dynamically allocated)")
    parser.add_argument("--unix", default=None, help="Unix socket path (where the platform supports it)")
    parser.add_argument("--provider", help="Worker model provider (anthropic | openai | faux | ...)")
    parser.add_argument("--model", help="Worker model id")
    parser.add_argument("--api-key", help="Worker API key")
    parser.add_argument("--base-url", help="Worker provider base URL override")
    parser.add_argument("--concurrency", type=int, default=1, help="Worker scheduler concurrency")
    return parser


def _model_args(args: argparse.Namespace) -> List[str]:
    extra: List[str] = []
    for flag in ("provider", "model", "api_key", "base_url"):
        value = getattr(args, flag)
        if value:
            extra += [f"--{flag.replace('_', '-')}", str(value)]
    return extra


async def async_main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.session_dir, exist_ok=True)
    manager = SessionWorkerManager(args.session_dir, model_args=_model_args(args), concurrency=args.concurrency)
    host = DurableServerHost(manager)
    server = Server(host, server_id=args.server_id)
    host.server_id = server.server_id
    if args.unix:
        print(f"Socket: {await server.start_unix(args.unix)}", flush=True)
    else:
        bound_host, port, *_rest = await server.start_tcp(args.host, args.port)
        print(f"Address: {bound_host} {port}", flush=True)
    print(f"Server: {server.server_id}", flush=True)
    closed = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, closed.set)
        except (NotImplementedError, RuntimeError):
            pass  # Windows: SIGINT arrives as KeyboardInterrupt, SIGTERM cannot be handled
    try:
        await closed.wait()  # foreground server: runs until a signal
    finally:
        await server.close()
        await manager.shutdown()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
