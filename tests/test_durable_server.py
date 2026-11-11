"""End-to-end tests for the durable coding-agent service host.

Runs the real routed server over loopback TCP with real worker subprocesses
(one per session, faux provider), covering the session lifecycle, generation,
worker exit monitoring, crash recovery, removal, and the CLI entrypoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.chord import ClientServiceTransport, RemoteServiceBinding
from pi.client import Client, ServerError
from pi.durable.server import (
    DurableServerHost,
    SessionWorkerManager,
)
from pi.server import Server

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def management(member, *args):
    return {"serviceId": "pi.session-management", "member": member, "args": list(args)}


def controller(member, *args):
    return {"serviceId": "pi.agent-controller", "member": member, "args": list(args)}


async def until(predicate, timeout=10.0):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.05)
    await asyncio.wait_for(wait(), timeout)


async def wait_for_transcript(client, minimum, timeout=20.0):
    async def poll():
        while True:
            result = await client.request(client.attachment, controller("transcript"), timeout=10)
            if len(result["entries"]) >= minimum:
                return result["entries"]
            await asyncio.sleep(0.05)
    return await asyncio.wait_for(poll(), timeout)


async def _create_and_attach(server_id, client, cwd, timeout=20.0):
    target = {"serverId": server_id}
    created = await client.request(target, management("create", {"cwd": cwd}), timeout=timeout)
    await client.request(target, management("attach", {"sessionId": created["sessionId"]}), timeout=timeout)
    return created["sessionId"]


def test_session_lifecycle_and_generation(tmp_path):
    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "worker answer"}])
        root = str(tmp_path / "sessions")
        manager = SessionWorkerManager(root, model_args=["--provider", "faux", "--model", "faux"])
        server = Server(DurableServerHost(manager))
        address = await server.start_tcp()
        client = Client(server.server_id)
        try:
            await client.connect_tcp(*address[:2])
            session_id = await _create_and_attach(server.server_id, client, str(tmp_path))
            assert os.path.isfile(os.path.join(root, session_id, "meta.json"))
            assert client.attachment["sessionId"] == session_id

            await client.request(client.attachment, controller("prompt", {"text": "hello"}), timeout=20)
            entries = await wait_for_transcript(client, 2)
            assert entries[0]["kind"] == "pi.user"
            assert entries[0]["messages"][0]["content"] == "hello"
            assert entries[1]["kind"] == "pi.assistant"
            assert entries[1]["messages"][0]["content"][0]["text"] == "worker answer"

            # The tasks member reports the generation task; its terminal outcome is durable.
            tasks = await client.request(client.attachment, controller("tasks"), timeout=10)
            assert any(task["kind"] == "pi.generation" for task in tasks["tasks"])
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            await client.close()
            await server.close()
            await manager.shutdown()
    asyncio.run(asyncio.wait_for(check(), 120))


def test_worker_shutdown_exits_without_waiting_for_timeout(tmp_path):
    """close_session must not stall for WORKER_STOP_TIMEOUT (5s) on every close."""

    async def check():
        root = str(tmp_path / "sessions")
        manager = SessionWorkerManager(root, model_args=["--provider", "faux", "--model", "faux"])
        server = Server(DurableServerHost(manager))
        address = await server.start_tcp()
        client = Client(server.server_id)
        try:
            await client.connect_tcp(*address[:2])
            session_id = await _create_and_attach(server.server_id, client, str(tmp_path))
            started = time.monotonic()
            await manager.close_session(session_id)
            elapsed = time.monotonic() - started
            assert session_id not in manager.workers
            assert elapsed < 3
        finally:
            await client.close()
            await server.close()
            await manager.shutdown()

    asyncio.run(asyncio.wait_for(check(), 60))


def test_worker_exit_monitoring_and_recovery(tmp_path):
    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "persisted answer"}])
        root = str(tmp_path / "sessions")
        manager = SessionWorkerManager(root, model_args=["--provider", "faux", "--model", "faux"])
        server = Server(DurableServerHost(manager))
        address = await server.start_tcp()
        client = Client(server.server_id)
        try:
            await client.connect_tcp(*address[:2])
            session_id = await _create_and_attach(server.server_id, client, str(tmp_path))
            await client.request(client.attachment, controller("prompt", {"text": "hello"}), timeout=20)
            await wait_for_transcript(client, 2)

            # Killing the worker is the failure signal: attachments are released.
            handle = manager.workers[session_id]
            handle.process.kill()
            await handle.process.wait()
            await until(lambda: client.attachment is None)
            assert session_id not in manager.workers

            # A re-attach opens a fresh worker that replays from durable storage.
            await client.request({"serverId": server.server_id},
                                 management("attach", {"sessionId": session_id}), timeout=20)
            entries = await client.request(client.attachment, controller("transcript"), timeout=10)
            assert [entry["kind"] for entry in entries["entries"]] == ["pi.user", "pi.assistant"]
            assert entries["entries"][1]["messages"][0]["content"][0]["text"] == "persisted answer"
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            await client.close()
            await server.close()
            await manager.shutdown()
    asyncio.run(asyncio.wait_for(check(), 120))


def test_session_removal_deletes_metadata(tmp_path):
    async def check():
        root = str(tmp_path / "sessions")
        manager = SessionWorkerManager(root, model_args=["--provider", "faux", "--model", "faux"])
        server = Server(DurableServerHost(manager))
        address = await server.start_tcp()
        client = Client(server.server_id)
        try:
            await client.connect_tcp(*address[:2])
            session_id = await _create_and_attach(server.server_id, client, str(tmp_path))
            await client.request({"serverId": server.server_id},
                                 management("remove", {"sessionId": session_id}), timeout=20)
            await until(lambda: client.attachment is None)
            assert session_id not in manager.workers
            assert not os.path.exists(os.path.join(root, session_id))
            with pytest.raises(ServerError) as error:
                await client.request({"serverId": server.server_id},
                                     management("attach", {"sessionId": session_id}), timeout=20)
            assert error.value.code == "session_not_found"
        finally:
            await client.close()
            await server.close()
            await manager.shutdown()
    asyncio.run(asyncio.wait_for(check(), 120))


def test_server_cli_smoke(tmp_path):
    async def check():
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT if not env.get("PYTHONPATH") else REPO_ROOT + os.pathsep + env["PYTHONPATH"]
        env["FAUX_SCRIPT"] = json.dumps([{"text": "cli answer"}])
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pi", "server", "--session-dir", str(tmp_path / "sessions"),
            "--provider", "faux", "--model", "faux",
            stdout=asyncio.subprocess.PIPE, cwd=REPO_ROOT, env=env)
        try:
            address_line = await asyncio.wait_for(process.stdout.readline(), 20)
            text = address_line.decode().strip()
            assert text.startswith("Address:")
            _prefix, host_name, port = text.split(" ")
            server_line = await asyncio.wait_for(process.stdout.readline(), 20)
            server_id = server_line.decode().strip().split("Server: ")[1]

            client = Client(server_id)
            await client.connect_tcp(host_name, int(port))
            session_id = await _create_and_attach(server_id, client, str(tmp_path))
            assert client.attachment["sessionId"] == session_id
            await client.close()
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 20)
    asyncio.run(asyncio.wait_for(check(), 120))


def test_session_directory_subscription_receives_updates(tmp_path):
    async def check():
        root = str(tmp_path / "sessions")
        manager = SessionWorkerManager(root, model_args=["--provider", "faux", "--model", "faux"])
        host = DurableServerHost(manager)
        server = Server(host)
        host.server_id = server.server_id
        address = await server.start_tcp()
        client = Client(server.server_id)
        transport = None
        binding = None
        try:
            await client.connect_tcp(*address[:2])
            transport = ClientServiceTransport(client, lambda: {"serverId": server.server_id})
            binding = RemoteServiceBinding(["pi.session-directory"], transport)
            directory = binding.use("pi.session-directory")
            await binding.ready()
            assert directory.state.value == {"revision": 0, "sessions": []}

            created = await client.request({"serverId": server.server_id},
                                           management("create", {"cwd": str(tmp_path)}), timeout=20)
            await until(lambda: directory.state.value is not None
                        and any(item["sessionId"] == created["sessionId"]
                                for item in directory.state.value["sessions"]))
            value = directory.state.value
            assert value["revision"] >= 1
            summary = value["sessions"][0]
            assert summary["serverId"] == server.server_id
            assert summary["createdAt"] == created["createdAt"]

            await client.request({"serverId": server.server_id},
                                 management("remove", {"sessionId": created["sessionId"]}), timeout=20)
            await until(lambda: directory.state.value["sessions"] == [])
        finally:
            if binding is not None:
                await binding.close()
            if transport is not None:
                await transport.dispose()
            await client.close()
            await server.close()
            await manager.shutdown()
    asyncio.run(asyncio.wait_for(check(), 120))
