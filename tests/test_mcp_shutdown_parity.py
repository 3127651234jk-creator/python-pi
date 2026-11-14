"""MCP manager shutdown parity: closing must settle connects that are still opening.

Covers M01-M07 of docs/specs/v1.0.4-parity-completion.md.
"""

import asyncio
import sys
import time

import httpx
import pytest

from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.mcp import McpClient, McpManager


def slow_initialize_server(tmp_path):
    """Stdio server that receives initialize and then stays silent."""
    script = tmp_path / "slow_server.py"
    script.write_text(
        "import sys,json,time\n"
        "line = sys.stdin.readline()\n"
        "req = json.loads(line)\n"
        "time.sleep(60)\n"
        'print(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{"protocolVersion":"2025-03-26"}}),flush=True)\n',
        encoding="utf-8")
    return {"command": sys.executable, "args": [str(script)], "timeout": 30}


def http_config(tmp_path, name="remote", **settings):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json",
               {"mcpServers": {name: {"url": "https://example.test/mcp", **settings}}})
    return config


async def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.01)


def test_m01_close_waits_for_stdio_connect_in_progress(tmp_path):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json",
               {"mcpServers": {"slow": slow_initialize_server(tmp_path)}})

    async def scenario():
        manager = McpManager(config)
        reload_task = asyncio.create_task(manager.reload())
        await wait_for(lambda: bool(manager._opening))
        client, _ = next(iter(manager._opening.values()))
        await wait_for(lambda: client.process is not None)
        process = client.process

        await asyncio.wait_for(manager.close(), 5)
        await asyncio.wait_for(reload_task, 5)

        assert process.returncode is not None
        assert client.pending == {}
        assert client.pump_task is None
        assert manager.clients == {}

    asyncio.run(scenario())


def test_m02_close_interrupts_http_connect_and_listener(tmp_path, monkeypatch):
    config = http_config(tmp_path)

    async def scenario():
        started = asyncio.Event()
        transport_closed = asyncio.Event()

        async def handler(request):
            started.set()
            # A real transport fails in-flight requests when the client closes; the
            # mock waits for the close signal to reproduce that behavior.
            await transport_closed.wait()
            raise httpx.ConnectError("transport closed", request=request)

        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        manager = McpManager(config)
        reload_task = asyncio.create_task(manager.reload())
        await asyncio.wait_for(started.wait(), 5)
        client, _ = next(iter(manager._opening.values()))
        http = client.http
        assert http is not None
        original_aclose = http.aclose

        async def aclose():
            transport_closed.set()
            await original_aclose()

        monkeypatch.setattr(http, "aclose", aclose)

        await asyncio.wait_for(manager.close(), 5)
        await asyncio.wait_for(reload_task, 5)

        assert http.is_closed
        assert client.http is None
        assert client.pump_task is None
        assert manager.clients == {}
        # Nothing may register after close returned.
        assert manager._opening == {}

    asyncio.run(scenario())


def test_m03_close_releases_unconnected_replacement(tmp_path, monkeypatch):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"remote": {"url": "https://example.test/mcp"}}})

    async def scenario():
        manager = McpManager(config)
        old = McpClient("remote", {"url": "https://example.test/mcp"}, tmp_path)
        manager.clients["remote"] = old
        replacements = []
        connecting = asyncio.Event()

        async def connect(client, namespace):
            replacements.append(client)
            if client is old:
                return
            connecting.set()
            # A real transport fails once close() tears it down.
            await client.shutdown.wait()
            raise RuntimeError("transport closed")

        async def request(method, params):
            raise RuntimeError("MCP server disconnected")

        monkeypatch.setattr(manager, "_connect_client", connect)
        old.request = request

        call_task = asyncio.create_task(manager.call("remote", "tools/list", {}))
        await asyncio.wait_for(connecting.wait(), 5)
        await asyncio.wait_for(manager.close(), 5)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(call_task, 5)

        assert len(replacements) == 1
        assert replacements[0].closed
        assert old.closed
        assert manager.clients == {}
        assert manager._opening == {}

    asyncio.run(scenario())


def test_m04_concurrent_client_close_waits_for_one_teardown(tmp_path):
    async def scenario():
        client = McpClient("remote", {"command": sys.executable}, tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def teardown():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()

        client._teardown_transports = teardown
        first = asyncio.create_task(client.close())
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        assert calls == 1

    asyncio.run(scenario())


def test_m04_concurrent_manager_close_waits_for_shared_cleanup(tmp_path):
    class SlowClient:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def close(self):
            self.calls += 1
            self.entered.set()
            await self.release.wait()

    async def scenario():
        manager = McpManager(Configuration(str(tmp_path), str(tmp_path / "user")))
        client = SlowClient()
        manager.clients["slow"] = client
        first = asyncio.create_task(manager.close())
        await asyncio.wait_for(client.entered.wait(), 5)
        second = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        client.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        assert client.calls == 1
        assert manager._cleanup_tasks == set()

    asyncio.run(scenario())


def test_m05_close_then_reload_and_repeated_reloads(tmp_path, monkeypatch):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"remote": {"command": "unused"}}})

    async def scenario():
        manager = McpManager(config)
        connected = []

        async def connect(client, namespace):
            connected.append(client.name)

        monkeypatch.setattr(manager, "_connect_client", connect)

        await manager.reload()
        await manager.reload()
        await manager.close()
        assert manager.clients == {}
        await manager.reload()
        assert connected == ["remote", "remote", "remote"]
        assert "remote" in manager.clients

        await manager.register_extension_server("ext", {"command": "unused"}, "owner")
        # Extension servers are merged before file servers, so the reload connects
        # both again.
        assert connected == ["remote", "remote", "remote", "ext", "remote"]
        assert "ext" in manager.clients

        await manager.close()
        await manager.reload()
        assert "remote" in manager.clients
        assert "ext" in manager.clients

    asyncio.run(scenario())


def test_m05_stale_generation_cannot_register(tmp_path, monkeypatch):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"remote": {"command": "unused"}}})

    async def scenario():
        manager = McpManager(config)
        manager._ensure_locks()
        connected = []

        async def connect(client, namespace):
            connected.append(client.name)

        monkeypatch.setattr(manager, "_connect_client", connect)
        await manager.reload()
        registered = manager.clients["remote"]
        stale = manager._generation
        manager._generation += 1
        late = McpClient("remote", {"command": "unused"}, tmp_path)
        with pytest.raises(RuntimeError, match="cancelled"):
            await manager._open_client(late, "remote", stale, register_name="remote")
        assert manager.clients["remote"] is registered
        assert late not in manager.clients.values()
        await manager.close()

    asyncio.run(scenario())


def test_m06_close_interrupts_listener_backoff(tmp_path, caplog):
    async def scenario():
        client = McpClient("remote", {"url": "https://example.test/mcp"}, tmp_path)
        requests = []

        async def handler(request):
            requests.append(request)
            raise httpx.ConnectError("down", request=request)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client.pump_task = asyncio.create_task(client._http_listener())
        await wait_for(lambda: bool(requests))
        started = time.monotonic()
        await asyncio.wait_for(client.close(), 0.8)
        assert time.monotonic() - started < 0.8
        assert client.pump_task is None

    asyncio.run(scenario())


def test_m07_one_failed_cleanup_does_not_block_others(tmp_path):
    class Failing:
        async def close(self):
            raise RuntimeError("cleanup failed")

    class Recording:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    async def scenario():
        manager = McpManager(Configuration(str(tmp_path), str(tmp_path / "user")))
        failing, recording = Failing(), Recording()
        manager.clients["bad"] = failing
        manager.clients["good"] = recording
        await asyncio.wait_for(manager.close(), 5)
        assert recording.closed
        assert manager.clients == {}
        assert manager._cleanup_tasks == set()

    asyncio.run(scenario())
