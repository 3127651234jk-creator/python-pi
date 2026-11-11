"""MCP tool refresh must publish new declarations without losing working tools."""

import asyncio

import pytest

from pi.coding_agent.config import Configuration
from pi.coding_agent.mcp import McpClient, McpManager, _maybe_call


def test_failed_discovery_preserves_registered_tools(tmp_path):
    async def scenario():
        manager = McpManager(Configuration(str(tmp_path), str(tmp_path / "user")))
        client = McpClient("remote", {}, tmp_path)
        client.tools = [{"name": "original"}]
        manager.clients["remote"] = client
        manager._register_server_tools("remote", client)
        original = list(manager.tools)
        client.tools_changed = True

        async def request(method, params):
            raise RuntimeError("discovery unavailable")

        client.request = request
        with pytest.raises(RuntimeError, match="discovery unavailable"):
            await manager.ensure_tools("remote")
        assert manager.tools == original
        assert client.tools_changed

    asyncio.run(scenario())


@pytest.mark.parametrize("new_tools", [
    [{"name": "replacement"}],
    [{"name": "original", "inputSchema": {"type": "object", "required": ["path"]}}],
    [{"name": "original"}, {"name": "extra"}],
    [],
])
def test_refresh_notifies_cli_with_server_name(tmp_path, new_tools):
    async def scenario():
        manager = McpManager(Configuration(str(tmp_path), str(tmp_path / "user")))
        client = McpClient("remote", {}, tmp_path)
        client.tools = [{"name": "original"}]
        manager.clients["remote"] = client
        manager._register_server_tools("remote", client)
        client.tools_changed = True
        notifications = []
        manager.on_tools_changed = lambda server: notifications.append(server)

        async def request(method, params):
            return {"tools": new_tools}

        client.request = request
        await manager.ensure_tools("remote")
        assert notifications == ["remote"]
        assert len(manager.tools) == len(new_tools)

    asyncio.run(scenario())


def test_callback_future_is_awaited():
    async def scenario():
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(future.set_result, None)
        await _maybe_call(lambda: future)
        assert future.done()

    asyncio.run(scenario())
