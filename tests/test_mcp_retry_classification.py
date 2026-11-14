import asyncio
from unittest.mock import AsyncMock

import pytest

from pi.coding_agent.config import Configuration
from pi.coding_agent.mcp import McpClient, McpManager


@pytest.mark.parametrize("message", [
    "storage disconnected mid-txn", "MCP server disconnected", "MCP HTTP 503",
])
def test_server_error_does_not_reconnect_or_replay(tmp_path, message):
    async def scenario():
        client = McpClient("test", {}, tmp_path)
        calls = 0

        async def send(request):
            nonlocal calls
            calls += 1
            return {"jsonrpc": "2.0", "id": request["id"],
                    "error": {"code": -32000, "message": message}}

        client.send = send
        manager = McpManager(Configuration(tmp_path, tmp_path / "agent"))
        manager.clients["test"] = client
        manager.reconnect = AsyncMock()
        with pytest.raises(RuntimeError, match="MCP request failed"):
            await manager.call("test", "tools/call", {"name": "write", "arguments": {}})
        assert calls == 1
        manager.reconnect.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("error", [RuntimeError("MCP server disconnected"),
                                       ConnectionError("MCP stream disconnected before a response")])
def test_transport_disconnect_still_retries_once(tmp_path, error):
    async def scenario():
        manager = McpManager(Configuration(tmp_path, tmp_path / "agent"))
        client = McpClient("test", {}, tmp_path)
        client.request = AsyncMock(side_effect=[error, {"ok": True}])
        manager.clients["test"] = client
        manager.reconnect = AsyncMock()
        assert await manager.call("test", "tools/call", {}) == {"ok": True}
        assert client.request.await_count == 2
        manager.reconnect.assert_awaited_once_with("test", client)

    asyncio.run(scenario())
