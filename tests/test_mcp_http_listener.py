"""Push streams survive idle periods and transient disconnects."""

import asyncio

import httpx
import pytest

from pi.coding_agent.mcp import McpClient


@pytest.mark.parametrize("failure", ["timeout", "eof", "503"])
def test_listener_reconnects_and_delivers_notifications(tmp_path, caplog, failure):
    async def scenario():
        client = McpClient("remote", {"url": "https://example.test/mcp", "timeout": 7}, tmp_path)
        client.session_id = "session"
        received = asyncio.Event()
        client.on_tools_changed = lambda name: received.set()
        requests = []

        async def handle(request):
            requests.append(request)
            assert request.extensions["timeout"] == {"connect": 7, "read": None, "write": 7, "pool": 7}
            assert request.headers["Mcp-Session-Id"] == "session"
            if len(requests) == 1:
                if failure == "timeout":
                    raise httpx.ReadTimeout("stream interrupted", request=request)
                return httpx.Response(503 if failure == "503" else 200, content=b"")
            return httpx.Response(200, content=b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n')

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        client.pump_task = asyncio.create_task(client._http_listener())
        try:
            await asyncio.wait_for(received.wait(), 4)
            assert client.tools_changed
            assert len(requests) == 2
            assert "remote" in caplog.text
        finally:
            await client.close()
        count = len(requests)
        await asyncio.sleep(0)
        assert len(requests) == count
        assert client.pump_task is None

    asyncio.run(scenario())


def test_optional_get_405_is_not_retried(tmp_path):
    async def scenario():
        client = McpClient("remote", {"url": "https://example.test/mcp"}, tmp_path)
        requests = []

        async def handle(request):
            requests.append(request)
            return httpx.Response(405)

        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            await asyncio.wait_for(client._http_listener(), 1)
            assert len(requests) == 1
        finally:
            await client.close()

    asyncio.run(scenario())
