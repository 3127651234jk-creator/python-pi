import asyncio
import json
import sys

import httpx
import pytest

from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.mcp import McpClient, McpManager


@pytest.mark.parametrize("newline", ["\r", "\n", "\r\n"])
@pytest.mark.parametrize("field", ["command", "args"])
def test_reload_rejects_stdio_newlines_before_connect(tmp_path, monkeypatch, newline, field):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    bad = {"command": "server.cmd", "args": ["safe", "value"]}
    if field == "command":
        bad["command"] = "server.cmd" + newline + "other"
    else:
        bad["args"][1] = "line1" + newline + "line2"
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "bad": bad, "good": {"command": "server.cmd", "args": ["", "safe"]},
    }})

    async def scenario():
        manager = McpManager(config)
        connected = []

        async def connect(client, namespace):
            connected.append(client.name)

        monkeypatch.setattr(manager, "_connect_client", connect)
        try:
            await manager.reload()
            assert manager.errors == {"bad": "stdio command and args must not contain CR or LF"}
            assert connected == ["good"]
            assert list(manager.clients) == ["good"]
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_stdio_discovery_call_and_close(tmp_path):
    script = tmp_path / "server.py"
    script.write_text('''import sys,json
for line in sys.stdin:
    req=json.loads(line)
    if "id" not in req: continue
    method=req["method"]
    result={}
    if method=="initialize": result={"protocolVersion":"2025-03-26"}
    if method=="tools/list": result={"tools":[{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}
    if method=="tools/call": result={"content":[{"type":"text","text":req["params"]["arguments"]["text"]}],"structuredContent":{"ok":True}}
    print(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":result}),flush=True)
''', encoding="utf-8")
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "test": {"command": sys.executable, "args": [str(script)]},
        "bad": {"command": "definitely-nonexistent-mcp-executable"},
    }})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            assert "bad" in manager.errors
            assert len(manager.tools) == 1
            assert manager.tools[0].name.startswith("mcp__test__echo")
            result = await manager.tools[0].execute("call", {"text": "hello"})
            assert result.content[0].text == "hello"
            assert result.structured_content == {"ok": True}
            process = manager.clients["test"].process
        finally:
            await manager.close()
        assert process.returncode is not None
    asyncio.run(check())


def test_invalid_server_config_is_reported_not_raised(tmp_path, monkeypatch):
    """One bad server must be skipped; the rest of the reload keeps going."""
    monkeypatch.delenv("PI_M5_MISSING", raising=False)
    script = tmp_path / "server.py"
    script.write_text('''import sys,json
for line in sys.stdin:
    req=json.loads(line)
    if "id" not in req: continue
    result={"protocolVersion":"2025-03-26"} if req["method"]=="initialize" else {"tools":[]}
    print(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":result}),flush=True)
''', encoding="utf-8")
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "bad-url": {"url": "notaurl"},
        "bad-env": {"command": sys.executable, "env": {"TOKEN": "${PI_M5_MISSING}"}},
        "good": {"command": sys.executable, "args": [str(script)]},
    }})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            assert "HTTP(S) URL" in manager.errors["bad-url"]
            assert "PI_M5_MISSING" in manager.errors["bad-env"]
            assert "good" in manager.clients
            assert "good" not in manager.errors
        finally:
            await manager.close()

    asyncio.run(check())


def test_empty_log_notification_still_writes_a_line(tmp_path):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    config.agent_dir.mkdir(parents=True, exist_ok=True)
    manager = McpManager(config)
    manager._log_message("srv", {"level": "info", "data": ""})
    log = (config.agent_dir / "mcp.log").read_text(encoding="utf-8")
    assert "srv:" in log


def test_failed_reconnect_keeps_a_usable_client(tmp_path, monkeypatch):
    """A failed reconnect must not leave a closed client that raises AttributeError."""

    async def send(self, request, **kwargs):
        body = json.loads(request.content)
        if failing["on"]:
            raise httpx.ConnectError("server down")
        if "id" not in body:
            return httpx.Response(202)
        result = {"protocolVersion": "2025-03-26"} if body["method"] == "initialize" else {"tools": []}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    failing = {"on": False}
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"remote": {"url": "https://example.test/mcp"}}})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            failing["on"] = True
            with pytest.raises(httpx.ConnectError):
                await manager.call("remote", "tools/list", {})
            # Before the fix this raised AttributeError on the closed client.
            with pytest.raises(httpx.ConnectError):
                await manager.call("remote", "tools/list", {})
        finally:
            await manager.close()

    asyncio.run(check())


def test_http_session_and_error_mapping(tmp_path, monkeypatch):
    requests = []

    async def send(self, request, **kwargs):
        body = json.loads(request.content)
        requests.append((body, dict(request.headers)))
        result = {}
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26"}
        elif body["method"] == "tools/list":
            result = {"tools": [{"name": "fail", "inputSchema": {"type": "object"}}]}
        elif body["method"] == "tools/call":
            result = {"content": [{"type": "text", "text": "tool failed"}], "isError": True}
        if "id" not in body:
            return httpx.Response(202)
        return httpx.Response(200, headers={"Mcp-Session-Id": "session"}, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"remote": {"url": "https://example.test/mcp"}}})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            assert not manager.errors
            assert (await manager.tools[0].execute("call", {})).is_error
            assert requests[-1][1]["mcp-session-id"] == "session"
        finally:
            await manager.close()
    asyncio.run(check())

def test_concurrent_reconnects_share_replacement(tmp_path, monkeypatch):
    async def check():
        manager = McpManager(Configuration(str(tmp_path), str(tmp_path / "user")))
        old = McpClient("test", {"command": sys.executable}, tmp_path)
        manager.clients["test"] = old
        replacements = []

        async def connect(client, namespace):
            replacements.append(client)
            await asyncio.sleep(0)

        async def refresh(server):
            pass

        monkeypatch.setattr(manager, "_connect_client", connect)
        monkeypatch.setattr(manager, "ensure_tools", refresh)
        try:
            await asyncio.gather(manager.reconnect("test", old), manager.reconnect("test", old))
            assert len(replacements) == 1
            assert manager.clients["test"] is replacements[0]
            await manager.reconnect("test", old)
            assert len(replacements) == 1
        finally:
            await manager.close()

    asyncio.run(check())

