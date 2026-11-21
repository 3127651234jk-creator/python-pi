import asyncio
import json
import sys
import time

import httpx
import pytest

from pi.coding_agent.cli import _mcp_login_args
from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.mcp import McpClient, McpManager, create_tool_name_matcher, is_mcp_tool_name


async def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.01)


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


@pytest.mark.parametrize("field,value", [
    ("env", None), ("env", []), ("env", {"TOKEN": 1}),
    ("headers", None), ("headers", []), ("headers", {"X-Test": 1}),
])
def test_reload_rejects_non_string_env_and_headers_without_aborting(tmp_path, monkeypatch, field, value):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "bad": {"command": "bad", field: value},
        "good": {"command": "good"},
    }})

    async def scenario():
        manager = McpManager(config)
        connected = []

        async def connect(client, namespace):
            connected.append(client.name)

        monkeypatch.setattr(manager, "_connect_client", connect)
        try:
            await manager.reload()
            assert manager.errors["bad"] == f"{field} must map strings to strings"
            assert connected == ["good"]
        finally:
            await manager.close()

    asyncio.run(scenario())


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


# -- v1.0.4 port: tool name matching (04b97ef00) and closing in-flight connects (8c911797c)


def test_tool_name_matcher_exact_names_and_star_patterns():
    matcher = create_tool_name_matcher(["read", "mcp__radius__*"])
    assert matcher("read")
    assert not matcher("write")
    assert matcher("mcp__radius__search")
    assert matcher("mcp__radius__deep__nested")
    assert not matcher("mcp__other__search")

    pattern_only = create_tool_name_matcher(["*_search"])
    assert pattern_only("mcp__radius__search")
    assert not pattern_only("mcp__radius__index")


def test_is_mcp_tool_name_covers_server_and_resource_tools():
    assert is_mcp_tool_name("mcp__dev__echo")
    assert is_mcp_tool_name("list_mcp_resources")
    assert is_mcp_tool_name("list_mcp_resource_templates")
    assert is_mcp_tool_name("read_mcp_resource")
    assert not is_mcp_tool_name("read")
    assert not is_mcp_tool_name("codemode")


def write_slow_initialize_server(tmp_path):
    """Stdio MCP server that accepts the initialize request but never answers it."""
    script = tmp_path / "slow_server.py"
    script.write_text(
        "import sys,json,time\n"
        "line = sys.stdin.readline()\n"
        "req = json.loads(line)\n"
        "time.sleep(60)\n"
        'print(json.dumps({"jsonrpc":"2.0","id":req["id"],"result":{"protocolVersion":"2025-03-26"}}),flush=True)\n',
        encoding="utf-8")
    return {"command": sys.executable, "args": [str(script)], "timeout": 30}


def test_close_aborts_connect_in_progress(tmp_path):
    config = write_slow_initialize_server(tmp_path)

    async def scenario():
        client = McpClient("slow", config, tmp_path)
        task = asyncio.create_task(client.connect())
        while client.process is None and not task.done():
            await asyncio.sleep(0.02)
        process = client.process
        assert process is not None
        # close() while initialize is in flight: the connect settles with an error
        # and the spawned server is gone instead of running until its timeout.
        await client.close()
        with pytest.raises(RuntimeError):
            await task
        assert client.closed
        assert client.http is None and client.process is None
        assert process.returncode is not None

    asyncio.run(scenario())


def test_cancelled_reload_closes_half_open_client(tmp_path, monkeypatch):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json",
               {"mcpServers": {"slow": write_slow_initialize_server(tmp_path)}})

    created = []

    class RecordingClient(McpClient):
        def __init__(self, name, client_config, cwd):
            super().__init__(name, client_config, cwd)
            created.append(self)

    monkeypatch.setattr("pi.coding_agent.mcp.McpClient", RecordingClient)

    async def scenario():
        manager = McpManager(config)
        task = asyncio.create_task(manager.reload())
        while not created and not task.done():
            await asyncio.sleep(0.02)
        while created[0].process is None and not task.done():
            await asyncio.sleep(0.02)
        process = created[0].process
        assert process is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The reload's cleanup closed the client it had half opened.
        assert created[0].closed
        assert created[0].process is None
        assert process.returncode is not None
        assert manager.clients == {}

    asyncio.run(scenario())


def test_close_is_idempotent_after_failed_connect(tmp_path):
    async def scenario():
        client = McpClient("bad", {"command": "definitely-nonexistent-mcp-executable"}, tmp_path)
        with pytest.raises(Exception):
            await client.connect()
        await client.close()
        await client.close()
        assert client.closed

    asyncio.run(scenario())


# -- B-CA-04: reload can connect servers in the background so startup is not blocked


def write_remote_server_config(tmp_path):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json",
               {"mcpServers": {"remote": {"url": "https://example.test/mcp"}}})
    return config


def test_background_reload_does_not_block_startup(tmp_path, monkeypatch):
    async def scenario():
        gate = asyncio.Event()  # the server does not answer initialize until released
        manager = McpManager(write_remote_server_config(tmp_path))
        transitions = []
        changed = []

        async def send(self, request, **kwargs):
            body = json.loads(request.content)
            if body.get("method") == "initialize":
                await gate.wait()
            if "id" not in body:
                return httpx.Response(202)
            result = {"protocolVersion": "2025-03-26"} if body["method"] == "initialize" else {
                "tools": [{"name": "echo", "description": "Echo",
                           "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        monkeypatch.setattr(httpx.AsyncClient, "send", send)
        manager.on_state_changed = lambda server, state, error=None: transitions.append((server, state))
        manager.on_tools_changed = lambda server: changed.append(server)
        try:
            await asyncio.wait_for(manager.reload(wait=False), 5)
            # The reload returned while the connect is still waiting on the server.
            assert manager.clients == {}
            assert "remote" not in manager.errors
            assert manager.server_states().get("remote") == "connecting"
            gate.set()
            await wait_until(lambda: "remote" in manager.clients)
            assert manager.server_states()["remote"] == "connected, 1 tools"
            assert transitions == [("remote", "connecting"), ("remote", "connected")]
            assert changed == ["remote"]
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_background_connect_does_not_register_into_a_closed_generation(tmp_path, monkeypatch):
    async def scenario():
        gate = asyncio.Event()
        manager = McpManager(write_remote_server_config(tmp_path))

        async def send(self, request, **kwargs):
            body = json.loads(request.content)
            if body.get("method") == "initialize":
                await gate.wait()
            if "id" not in body:
                return httpx.Response(202)
            result = {"protocolVersion": "2025-03-26"} if body["method"] == "initialize" else {"tools": []}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        monkeypatch.setattr(httpx.AsyncClient, "send", send)
        await asyncio.wait_for(manager.reload(wait=False), 5)
        # close() invalidates the generation and waits for the connect to settle.
        close_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0.05)
        assert not close_task.done()
        gate.set()
        await asyncio.wait_for(close_task, 5)
        assert manager.clients == {}
        assert manager.tools == []
        assert not any(state.startswith("connected") for state in manager.server_states().values())

    asyncio.run(scenario())


def test_server_states_reports_disabled_and_invalid_servers(tmp_path):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "off": {"command": "unused", "enabled": False},
        "bad": {"url": "notaurl"},
    }})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            states = manager.server_states()
            assert states["off"] == "disabled"
            assert "HTTP(S) URL" in states["bad"]
        finally:
            await manager.close()

    asyncio.run(check())


def test_tool_exposure_overrides_server_exposure(tmp_path, monkeypatch):
    """toolExposure maps raw tool names/patterns over the server exposure (F-CA-09)."""
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "dev": {"command": "dev", "exposure": "direct",
                "toolExposure": {"search": "deferred", "secret_*": "codemode"}},
    }})

    async def scenario():
        manager = McpManager(config)

        async def open_client(client, namespace, generation, register_name=None):
            client.tools = [{"name": "search"}, {"name": "other"}, {"name": "secret_key"}]
            async with manager._state_lock:
                manager.clients[register_name] = client

        monkeypatch.setattr(manager, "_open_client", open_client)
        try:
            await manager.reload()
            by_raw = {tool.name.split("__")[2].rsplit("_", 1)[0]: tool for tool in manager.tools}
            assert by_raw["search"].exposure == "deferred" and not by_raw["search"].declared
            assert by_raw["other"].exposure == "direct" and by_raw["other"].declared
            assert by_raw["secret_key"].exposure == "codemode" and not by_raw["secret_key"].declared
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_tool_exposure_invalid_mapping_is_rejected(tmp_path):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "bad-list": {"command": "x", "toolExposure": ["search"]},
        "bad-value": {"command": "x", "toolExposure": {"search": "shown"}},
        "good": {"command": "x", "toolExposure": {"search": "direct"}},
    }})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            assert manager.errors["bad-list"] == "toolExposure must map tool names to direct, deferred, or codemode"
            assert manager.errors["bad-value"] == "toolExposure must map tool names to direct, deferred, or codemode"
            assert "toolExposure" not in manager.errors.get("good", "")
        finally:
            await manager.close()

    asyncio.run(check())


def test_project_mcp_json_override_rules(tmp_path):
    """A trusted project's .pi/mcp.json may only override enabled/exposure/toolExposure (F-CA-09)."""
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "dev": {"command": "dev"},
        "dev2": {"command": "dev2"},
    }})
    write_json(config.cwd / ".pi" / "mcp.json", {"mcpServers": {
        "dev": {"exposure": "deferred"},
        "ghost": {"enabled": True},
        "dev2": {"timeout": 30},
    }})

    async def check():
        manager = McpManager(config)
        try:
            await manager.reload()
            assert "or a global server to override" in manager.errors["ghost"]
            assert "an override can only set enabled, exposure, toolExposure" in manager.errors["dev2"]
            assert "override" not in manager.errors.get("dev", "")
        finally:
            await manager.close()

    asyncio.run(check())


def test_oauth_login_timeout_limits_the_whole_flow(tmp_path, monkeypatch):
    """--timeout caps discovery, browser wait, and token exchange together (F-CA-07)."""
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"dev": {"url": "https://example.test/mcp"}}})

    async def scenario():
        manager = McpManager(config)

        async def slow_login(server, signal):
            await asyncio.sleep(30)

        monkeypatch.setattr(manager, "_oauth_login", slow_login)
        try:
            with pytest.raises(ValueError, match="timed out after"):
                await asyncio.wait_for(manager.oauth_login("dev", timeout=0.05), 5)
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_oauth_login_without_timeout_keeps_waiting(tmp_path, monkeypatch):
    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {"dev": {"url": "https://example.test/mcp"}}})

    async def scenario():
        manager = McpManager(config)

        async def quick_login(server, signal):
            return

        monkeypatch.setattr(manager, "_oauth_login", quick_login)
        try:
            await asyncio.wait_for(manager.oauth_login("dev"), 5)
        finally:
            await manager.close()

    asyncio.run(scenario())


def test_mcp_cli_login_arg_parsing():
    from pi.coding_agent.mcp_cli import _parse_login_args

    assert _parse_login_args(["dev"]) == ("dev", None)
    assert _parse_login_args(["dev", "--timeout", "30"]) == ("dev", 30.0)
    assert _parse_login_args(["--timeout", "10", "dev"]) == ("dev", 10.0)
    with pytest.raises(ValueError, match="--timeout requires"):
        _parse_login_args(["dev", "--timeout"])
    with pytest.raises(ValueError, match="usage"):
        _parse_login_args([])


def test_interactive_mcp_login_reuses_the_cli_parser():
    """`/mcp login` must resolve --timeout through the shared parser, not a
    missing helper (the command used to fail with NameError)."""

    assert _mcp_login_args(["dev"]) == ("dev", None)
    assert _mcp_login_args(["dev", "--timeout", "30"]) == ("dev", 30.0)
    with pytest.raises(ValueError, match="usage"):
        _mcp_login_args([])


def test_mcp_cli_list_and_login_routing(tmp_path, monkeypatch, capsys):
    from pi.coding_agent import mcp_cli

    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    write_json(config.agent_dir / "mcp.json", {"mcpServers": {
        "remote": {"url": "https://example.test/mcp"},
        "local": {"command": "dev"},
    }})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    # Neither server is reachable here, so the listing reports them failed.
    assert mcp_cli.main(["list", "--json"]) == 1
    parsed = json.loads(capsys.readouterr().out)
    states = {server["name"]: server["state"] for server in parsed["servers"]}
    assert states["remote"] == "failed" and states["local"] == "failed"

    # Unknown server names fail with a clear error before any browser opens.
    assert mcp_cli.main(["login", "ghost", "--timeout", "1"]) == 1
    assert "Unknown MCP server: ghost" in capsys.readouterr().err

