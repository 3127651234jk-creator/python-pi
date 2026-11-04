"""TS extension bridge tests: offline dispatch/protocol tests plus real-node e2e."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.agent.agent import Agent
from pi.agent.types import BeforeToolCallResult
from pi.ai.models import resolve_model
from pi.coding_agent.extensions import ExtensionAPI, ExtensionContext
from pi.coding_agent.ts_extension import TsExtensionBridge


def make_api(tmp_path) -> ExtensionAPI:
    agent = Agent(resolve_model("faux"))
    return ExtensionAPI(tmp_path / "demo.ts", ExtensionContext(agent, tmp_path))


def make_bridge(tmp_path) -> TsExtensionBridge:
    """Bridge without a live process; replies to child requests are recorded."""
    bridge = TsExtensionBridge(tmp_path / "demo.ts", make_api(tmp_path), node_argv=["unused"])
    sent = []

    async def fake_reply(request, result=None, error=None):
        sent.append({"id": request.get("id"), "result": result, "error": error})

    bridge._reply = fake_reply  # type: ignore[method-assign]
    bridge.sent = sent
    return bridge


def test_dispatch_registers_tool_command_provider_and_event(tmp_path):
    async def check():
        bridge = make_bridge(tmp_path)
        await bridge.dispatch({"id": 1, "method": "registerTool", "params": {
            "name": "echo", "description": "Echo", "handlerId": "t1",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}})
        assert [tool.name for tool in bridge.api.tools] == ["echo"]
        assert bridge.tool_handlers["echo"] == "t1"

        await bridge.dispatch({"id": 2, "method": "registerCommand", "params": {
            "name": "hello", "description": "Greeting", "handlerId": "c1"}})
        assert "hello" in bridge.api.commands

        await bridge.dispatch({"id": 3, "method": "registerProvider", "params": {
            "name": "acme", "config": {"api": "openai-completions", "base_url": "https://x"}}})
        assert "acme" in bridge.api.providers

        await bridge.dispatch({"id": 4, "method": "on", "params": {"event": "tool_call", "handlerId": "h1"}})
        assert bridge.event_handlers["h1"] == "tool_call"
        assert bridge.api.handlers["tool_call"]

        await bridge.dispatch({"id": 5, "method": "registerKeybinding", "params": {
            "action": "refresh", "keys": ["ctrl+r"], "handlerId": "k1"}})
        assert bridge.api.keybindings[0][:2] == ("refresh", ["ctrl+r"])

        assert bridge.sent == [
            {"id": 1, "result": {}, "error": None},
            {"id": 2, "result": {}, "error": None},
            {"id": 3, "result": {}, "error": None},
            {"id": 4, "result": {}, "error": None},
            {"id": 5, "result": {}, "error": None},
        ]
    asyncio.run(check())


def test_dispatch_unknown_method_and_invalid_registration_reply_errors(tmp_path):
    async def check():
        bridge = make_bridge(tmp_path)
        await bridge.dispatch({"id": 1, "method": "timeTravel", "params": {}})
        assert bridge.sent[-1]["error"] == "unsupported bridge method: timeTravel"

        # registerTool without a name: ExtensionAPI validation fails, error goes to the child.
        await bridge.dispatch({"id": 2, "method": "registerTool", "params": {"handlerId": "t1"}})
        assert bridge.sent[-1]["error"] is not None

        await bridge.dispatch({"id": 3, "method": "ui", "params": {"call": "nope"}})
        assert "unsupported ui call" in bridge.sent[-1]["error"]
    asyncio.run(check())


def test_event_handler_result_mapping(tmp_path):
    async def check():
        bridge = make_bridge(tmp_path)
        canned = {"value": {"result": {}}}

        async def fake_request(method, params, timeout=None):
            return canned["value"]

        bridge._request = fake_request  # type: ignore[method-assign]

        await bridge.dispatch({"id": 1, "method": "on", "params": {"event": "tool_call", "handlerId": "h1"}})
        await bridge.dispatch({"id": 2, "method": "on", "params": {"event": "input", "handlerId": "h2"}})
        await bridge.dispatch({"id": 3, "method": "on", "params": {"event": "session_start", "handlerId": "h3"}})
        tool_call = bridge.api.handlers["tool_call"][0]
        input_ = bridge.api.handlers["input"][0]
        other = bridge.api.handlers["session_start"][0]

        canned["value"] = {"result": {"block": True, "reason": "nope"}}
        outcome = await tool_call({"tool_name": "write"}, None)
        assert isinstance(outcome, BeforeToolCallResult) and outcome.block and outcome.reason == "nope"

        canned["value"] = {"result": None}
        assert await tool_call({"tool_name": "read"}, None) is None

        canned["value"] = {"result": {"action": "transform", "text": "new"}}
        outcome = await input_({"text": "original"}, None)
        assert outcome is not None and outcome.text == "new"

        canned["value"] = {"error": "handler exploded"}
        assert await input_({"text": "x"}, None) is None

        canned["value"] = {"result": {"custom": 1}}
        assert await other({"kind": "session_start"}, None) == {"custom": 1}
    asyncio.run(check())


class FakeStdin:
    def __init__(self):
        self.frames = []

    def write(self, data):
        self.frames.append(data.decode())

    async def drain(self):
        return None


class FakeProcess:
    def __init__(self, stdout):
        self.stdin = FakeStdin()
        self.stdout = stdout
        self.returncode = None

    def kill(self):
        self.returncode = 0

    async def wait(self):
        return 0


def _fake_exec(process):
    async def factory(*args, **kwargs):
        return process
    return factory


def test_request_framing_and_error_propagation(tmp_path, monkeypatch):
    """start() frames the request on stdin; child errors and EOF reject it."""
    async def check():
        # Child answers the start request with an error reply.
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"id":1,"error":"boom"}\n')
        reader.feed_eof()
        process = FakeProcess(reader)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec(process))
        bridge = TsExtensionBridge(tmp_path / "demo.ts", make_api(tmp_path), node_argv=["node"])
        with pytest.raises(RuntimeError, match="boom"):
            await bridge.start()
        assert json.loads(process.stdin.frames[0]) == {
            "id": 1, "method": "start", "params": {"path": str(tmp_path / "demo.ts")}}
        await bridge.close()

        # Child exits without answering: the pending start fails on EOF.
        idle_reader = asyncio.StreamReader()
        idle_reader.feed_eof()
        idle = FakeProcess(idle_reader)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec(idle))
        bridge = TsExtensionBridge(tmp_path / "demo.ts", make_api(tmp_path), node_argv=["node"])
        with pytest.raises(RuntimeError, match="process exited"):
            await bridge.start()

        # A call the child never answers times out into a clear error.
        with pytest.raises(RuntimeError, match="timed out"):
            await bridge._request("callTool", {"handlerId": "t1", "toolCallId": "c", "params": {}}, timeout=0.05)
        await bridge.close()
    asyncio.run(check())


NODE_AVAILABLE = shutil.which("node") is not None

FAIL_LOUD_EXTENSION = "export async function setup(pi) { pi.exec(); }\n"

TOOL_EXTENSION = '''
export async function setup(pi) {
	pi.registerTool({
		name: "upper",
		description: "Uppercase text",
		parameters: { type: "object", properties: { text: { type: "string" } } },
		async execute(callId, params) {
			return { content: [{ type: "text", text: String(params.text).toUpperCase() }] };
		},
	});
}
'''


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_node_fail_loud_stub(tmp_path):
    async def check():
        path = tmp_path / "fails.ts.mjs"
        path.write_text(FAIL_LOUD_EXTENSION, encoding="utf-8")
        from pi.coding_agent.ts_extension import load_ts_extension

        with pytest.raises(RuntimeError, match="exec is not available when the extension runs under the Python host"):
            await load_ts_extension(path, make_api(tmp_path))
    asyncio.run(check())


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_node_tool_roundtrip(tmp_path):
    async def check():
        path = tmp_path / "tools.ts.mjs"
        path.write_text(TOOL_EXTENSION, encoding="utf-8")
        from pi.coding_agent.ts_extension import load_ts_extension

        api = make_api(tmp_path)
        bridge = await load_ts_extension(path, api)
        try:
            assert [tool.name for tool in api.tools] == ["upper"]
            result = await api.tools[0].execute("call-1", {"text": "hi"})
            assert result.content[0].text == "HI"
        finally:
            await bridge.close()
    asyncio.run(check())
