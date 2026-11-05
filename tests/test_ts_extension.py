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
from pi.ai.models import ModelError, resolve_model
from pi.coding_agent.extensions import ExtensionAPI, ExtensionContext
from pi.coding_agent.cli import build_parser, split_extension_flags
from pi.coding_agent.config import Configuration
from pi.coding_agent.extensions import ExtensionManager
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.ts_extension import TsExtensionBridge, load_ts_extension


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


def test_extension_cli_flag_preparse_matches_ts_rules():
    filtered, values = split_extension_flags(
        ["--provider", "faux", "--plan", "--profile=fast", "prompt"], build_parser())
    assert filtered == ["--provider", "faux", "prompt"]
    assert values == {"plan": True, "profile": "fast"}

    filtered, values = split_extension_flags(["--label", "value", "--", "--literal"], build_parser())
    assert filtered == ["--", "--literal"]
    assert values == {"label": "value"}


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
            "id": 1, "method": "start", "params": {
                "path": str(tmp_path / "demo.ts"), "context": bridge._context_payload()}}
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
        bridge = TsExtensionBridge(tmp_path / "demo.ts", make_api(tmp_path), node_argv=["node"])
        bridge.process = FakeProcess(asyncio.StreamReader())
        with pytest.raises(RuntimeError, match="timed out"):
            await bridge._request("callTool", {"handlerId": "t1", "toolCallId": "c", "params": {}}, timeout=0.05)
        await bridge.close()
    asyncio.run(check())


NODE_AVAILABLE = shutil.which("node") is not None


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
def test_cross_extension_bus_roundtrip_and_cleanup(tmp_path):
    async def check():
        context = ExtensionContext(Agent(resolve_model("faux")), tmp_path)
        receiver = tmp_path / "receiver.mjs"
        receiver.write_text('''export function setup(pi) {
            let count = 0, once = 0;
            pi.events.once("ping", () => { once++; });
            const removed = () => { throw Error("removed handler ran"); };
            pi.events.on("ping", removed);
            pi.events.off("ping", removed);
            const unsubscribe = pi.events.on("ping", () => { count += 100; });
            unsubscribe();
            pi.events.on("ping", () => { throw Error("isolated failure"); });
            pi.events.on("ping", async data => {
                await pi.log("received ping");
                count++;
                pi.events.emit("pong", { count, once, data });
            });
        }''', encoding="utf-8")
        sender = tmp_path / "sender.mjs"
        sender.write_text('''export function setup(pi) {
            pi.registerCommand("ping", { handler: args => new Promise(resolve => {
                pi.events.once("pong", data => resolve(JSON.stringify(data)));
                pi.events.emit("ping", { text: args });
            }) });
        }''', encoding="utf-8")
        first_api = ExtensionAPI(receiver, context)
        second_api = ExtensionAPI(sender, context)
        first = await load_ts_extension(receiver, first_api)
        second = None
        try:
            second = await load_ts_extension(sender, second_api)
            for count in (1, 2):
                result = await asyncio.wait_for(second_api.commands["ping"][0]("hello"), 5)
                assert json.loads(result) == {"count": count, "once": 1, "data": {"text": "hello"}}
            await first.close()
            assert len(context.event_receivers) == 1
            first = await load_ts_extension(receiver, ExtensionAPI(receiver, context))
            result = await asyncio.wait_for(second_api.commands["ping"][0]("reloaded"), 5)
            assert json.loads(result) == {"count": 1, "once": 1, "data": {"text": "reloaded"}}
        finally:
            await first.close()
            if second is not None:
                await second.close()
        assert context.event_receivers == []
    asyncio.run(check())

REFRESH_EXTENSION = "export async function setup(pi) { pi.refreshTools(); }\n"

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
def test_node_refresh_tools(tmp_path):
    async def check():
        path = tmp_path / "refresh.ts.mjs"
        path.write_text(REFRESH_EXTENSION, encoding="utf-8")
        from pi.coding_agent.ts_extension import load_ts_extension

        bridge = await load_ts_extension(path, make_api(tmp_path))
        await bridge.close()
    asyncio.run(check())


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
def test_node_renderers_transformer_virtual_model_and_provider_removal(tmp_path):
    async def check():
        path = tmp_path / "compat.mjs"
        path.write_text('''export async function setup(pi) {
            pi.registerFlag("profile", {type:"string", default:"default"});
            pi.registerCommand("profile", {handler: () => pi.getFlag("profile")});
            pi.registerMarkdownTransformer((text, ctx) => `${ctx.messageType}:${text}`);
            pi.registerMessageRenderer("notice", message => ({render: width => [`message:${message.content}:${width}`]}));
            pi.registerEntryRenderer("state", entry => ({render: () => [`entry:${entry.data.value}`]}));
            pi.registerToolRenderer(name => name === "echo" ? {
                renderCall: args => ({render: () => [`call:${args.text}`]}),
                renderResult: result => ({render: () => [`result:${result.content[0].text}`]})
            } : undefined);
            pi.registerProvider("temporary", {api:"faux", models:[{id:"one"}]});
            pi.unregisterProvider("temporary");
            pi.registerVirtualModel({provider:"routes", id:"fast", name:"Fast", async route(request) {
                return {model:{provider:"faux", id:"faux"}, thinkingLevel:"off", state:{count:(request.state?.count ?? 0)+1}};
            }});
        }''', encoding="utf-8")
        api = make_api(tmp_path)
        bridge = await load_ts_extension(path, api)
        try:
            await bridge.set_flag_values({"profile": "cli"})
            assert api.flags["profile"]["default"] == "default"
            assert await api.commands["profile"][0]("") == "cli"
            assert "temporary" not in api.providers
            assert await api.markdown_transformers[0]("hello", {"messageType": "assistant"}) == "assistant:hello"
            assert await api.message_renderers["notice"]({"content": "ok"}, {}, 72) == ["message:ok:72"]
            assert await api.entry_renderers["state"]({"data": {"value": 3}}, {}, 80) == ["entry:3"]
            assert await api.tool_renderers[0]("echo", "call", {"text": "x"}, {}, 80) == ["call:x"]
            route = api.virtual_models[("routes", "fast")]["route"]
            routed = await route({"state": {"count": 2}})
            assert routed["model"] == {"provider": "faux", "id": "faux"}
            assert routed["state"] == {"count": 3}
        finally:
            await bridge.close()
    asyncio.run(check())


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
def test_extension_manager_applies_cli_flags_after_setup(tmp_path):
    async def check():
        extension_dir = tmp_path / ".pi" / "extensions"
        extension_dir.mkdir(parents=True)
        (extension_dir / "flags.mjs").write_text('''export function setup(pi) {
            pi.registerFlag("profile", {type:"string", default:"default"});
            pi.registerCommand("profile", {handler: () => pi.getFlag("profile")});
        }''', encoding="utf-8")
        config = Configuration(str(tmp_path), str(tmp_path / "agent"))
        manager = ExtensionManager(config, Agent(resolve_model("faux")), flag_values={"profile": "fast"})
        try:
            await manager.reload()
            assert await manager.commands["profile"][0]("") == "fast"
        finally:
            await manager.close()
    asyncio.run(check())


def test_dispatch_dynamic_mcp_registration(tmp_path):
    class FakeMcp:
        def __init__(self):
            self.servers = {}

        async def register_extension_server(self, name, config, owner):
            self.servers[name] = (config, owner)

        async def unregister_extension_server(self, name, owner):
            self.servers.pop(name, None)

        def registered_extension_servers(self):
            return [{"name": name, "config": config, "extensionPath": owner}
                    for name, (config, owner) in self.servers.items()]

    async def check():
        bridge = make_bridge(tmp_path)
        bridge.api.context.mcp = FakeMcp()
        await bridge.dispatch({"id": 1, "method": "registerMcpServer", "params": {
            "name": "docs", "config": {"url": "https://example.test/mcp"}}})
        assert bridge.api.mcp_servers == {"docs": {"url": "https://example.test/mcp"}}
        assert bridge._context_payload()["mcpServers"][0]["name"] == "docs"
        await bridge.dispatch({"id": 2, "method": "unregisterMcpServer", "params": {"name": "docs"}})
        assert bridge.api.mcp_servers == {}
    asyncio.run(check())


def test_virtual_model_routes_agent_requests(tmp_path):
    async def check():
        extension_dir = tmp_path / ".pi" / "extensions"
        extension_dir.mkdir(parents=True)
        (extension_dir / "route.py").write_text('''async def route(request):
    return {"model": {"provider": "faux", "id": "faux"}, "thinkingLevel": "off", "state": {"used": True}}

def setup(api):
    api.register_virtual_model({"provider": "routes", "id": "fast", "name": "Fast", "route": route})
''', encoding="utf-8")
        config = Configuration(str(tmp_path), str(tmp_path / "agent"))
        registry = ModelRegistry(config)
        agent = Agent(resolve_model("faux"))
        manager = ExtensionManager(config, agent, registry=registry)
        try:
            await manager.reload()
            agent.model = registry.resolve("routes", "fast")
            messages = await agent.prompt("hello")
            assistants = [message for message in messages if getattr(message, "role", None) == "assistant"]
            assert assistants and assistants[-1].provider == "faux"
        finally:
            await manager.close()
        with pytest.raises(ModelError, match="Unknown provider"):
            registry.resolve("routes", "fast")
    asyncio.run(check())


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
def test_node_control_interfaces(tmp_path):
    async def check():
        path = tmp_path / "controls.mjs"
        path.write_text('''export async function setup(pi) {
            pi.registerFlag("verbose", {type:"boolean", default:true});
            if (pi.getFlag("verbose") !== true) throw Error("flag missing");
            pi.registerShortcut("ctrl+r", {handler: ctx => {}});
            pi.registerCommand("controls", {handler: async () => {
                const tools = pi.getActiveTools();
                await pi.setActiveTools([]);
                if (pi.getActiveTools().length) throw Error("tools not disabled");
                await pi.setActiveTools(tools);
                const settings = pi.getSettings(); settings.changed = true;
                if (pi.getSettings().changed) throw Error("settings leaked");
                if (!pi.getCommands().some(c => c.name === "controls")) throw Error("commands missing");
                if (!await pi.setModel({provider:"faux", id:"faux"})) throw Error("model failed");
                const result = await pi.exec(process.execPath, ["-e", "process.stdout.write('ok')"]);
                if (result.stdout !== "ok" || result.code !== 0) throw Error("exec failed");
                return "done";
            }});
        }''', encoding="utf-8")
        api = make_api(tmp_path)
        bridge = await load_ts_extension(path, api)
        try:
            assert api.keybindings[0][1] == ["c-r"]
            assert await asyncio.wait_for(api.commands["controls"][0](""), 10) == "done"
        finally:
            await bridge.close()
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


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node unavailable")
def test_node_state_requests_with_colliding_ids(tmp_path):
    async def check():
        path = tmp_path / "state.mjs"
        path.write_text('''export async function setup(pi) {
            await pi.setThinkingLevel("high");
            if (pi.getThinkingLevel() !== "high") throw Error("stale thinking");
            if (pi.getSessionName() !== undefined) throw Error("unexpected session");
            pi.registerCommand("state", {handler: () => JSON.stringify({
                thinking: pi.getThinkingLevel(), tools: pi.getActiveTools()
            })});
        }''', encoding="utf-8")
        from pi.coding_agent.ts_extension import load_ts_extension

        api = make_api(tmp_path)
        bridge = await asyncio.wait_for(load_ts_extension(path, api), 10)
        try:
            assert api.context.agent.thinking_level == "high"
            result = await api.commands["state"][0]("")
            assert json.loads(result) == {"thinking": "high", "tools": []}
        finally:
            await bridge.close()
    asyncio.run(check())
