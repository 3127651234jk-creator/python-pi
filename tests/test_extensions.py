import asyncio
import io
import json

import pytest

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.ai.providers import PROVIDER_STREAM_FNS
from pi.ai.providers.faux import reset_script_index
from pi.coding_agent.config import Configuration
from pi.coding_agent.extensions import ExtensionManager
from pi.coding_agent.tools import create_default_tools
from pi.coding_agent.rpc import RpcServer, JsonWriter


SOURCE = '''
from pi.agent.types import AgentTool, AgentToolResult, BeforeToolCallResult
from pi.ai.types import TextContent
from pi.coding_agent.extensions import InputResult

def setup(api):
    print("extension loading log")
    async def echo(call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=params["text"])])
    api.register_tool(AgentTool(name="echo", description="Echo", parameters={"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}, execute=echo))
    api.register_command("hello", lambda args, ctx: "Hello " + args, "Greeting")
    api.register_command("handled", lambda args, ctx: None)
    api.on("input", lambda event, ctx: InputResult(text=event["text"].replace("original", "transformed")))
    api.on("tool_call", lambda event, ctx: BeforeToolCallResult(block=event.tool_name == "write", reason="Blocked by extension"))
    api.on("session_start", lambda event, ctx: api.log("started"))
    api.on("session_shutdown", lambda event, ctx: (ctx.cwd / "closed.txt").write_text("closed"))
'''


def test_factory_tools_commands_reload_cleanup(tmp_path, capsys):
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        directory = tmp_path / ".pi/extensions"
        directory.mkdir(parents=True)
        path = directory / "demo.py"
        path.write_text(SOURCE, encoding="utf-8")
        agent = Agent(resolve_model("faux"))
        manager = ExtensionManager(config, agent)
        await manager.reload()
        assert await manager.process_input("/hello world") == "Hello world"
        assert await manager.process_input("original") == "transformed"
        assert await manager.process_input("/handled") is None
        assert len(agent.tools) == 1
        path.write_text(SOURCE.replace('"Hello "', '"Updated "'), encoding="utf-8")
        await manager.reload()
        assert await manager.process_input("/hello world") == "Updated world"
        assert len(agent.tools) == 1
        assert (tmp_path / "closed.txt").exists()
        await manager.close()
        assert not agent.tools and not manager.commands
        assert not agent._listeners
    asyncio.run(check())
    captured = capsys.readouterr()
    assert not captured.out
    assert "extension loading log" in captured.err


def test_failed_flag_application_rolls_back_loaded_extensions(tmp_path):
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        directory = tmp_path / ".pi/extensions"
        directory.mkdir(parents=True)
        (directory / "demo.py").write_text(SOURCE, encoding="utf-8")
        agent = Agent(resolve_model("faux"))
        manager = ExtensionManager(config, agent, flag_values={"unknown_flag": "x"})
        with pytest.raises(ValueError, match="Unknown option"):
            await manager.reload()
        assert not manager.extensions
        assert not agent.tools
        assert not manager.commands
        assert not agent._listeners

    asyncio.run(check())


def test_tool_gate_runs_in_agent_loop(tmp_path, monkeypatch):
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        directory = tmp_path / ".pi/extensions"
        directory.mkdir(parents=True)
        (directory / "demo.py").write_text(SOURCE, encoding="utf-8")
        agent = Agent(resolve_model("faux"), tools=create_default_tools(str(tmp_path)))
        manager = ExtensionManager(config, agent)
        await manager.reload()
        try:
            reset_script_index()
            await agent.prompt("write file")
            assert not (tmp_path / "blocked.txt").exists()
            result = next(m for m in agent.messages if m.role == "toolResult")
            assert result.is_error and "Blocked" in result.content[0].text
        finally:
            await manager.close()
    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
        {"toolCalls": [{"id": "call", "name": "write", "arguments": {"path": "blocked.txt", "content": "bad"}}]},
        {"text": "done"},
    ]))
    asyncio.run(check())


def test_failed_setup_rolls_back_and_rpc_command_is_handled(tmp_path):
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        directory = tmp_path / ".pi/extensions"
        directory.mkdir(parents=True)
        (directory / "a.py").write_text('def setup(api):\n api.register_command("leaked", lambda args, ctx: None)\n raise ValueError("setup failed")', encoding="utf-8")
        (directory / "b.py").write_text(SOURCE, encoding="utf-8")
        agent = Agent(resolve_model("faux"))
        manager = ExtensionManager(config, agent)
        await manager.reload()
        try:
            assert "leaked" not in manager.commands and "hello" in manager.commands
            output = io.StringIO()
            server = RpcServer(agent, JsonWriter(output), extensions=manager)
            await server.dispatch({"id": "one", "type": "prompt", "message": "/handled"})
            reply = json.loads(output.getvalue())
            assert reply["data"]["disposition"] == "handled"
            assert server.task is None
        finally:
            await manager.close()
    asyncio.run(check())


PROVIDER_SOURCE = '''
async def setup(api):
    async def stream_simple(model, context, options=None):
        if False:
            yield None
    stream_simple.owner = "good"
    api.register_provider("leaky", {"api": "openai-completions", "stream_simple": stream_simple})
'''


class FakeMcp:
    def __init__(self):
        self.servers = {}

    async def register_extension_server(self, name, settings, owner):
        self.servers[name] = (settings, owner)

    async def unregister_extension_server(self, name, owner):
        if self.servers.get(name, (None, None))[1] == owner:
            self.servers.pop(name)

    def registered_extension_servers(self):
        return [{"name": name, "config": config, "extensionPath": owner}
                for name, (config, owner) in self.servers.items()]


def test_failed_seal_releases_provider_and_mcp_registrations(tmp_path):
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        directory = tmp_path / ".pi/extensions"
        directory.mkdir(parents=True)
        (directory / "a_good.py").write_text(PROVIDER_SOURCE, encoding="utf-8")
        (directory / "b_bad.py").write_text('''
from pi.agent.types import AgentTool, AgentToolResult

async def setup(api):
    async def stream_simple(model, context, options=None):
        if False:
            yield None

    stream_simple.owner = "bad"

    async def echo(call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[])

    api.register_provider("leaky", {"api": "openai-completions", "stream_simple": stream_simple})
    await api.register_mcp_server("leaky_server", {"command": "echo"})
    tool = AgentTool(name="dup", description="Dup", parameters={"type": "object"}, execute=echo)
    api.register_tool(tool)
    api.register_tool(tool)
''', encoding="utf-8")
        mcp = FakeMcp()
        manager = ExtensionManager(config, Agent(resolve_model("faux")), mcp=mcp)
        await manager.reload()
        try:
            assert len(manager.extensions) == 1
            # The failed extension's provider registration is gone and the live
            # extension's stream function is restored.
            assert getattr(PROVIDER_STREAM_FNS.get("leaky"), "owner", None) == "good"
            assert mcp.servers == {}
        finally:
            await manager.close()
        assert "leaky" not in PROVIDER_STREAM_FNS

    asyncio.run(check())



def test_reload_keeps_the_tool_list_identity(tmp_path):
    # codemode captures agent.tools by reference; rebinding it on reload/close
    # left codemode with a stale tool set.
    async def check():
        config = Configuration(str(tmp_path), str(tmp_path / "user"))
        agent = Agent(resolve_model("faux"))
        tools = agent.tools
        manager = ExtensionManager(config, agent)
        await manager.reload()
        assert agent.tools is tools
        await manager.reload()
        assert agent.tools is tools
        await manager.close()
        assert agent.tools is tools

    asyncio.run(check())
