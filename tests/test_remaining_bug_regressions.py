"""Regression coverage for persistence, transport, and tool boundary failures."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import httpx
import pytest

from pi.agent.agent import Agent
from pi.agent.types import AfterToolCallContext, AgentContext, AgentToolResult, BeforeToolCallContext
from pi.ai.models import resolve_model
from pi.ai.providers import faux
from pi.ai.providers.base import SimpleStreamOptions, run_provider_stream
from pi.ai.providers.pi_messages import stream_pi_messages, wire_message
from pi.ai.types import AssistantMessage, TextContent, ToolCall, TranscriptContext, UserMessage, Usage
from pi.coding_agent import catalog_sync, radius_catalog
from pi.coding_agent.cli import CliRenderer, async_main, interactive_loop
from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.extensions import ExtensionAPI, ExtensionContext
from pi.coding_agent.mcp import McpClient, McpManager
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.session import SessionManager
from pi.coding_agent.subscription_auth import token_record
from pi.coding_agent.tools.bash import create_bash_tool
from pi.coding_agent.tools.edit_diff import Edit, apply_edits_to_normalized_content, generate_diff_string
from pi.coding_agent.tools.grep import create_grep_tool
from pi.coding_agent.trust import TrustStore
from pi.coding_agent.ts_extension import TsExtensionBridge, node_command
from pi.durable.documents import define_doc
from pi.durable.env import Context, ShellExecOptions, ShellOutputSkip, ShellOutputWindow
from pi.durable.env.local import LocalExecutionEnv, _CommandRunner
from pi.durable.session import create_session
from pi.durable.storage import MemoryStorage
from pi.telemetry import InMemoryTelemetryContext, SpanOptions
from pi.telemetry.otlp import OTLPTelemetryContext
from test_nested_tool_calls import leaf_tool, outer_tool, run_once, tool_results


@pytest.mark.parametrize("tail", [b'{"type":', b'{"text":"\xe4'])
def test_resume_incomplete_tail_then_append(tmp_path, tail):
    session = SessionManager(str(tmp_path))
    path = Path(session.start_new())
    session.append(UserMessage(content="saved", timestamp=0))
    with path.open("ab") as handle:
        handle.write(tail)
    restored = SessionManager(str(tmp_path))
    assert len(restored.load(str(path))) == 1
    restored.append(UserMessage(content="new", timestamp=1))
    assert [message.content for message in session.load(str(path))] == ["saved", "new"]


def test_session_rejects_corruption_in_complete_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"broken":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl"):
        SessionManager(str(tmp_path)).load(str(path))


@pytest.mark.parametrize("loader,filename", [(catalog_sync.load_cache, "models-cache.json"),
                                               (radius_catalog.load_cache, "radius-cache.json")])
def test_corrupt_catalog_is_ignored_with_path(tmp_path, caplog, loader, filename):
    (tmp_path / filename).write_text("{", encoding="utf-8")
    assert loader(tmp_path) is None
    assert filename in caplog.text


def test_corrupt_trust_does_not_grant_trust(tmp_path, caplog):
    (tmp_path / "trust.json").write_text("{", encoding="utf-8")
    assert TrustStore(tmp_path).get(str(tmp_path)) is None
    assert "trust.json" in caplog.text


def test_radius_user_models_keep_other_discovered_ids(tmp_path):
    radius_catalog.save_cache(tmp_path, "https://gateway.example", {
        "base_url": "https://gateway.example", "models": [{"id": "a"}, {"id": "b"}]})
    write_json(tmp_path / "models.json", {"providers": {"radius": {"models": [{"id": "a", "name": "User"}]}}})
    registry = ModelRegistry(Configuration(str(tmp_path), str(tmp_path)))
    assert registry.models["radius/a"].name == "User"
    assert "radius/b" in registry.models


def test_radius_token_refresh_margin(monkeypatch):
    monkeypatch.setattr("pi.coding_agent.subscription_auth.time.time", lambda: 1000)
    assert token_record({"access_token": "a", "refresh_token": "r", "expires_in": 300}, "radius")["expires"] == 1240


def test_faux_exhaustion_is_error_event(monkeypatch):
    async def scenario():
        faux.reset_script_index()
        monkeypatch.setenv("FAUX_SCRIPT", '[{"text":"first"}]')
        model = resolve_model("faux")
        await anext_events(model)
        events = await anext_events(model)
        assert events[-1].type == "error"
        assert "exhausted" in events[-1].error.error_message.lower()
    async def anext_events(model):
        return [event async for event in faux.stream_faux(model, TranscriptContext(messages=[]))]
    try:
        asyncio.run(scenario())
    finally:
        faux.reset_script_index()


def test_hosted_raw_item_roundtrip(monkeypatch):
    raw = {"type": "web_search_call", "nested_key": {"tool_name": "opaque"}}
    call = ToolCall(id="hosted", name="web_search", arguments={}, raw_item=raw)
    model = resolve_model("faux")
    assert wire_message(call)["rawItem"] == raw
    async def send(self, request, **kwargs):
        events = [{"type": "toolcall_start", "contentIndex": 0, "id": "hosted", "toolName": "web_search"},
                  {"type": "toolcall_end", "contentIndex": 0, "toolCall": wire_message(call)},
                  {"type": "done", "reason": "stop"}]
        return httpx.Response(200, request=request, text="".join("data: " + json.dumps(event) + "\n\n" for event in events))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    model.base_url = "https://gateway.example"
    async def scenario():
        events = [event async for event in stream_pi_messages(model, TranscriptContext(messages=[]))]
        assert events[-1].message.content[0].raw_item == raw
    asyncio.run(scenario())


def test_unknown_stream_event_with_telemetry(monkeypatch):
    async def send(self, request, **kwargs):
        return httpx.Response(200, request=request, text='data: {}\n\n')
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    async def convert(name, data, state):
        yield {"type": "future_event", "partial": state.snapshot()}
    async def scenario():
        model = resolve_model("faux")
        model.base_url = "https://gateway.example"
        events = [event async for event in run_provider_stream(model, lambda: {}, "/messages", lambda: {},
                  convert, SimpleStreamOptions(telemetry_context=InMemoryTelemetryContext()), 0)]
        assert events[1]["type"] == "future_event"
    asyncio.run(scenario())


def test_otlp_close_restores_cancelled_batch(monkeypatch):
    async def scenario():
        exporting = asyncio.Event()
        sent = []
        async def post(self, url, **kwargs):
            sent.append(kwargs["json"])
            if len(sent) == 1:
                exporting.set()
                await asyncio.Event().wait()
            return httpx.Response(200)
        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        context = OTLPTelemetryContext("https://collector.example")
        await context.start_span(SpanOptions(name="retained"), lambda span: None)
        context._export_task.cancel()
        await asyncio.gather(context._export_task, return_exceptions=True)
        context._export_task = asyncio.create_task(context.flush())
        await exporting.wait()
        await context.close()
        assert len(sent) == 2 and sent[0] == sent[1]
        assert context._records == []
    asyncio.run(scenario())


def test_document_cache_observes_other_session_updates():
    async def scenario():
        storage = MemoryStorage()
        reader, writer = create_session(storage), create_session(storage)
        doc = define_doc(kind="counter", version=1, scope="session", initial=lambda seed: {"n": 0})
        assert await reader.snapshot(doc) is None
        async def increment(tx):
            draft = await tx.doc(doc)
            draft.value["n"] += 1
            draft.dirty = True
        await writer.commit(increment)
        assert await reader.snapshot(doc) == {"n": 1}
        await writer.commit(increment)
        assert await reader.snapshot(doc) == {"n": 2}
    asyncio.run(scenario())


def test_pending_skip_survives_empty_chunk_and_final_flush(tmp_path):
    emitted = []
    options = ShellExecOptions(window=ShellOutputWindow(4, 4, 0, 0),
                               on_output=lambda text, context, info: emitted.append((text, info)))
    runner = _CommandRunner(LocalExecutionEnv(str(tmp_path)), "unused", [], str(tmp_path), options, Context(), None)
    runner._pending_skip = ShellOutputSkip(bytes=10, newlines=1, ends_with_newline=True)
    runner._window("", "stdout")
    runner._emit_final()
    assert len(emitted) == 1 and emitted[0][1].skipped.bytes == 10


def test_edit_normalizes_width_and_reports_actual_changed_line():
    _, edited = apply_edits_to_normalized_content("unchanged\nＡＢＣ\n", [Edit("ABC", "new")], "x")
    assert edited == "unchanged\nnew\n"
    _, first = generate_diff_string("a\nb\nc\nd\ne", "a\nb\nc\nchanged\ne")
    assert first == 4


def test_nested_ids_do_not_collide_with_model_ids():
    async def scenario():
        messages = await run_once([outer_tool("leaf"), leaf_tool()], [
            {"id": "root", "name": "outer", "arguments": {}},
            {"id": "root/1", "name": "leaf", "arguments": {"text": "direct"}}])
        result = next(result for result in tool_results(messages) if result.tool_call_id == "root")
        assert result.nested_calls.calls[0].id != "root/1"
    try:
        asyncio.run(scenario())
    finally:
        faux.reset_script_index()


def test_mcp_batch_sse_and_reconnect_after_eof(tmp_path, monkeypatch):
    async def scenario():
        calls = []
        async def send(self, request, **kwargs):
            calls.append(request)
            if len(calls) == 1:
                body = 'data: [{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}]\n\n'
            else:
                message = json.loads(request.content)
                body = "data: " + json.dumps([
                    {"method": "notifications/resources/list_changed"},
                    {"id": message["id"], "result": {"ok": True}}]) + "\n\n"
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, text=body)
        monkeypatch.setattr(httpx.AsyncClient, "send", send)
        client = McpClient("server", {"url": "https://mcp.example"}, tmp_path)
        client.http = httpx.AsyncClient()
        manager = McpManager(Configuration(str(tmp_path), str(tmp_path)))
        manager.clients["server"] = client
        async def reconnect(server, failed):
            assert failed is client
        monkeypatch.setattr(manager, "reconnect", reconnect)
        try:
            assert await manager.call("server", "tools/list", {}) == {"ok": True}
            assert client.tools_changed and client.resources_changed and len(calls) == 2
        finally:
            await client.close()
    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("rg") is None, reason="requires ripgrep")
def test_rg_context_url_is_not_a_match(tmp_path):
    (tmp_path / "file.txt").write_text("needle\nhttp://localhost:8080/path\nneedle\n", encoding="utf-8")
    async def scenario():
        result = await create_grep_tool(str(tmp_path)).execute("id", {"pattern": "needle", "context": 1, "limit": 2})
        assert result.details["resultLimitReached"] is False
        assert "file.txt:3:needle" in result.content[0].text
    asyncio.run(scenario())


@pytest.mark.parametrize("size,truncated", [(60000, False), (1024 * 1024 + 10, True)])
def test_bash_structured_truncation_is_independent(tmp_path, size, truncated):
    async def scenario():
        tool = create_bash_tool(str(tmp_path), shell_command=lambda command: (
            [sys.executable, "-c", f"import sys; sys.stdout.write('x' * {size})"], False))
        result = await tool.execute("id", {"command": "unused"})
        assert result.structured_content["truncated"] is truncated
        assert len(result.structured_content["output"]) == min(size, 1024 * 1024)
        os.unlink(result.structured_content["full_output_path"])
    asyncio.run(scenario())


def test_plain_theme_reports_current_and_invalid_name(tmp_path, monkeypatch, capsys):
    answers = iter(["/theme", "/theme invalid", "/exit"])
    async def read():
        return next(answers)
    monkeypatch.setattr("pi.coding_agent.cli.read_console_input", read)
    agent = Agent(model=resolve_model("faux"))
    asyncio.run(interactive_loop(agent, CliRenderer(), None))
    output = capsys.readouterr().out
    assert "theme: dark" in output and "unknown theme: invalid" in output


def test_ts_tool_hooks_have_native_event_shape_and_map_result(tmp_path):
    async def scenario():
        agent = Agent(model=resolve_model("faux"))
        api = ExtensionAPI(tmp_path / "extension.ts", ExtensionContext(agent, tmp_path))
        bridge = TsExtensionBridge(api.path, api, [])
        observed = []
        async def request(method, params, timeout=None):
            observed.append(params["event"])
            return {"content": [{"type": "text", "text": "redacted"}], "structuredContent": {"safe": True}}
        bridge._request = request
        message = AssistantMessage(content=[], api="faux", provider="faux", model="faux", timestamp=0,
                                   usage=Usage(), stop_reason="toolUse")
        context = AgentContext(messages=[])
        before = BeforeToolCallContext(message, "id", "bash", {}, {"snake_key": 1}, context)
        await bridge._event_handler("before", "tool_call")(before)
        assert observed[-1] == {"type": "tool_call", "toolCallId": "id", "toolName": "bash", "input": {"snake_key": 1}}
        after = AfterToolCallContext(message, "id", "bash", {}, {},
                                    AgentToolResult(content=[TextContent(text="secret")]), False, context)
        result = await bridge._event_handler("after", "tool_result")(after)
        assert observed[-1]["content"] == [{"type": "text", "text": "secret"}]
        assert observed[-1]["type"] == "tool_result" and observed[-1]["isError"] is False
        assert result.content[0].text == "redacted" and result.structured_content == {"safe": True}
    asyncio.run(scenario())


def test_continue_recovers_partial_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PI_OFFLINE", "1")
    session = SessionManager(str(tmp_path / "sessions"))
    path = Path(session.start_new())
    session.append(UserMessage(content="saved", timestamp=0))
    with path.open("ab") as handle:
        handle.write(b'{"type":')
    faux.reset_script_index()
    try:
        assert asyncio.run(async_main(["--provider", "faux", "--continue", "--tui", "off",
            "--agent-dir", str(tmp_path / "agent"), "--session-dir", str(tmp_path / "sessions"), "continue"])) == 0
        assert len(session.load(str(path))) > 1
    finally:
        faux.reset_script_index()


def test_real_js_extension_reads_tool_hook_fields(tmp_path):
    argv = node_command()
    if argv is None:
        pytest.skip("Node unavailable")
    path = tmp_path / "hook.mjs"
    path.write_text('''export default function(pi) {
        pi.on("tool_call", event => {
            if (event.type !== "tool_call" || event.toolName !== "bash" || event.input.command !== "echo x")
                throw new Error("Invalid hook event");
            return {block: true, reason: event.toolCallId};
        });
        pi.on("tool_result", event => {
            if (event.type !== "tool_result" || event.toolName !== "bash" || event.content[0].text !== "secret")
                throw new Error("Invalid result event");
            return {content: [{type: "text", text: "redacted"}]};
        });
    }''', encoding="utf-8")
    async def scenario():
        agent = Agent(model=resolve_model("faux"))
        api = ExtensionAPI(path, ExtensionContext(agent, tmp_path))
        bridge = TsExtensionBridge(path, api, argv)
        try:
            await bridge.start()
            message = AssistantMessage(content=[], api="faux", provider="faux", model="faux", timestamp=0,
                                       usage=Usage(), stop_reason="toolUse")
            context = AgentContext(messages=[])
            before = BeforeToolCallContext(message, "id", "bash", {}, {"command": "echo x"}, context)
            result = await api.handlers["tool_call"][0](before)
            assert result.block and result.reason == "id"
            after = AfterToolCallContext(message, "id", "bash", {}, {},
                AgentToolResult(content=[TextContent(text="secret")]), False, context)
            result = await api.handlers["tool_result"][0](after)
            assert result.content[0].text == "redacted"
        finally:
            await bridge.close()
    asyncio.run(scenario())
