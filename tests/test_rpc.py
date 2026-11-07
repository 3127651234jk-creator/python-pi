import asyncio
import io
import json
import os
from pi.ai.providers import STREAM_FNS
from pi.ai.providers.base import make_partial
from pi.ai.types import StreamErrorEvent, UserMessage

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.ai.providers.faux import reset_script_index
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.rpc import JsonWriter, RpcServer
from pi.coding_agent.rpc_client import RpcClient
from pi.coding_agent.cli import async_main
from pi.coding_agent.session import SessionManager


def test_rpc_validation_state_and_events():
    async def check():
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output))
        await server.dispatch({"id": "state", "type": "get_state"})
        await server.dispatch({"id": "bad", "type": "unknown"})
        reset_script_index()
        await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello\u2028world"})
        await server.task
        records = [json.loads(line) for line in output.getvalue().split("\n") if line]
        assert records[0]["id"] == "state"
        assert "api_key" not in records[0]["data"]["model"]
        assert not records[1]["success"]
        assert records[2]["data"]["disposition"] == "started"
        assert records[-1]["type"] == "agent_settled"
        assert any(record["type"] == "message_update" for record in records)
    asyncio.run(check())


def test_parse_error_does_not_stop_server():
    async def check():
        output = io.StringIO()
        source = io.BytesIO(b'not-json\n[]\n{"id":"ok","type":"get_state"}\n')
        await RpcServer(Agent(resolve_model("faux")), JsonWriter(output)).serve(source)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        assert len(records) == 3
        assert not records[0]["success"] and not records[1]["success"]
        assert records[2]["id"] == "ok" and records[2]["success"]
    asyncio.run(check())


def test_client_subprocess_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path))
    monkeypatch.setenv("FAUX_SCRIPT", '[{"text":"hello"}]')
    async def check():
        client = await RpcClient().start(["--provider", "faux", "--no-session"])
        try:
            state = await client.request("get_state")
            assert not state["isStreaming"]
            events = await client.prompt_and_wait("hello")
            assert events[-1]["type"] == "agent_settled"
            messages = await client.request("get_messages")
            assert messages["messages"][-1]["role"] == "assistant"
            states = await asyncio.gather(client.request("get_state"), client.request("get_tools"))
            assert states[1]["tools"]
        finally:
            await client.close()
        assert client.process.returncode == 0
    asyncio.run(check())


def test_json_mode_stdout_is_jsonl(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "user"))
    reset_script_index()
    assert asyncio.run(async_main(["--mode", "json", "--provider", "faux", "--no-session", "hello"])) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.split("\n") if line]
    assert records[-1]["type"] == "agent_settled"


def test_abort_remains_responsive_during_prompt(monkeypatch):
    async def stream(model, context, options):
        await options.signal.wait()
        message = make_partial(model, 0)
        message.stop_reason = "aborted"
        yield StreamErrorEvent(reason="aborted", error=message)

    monkeypatch.setitem(STREAM_FNS, "faux", stream)

    async def check():
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output))
        await server.dispatch({"type": "prompt", "message": "wait"})
        while not agent.is_streaming:
            await asyncio.sleep(0)
        await server.dispatch({"id": "busy", "type": "prompt", "message": "another"})
        await server.dispatch({"id": "state", "type": "get_state"})
        await server.dispatch({"id": "abort", "type": "abort"})
        await asyncio.wait_for(server.task, 2)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        replies = {record.get("id"): record for record in records if "id" in record}
        assert not replies["busy"]["success"]
        assert replies["state"]["data"]["isStreaming"]
        assert replies["abort"]["success"]
        assert records[-1]["type"] == "agent_settled"
    asyncio.run(check())


def test_rpc_wire_shapes(tmp_path):
    async def check():
        session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        session.start_new()
        session.append(UserMessage(content="hi", timestamp=0))
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), session=session, registry=None)
        for command in ({"id": "s", "type": "get_state"},
                        {"id": "e", "type": "get_entries"},
                        {"id": "t", "type": "get_tree"},
                        {"id": "f", "type": "get_fork_messages"},
                        {"id": "x", "type": "export_html"},
                        {"id": "c", "type": "clear_queue"}):
            await server.dispatch(command)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        by_id = {record.get("id"): record for record in records}
        state = by_id["s"]["data"]
        assert set(state) == {"model", "thinkingLevel", "isStreaming", "isCompacting", "steeringMode",
                              "followUpMode", "sessionFile", "sessionId", "autoCompactionEnabled",
                              "messageCount", "pendingMessageCount"}
        model = state["model"]
        assert set(model) == {"id", "name", "api", "provider", "baseUrl", "input", "cost", "reasoning",
                              "contextWindow", "maxTokens"}
        assert set(model["cost"]) == {"input", "output", "cacheRead", "cacheWrite"}
        assert state["sessionId"] == os.path.basename(session.session_file)[:-len(".jsonl")]
        assert state["sessionFile"] == session.session_file
        entries = by_id["e"]["data"]
        assert entries["leafId"] == session.leaf_id
        entry = entries["entries"][0]
        assert entry["type"] == "message" and entry["parentId"] is None
        assert entry["timestamp"].endswith("Z")
        assert entry["message"] == {"role": "user", "content": "hi", "timestamp": 0}
        tree = by_id["t"]["data"]
        assert tree["leafId"] == session.leaf_id
        assert tree["tree"][0]["entry"]["id"] == entry["id"] and tree["tree"][0]["children"] == []
        assert by_id["f"]["data"] == {"messages": [{"entryId": entry["id"], "text": "hi"}]}
        assert by_id["x"]["data"]["path"].endswith(".html")
        assert by_id["c"]["data"] == {"steering": [], "followUp": []}
    asyncio.run(check())


def test_rpc_queue_model_and_error_commands(tmp_path):
    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), registry=None)
        for command in ({"id": "st", "type": "steer", "message": "later"},
                        {"id": "fu", "type": "follow_up", "message": "after"},
                        {"id": "lv", "type": "cycle_thinking_level"},
                        {"id": "g", "type": "get_state"},
                        {"id": "c", "type": "clear_queue"},
                        {"id": "u", "type": "nope"},
                        {"id": "b", "type": "bash", "command": "dir"},
                        {"id": "m", "type": "set_model", "provider": "faux", "modelId": "faux"}):
            await server.dispatch(command)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        by_id = {record.get("id"): record for record in records}
        assert by_id["st"]["data"] == {"disposition": "queued"}
        assert by_id["fu"]["data"] == {"disposition": "queued"}
        assert by_id["lv"]["data"] == {"level": "minimal"}
        assert by_id["g"]["data"]["pendingMessageCount"] == 2
        assert by_id["g"]["data"]["steeringMode"] == "all"
        assert by_id["c"]["data"] == {"steering": ["later"], "followUp": ["after"]}
        updates = [record for record in records if record["type"] == "queue_update"]
        assert updates[0]["steering"] == ["later"] and updates[0]["followUp"] == []
        assert by_id["u"]["error"] == "Unknown command: nope"
        assert "not supported" in by_id["b"]["error"]
        assert "No model registry" in by_id["m"]["error"]
    asyncio.run(check())


def test_rpc_set_model_and_available_models(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_PY_AGENT_DIR", str(tmp_path / "agent"))

    async def check():
        registry = ModelRegistry(Configuration(str(tmp_path), str(tmp_path / "agent")))
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), registry=registry)
        await server.dispatch({"id": "m", "type": "set_model", "provider": "faux", "modelId": "faux"})
        await server.dispatch({"id": "a", "type": "get_available_models"})
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        by_id = {record.get("id"): record for record in records}
        assert by_id["m"]["data"]["provider"] == "faux" and by_id["m"]["data"]["id"] == "faux"
        assert by_id["a"]["data"]["models"]
        assert all(set(model) >= {"id", "name", "api", "provider", "baseUrl"} for model in by_id["a"]["data"]["models"])
    asyncio.run(check())


def test_message_update_wire_shape(monkeypatch):
    monkeypatch.setenv("FAUX_SCRIPT", '[{"text":"hello"}]')

    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output))
        reset_script_index()
        await server.dispatch({"id": "p", "type": "prompt", "message": "hello"})
        await server.task
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        updates = [record for record in records if record["type"] == "message_update"]
        assert updates
        assert set(updates[0]) == {"type", "usage", "assistantMessageEvent"}
        assert "partial" not in updates[0]["assistantMessageEvent"]
        text_start = next(event["assistantMessageEvent"] for event in updates
                          if event["assistantMessageEvent"]["type"] == "text_start")
        assert set(text_start) == {"type", "contentIndex"}
        text_end = next(event["assistantMessageEvent"] for event in updates
                        if event["assistantMessageEvent"]["type"] == "text_end")
        assert text_end["content"] == "hello"
        # The Python loop consumes the stream's terminal done event, so the
        # final authoritative message arrives through message_end.
        end = next(record for record in records
                   if record["type"] == "message_end" and record["message"]["role"] == "assistant")
        assert end["message"]["stopReason"] == "stop" and end["message"]["usage"]["totalTokens"] >= 0
    asyncio.run(check())


def test_toolcall_start_wire_shape(monkeypatch):
    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
        {"toolCalls": [{"id": "call1", "name": "read", "arguments": {"path": "x"}}]},
        {"text": "done"},
    ]))

    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output))
        reset_script_index()
        await server.dispatch({"id": "p", "type": "prompt", "message": "read x"})
        await server.task
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        updates = [record for record in records if record["type"] == "message_update"]
        start = next(event["assistantMessageEvent"] for event in updates
                     if event["assistantMessageEvent"]["type"] == "toolcall_start")
        assert set(start) == {"type", "contentIndex", "id", "toolName"}
        assert start["id"] == "call1" and start["toolName"] == "read"
        end = next(event["assistantMessageEvent"] for event in updates
                   if event["assistantMessageEvent"]["type"] == "toolcall_end")
        assert end["toolCall"] == {"type": "toolCall", "id": "call1", "name": "read", "arguments": {"path": "x"}}
    asyncio.run(check())


def test_prompt_while_busy_queues_with_streaming_behavior(monkeypatch):
    monkeypatch.setenv("FAUX_SCRIPT", '[{"text":"one"},{"text":"two"}]')

    async def check():
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output))
        reset_script_index()
        await server.dispatch({"type": "prompt", "message": "first"})
        while not agent.is_streaming:
            await asyncio.sleep(0)
        await server.dispatch({"id": "q", "type": "prompt", "message": "second",
                               "streamingBehavior": "followUp"})
        await server.dispatch({"id": "s", "type": "steer", "message": "third", "images": [
            {"type": "image", "data": "aGk=", "mimeType": "image/png"}]})
        await asyncio.wait_for(server.task, 5)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        by_id = {record.get("id"): record for record in records if "id" in record}
        assert by_id["q"]["data"] == {"disposition": "queued"}
        assert by_id["s"]["data"] == {"disposition": "queued"}
        assert records[-1]["type"] == "agent_settled"
    asyncio.run(check())


def test_get_entries_since(tmp_path):
    async def check():
        session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        session.start_new()
        session.append(UserMessage(content="one", timestamp=0))
        session.append(UserMessage(content="two", timestamp=0))
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), session=session, registry=None)
        await server.dispatch({"id": "a", "type": "get_entries", "since": session.entries[0].id})
        await server.dispatch({"id": "b", "type": "get_entries", "since": "zzzz"})
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        by_id = {record.get("id"): record for record in records}
        assert [entry["message"]["content"] for entry in by_id["a"]["data"]["entries"]] == ["two"]
        assert not by_id["b"]["success"] and "Entry not found" in by_id["b"]["error"]
    asyncio.run(check())
