import asyncio
import io
import json
from pi.ai.providers import STREAM_FNS
from pi.ai.providers.base import make_partial
from pi.ai.types import StreamErrorEvent

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.ai.providers.faux import reset_script_index
from pi.coding_agent.rpc import JsonWriter, RpcServer
from pi.coding_agent.rpc_client import RpcClient
from pi.coding_agent.cli import async_main


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
            assert not state["is_streaming"]
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
        assert replies["state"]["data"]["is_streaming"]
        assert replies["abort"]["success"]
        assert records[-1]["type"] == "agent_settled"
    asyncio.run(check())
