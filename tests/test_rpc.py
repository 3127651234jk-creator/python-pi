import asyncio
import io
import json
import os
import queue
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from pi.ai.providers import STREAM_FNS
from pi.ai.providers.base import make_partial
from pi.ai.types import (
    AssistantMessage,
    StreamErrorEvent,
    SystemMessage,
    TextContent,
    Usage,
    UserMessage,
    get_message_text,
)

from pi.agent.agent import Agent
from pi.agent.types import AgentToolResult, ToolExecutionEndEvent
from pi.ai.models import Model, resolve_model
from pi.ai.providers.faux import reset_script_index
from pi.ai.retry import RetryPolicy
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.rpc import JsonWriter, RpcServer
from pi.coding_agent.rpc_client import RpcClient
from pi.coding_agent.rpc_wire import event_to_ts
from pi.coding_agent.cli import async_main
from pi.coding_agent.session import SessionManager


def test_rpc_wire_tool_execution_end_duration_mapping():
    """tool_execution_end carries durationMs only for executed calls (#10549)."""
    timed = ToolExecutionEndEvent(
        tool_call_id="t", tool_name="bash",
        result=AgentToolResult(content=[TextContent(text="out")], details={}),
        is_error=False, duration_ms=1500,
    )
    assert event_to_ts(timed)["durationMs"] == 1500
    untimed = ToolExecutionEndEvent(
        tool_call_id="t", tool_name="bash",
        result=AgentToolResult(content=[], details={}), is_error=False,
    )
    assert "durationMs" not in event_to_ts(untimed)


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


def test_rpc_auto_retry_events_do_not_crash_the_run():
    """A retryable provider error maps to auto_retry_* wire events, not run_error (F-CA-11)."""

    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps([{"error": "terminated"}, {"text": "recovered"}])
        reset_script_index()
        try:
            output = io.StringIO()
            agent = Agent(resolve_model("faux"), retry=RetryPolicy(base_delay_ms=0))
            server = RpcServer(agent, JsonWriter(output))
            await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
            await server.task
            records = [json.loads(line) for line in output.getvalue().splitlines() if line]
            assert records[-1]["type"] == "agent_settled"
            assert not any(record["type"] == "run_error" for record in records)
            start = next(record for record in records if record["type"] == "auto_retry_start")
            assert start == {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3,
                             "delayMs": 0, "errorMessage": "terminated"}
            end = next(record for record in records if record["type"] == "auto_retry_end")
            assert end["success"] is True
            assert end["attempt"] == 1
            assert "finalError" not in end
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            reset_script_index()

    asyncio.run(check())


def test_rpc_agent_settled_reports_abort():
    """agent_settled carries the run's abort state (F-CA-02, #10607)."""

    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps(
            [{"toolCalls": [{"id": "t1", "name": "aborter", "arguments": {}}]}, {"text": "unused"}])
        reset_script_index()
        try:
            output = io.StringIO()
            agent = Agent(resolve_model("faux"), tools=[_aborting_tool()])
            server = RpcServer(agent, JsonWriter(output))
            await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
            await server.task
            records = [json.loads(line) for line in output.getvalue().splitlines() if line]
            settled = records[-1]
            assert settled["type"] == "agent_settled"
            assert settled["aborted"] is True
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            reset_script_index()

    asyncio.run(check())


def test_rpc_agent_settled_normal_run_not_aborted():
    async def check():
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output))
        reset_script_index()
        await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
        await server.task
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        assert records[-1] == {"type": "agent_settled", "aborted": False}

    asyncio.run(check())


def test_rpc_agent_settled_notifies_extensions():
    async def check():
        output = io.StringIO()
        seen = []

        class Extensions:
            async def process_input(self, text):
                return text

            async def emit(self, name, event):
                seen.append((name, event))

        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output), extensions=Extensions())
        reset_script_index()
        await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
        await server.task
        assert seen == [("agent_settled", {"type": "agent_settled", "aborted": False})]

    asyncio.run(check())


def _aborting_tool():
    from pi.agent.types import AgentTool

    async def execute(tool_call_id, params, signal=None, on_update=None):
        signal.abort()
        return AgentToolResult(content=[TextContent(text="ran")], details={})

    return AgentTool(
        name="aborter",
        label="aborter",
        description="abort the run",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def test_rpc_wire_session_event_shapes():
    """Session-level events map to the TS AgentSessionEvent JSON shapes (F-CA-12)."""
    from pi.agent.types import (
        CompactionEndEvent,
        CompactionStartEvent,
        SummarizationRetryAttemptStartEvent,
        SummarizationRetryFinishedEvent,
        SummarizationRetryScheduledEvent,
    )

    assert event_to_ts(CompactionStartEvent(reason="overflow")) == {
        "type": "compaction_start", "reason": "overflow"}
    end = event_to_ts(CompactionEndEvent(
        reason="manual", result={"summary": "s", "tokens_before": 10, "tokens_after": 2}, will_retry=True))
    assert end == {"type": "compaction_end", "reason": "manual",
                   "result": {"summary": "s", "tokensBefore": 10, "estimatedTokensAfter": 2},
                   "aborted": False, "willRetry": True}
    failed = event_to_ts(CompactionEndEvent(reason="threshold", error_message="boom"))
    assert "result" not in failed and failed["errorMessage"] == "boom"
    assert event_to_ts(SummarizationRetryScheduledEvent(
        attempt=1, max_attempts=3, delay_ms=2000, error_message="terminated")) == {
        "type": "summarization_retry_scheduled", "attempt": 1, "maxAttempts": 3,
        "delayMs": 2000, "errorMessage": "terminated"}
    assert event_to_ts(SummarizationRetryAttemptStartEvent(source="compaction", reason="overflow")) == {
        "type": "summarization_retry_attempt_start", "source": "compaction", "reason": "overflow"}
    assert event_to_ts(SummarizationRetryFinishedEvent()) == {"type": "summarization_retry_finished"}


def test_rpc_agent_end_carries_will_retry():
    """agent_end.willRetry mirrors the agent's retry decision (F-CA-12)."""

    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps([{"error": "terminated"}, {"text": "recovered"}])
        reset_script_index()
        try:
            output = io.StringIO()
            agent = Agent(resolve_model("faux"), retry=RetryPolicy(base_delay_ms=0))
            server = RpcServer(agent, JsonWriter(output))
            await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
            await server.task
            records = [json.loads(line) for line in output.getvalue().splitlines() if line]
            ends = [record for record in records if record["type"] == "agent_end"]
            assert [end["willRetry"] for end in ends] == [True, False]
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            reset_script_index()

    asyncio.run(check())


def test_rpc_message_persist_emits_entry_appended(tmp_path):
    async def check():
        session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        session.start_new()
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output), session=session)
        reset_script_index()
        await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
        await server.task
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        appended = [record for record in records if record["type"] == "entry_appended"]
        # One entry per persisted message: the user prompt arrives through the
        # session before the run, the assistant/toolResults during it.
        assert len(appended) >= 2
        assert appended[0]["entry"]["type"] == "message"
        assert appended[0]["entry"]["message"]["role"] == "user"
        ids = {record["entry"]["id"] for record in appended}
        assert ids == {entry.id for entry in session.entries}

    asyncio.run(check())


def test_rpc_thinking_level_and_session_name_broadcast():
    async def check():
        session = SessionManager(session_dir="unused", cwd=None)
        session.start_new()
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output), session=session)
        await server.dispatch({"id": "t", "type": "set_thinking_level", "level": "high"})
        await server.dispatch({"id": "n", "type": "set_session_name", "name": "deep dive"})
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        assert {"type": "thinking_level_changed", "level": "high"} in records
        assert {"type": "session_info_changed", "name": "deep dive"} in records

    asyncio.run(check())


def test_rpc_queue_update_broadcasts_on_mid_run_consumption():
    """Draining the steering queue mid-run emits queue_update (F-CA-12)."""

    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps([
            {"toolCalls": [{"id": "t1", "name": "steerer", "arguments": {}}]},
            {"text": "done"},
        ])
        reset_script_index()
        try:
            output = io.StringIO()

            def make_tool():
                from pi.agent.types import AgentTool

                async def execute(tool_call_id, params, signal=None, on_update=None):
                    server.agent.steer("mid-run note")
                    return AgentToolResult(content=[TextContent(text="steered")], details={})

                return AgentTool(name="steerer", label="steerer", description="steer the run",
                                 parameters={"type": "object", "properties": {}}, execute=execute)

            agent = Agent(resolve_model("faux"), tools=[make_tool()])
            server = RpcServer(agent, JsonWriter(output))
            await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
            await server.task
            records = [json.loads(line) for line in output.getvalue().splitlines() if line]
            updates = [record for record in records if record["type"] == "queue_update"]
            # The tool enqueued (no broadcast) and the next turn drained it (broadcast).
            assert updates == [{"type": "queue_update", "steering": [], "followUp": []}]
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            reset_script_index()

    asyncio.run(check())


def test_rpc_manual_compaction_emits_events(tmp_path):
    async def check():
        session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        session.start_new()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "part one"}, {"text": "part two"}])
        reset_script_index()
        try:
            output = io.StringIO()
            model = Model(id="faux", name="Faux", api="faux", provider="faux", base_url="",
                          context_window=3000, max_tokens=1024)
            agent = Agent(model)
            agent.messages = [UserMessage(content="a" * 700, timestamp=0),
                              UserMessage(content="b" * 700, timestamp=1)]
            server = RpcServer(agent, JsonWriter(output), session=session, registry=object())
            await server.dispatch({"id": "c", "type": "compact"})
            records = [json.loads(line) for line in output.getvalue().splitlines() if line]
            starts = [record for record in records if record["type"] == "compaction_start"]
            ends = [record for record in records if record["type"] == "compaction_end"]
            assert starts == [{"type": "compaction_start", "reason": "manual"}]
            assert len(ends) == 1
            assert ends[0]["aborted"] is False and ends[0]["willRetry"] is False
            assert ends[0]["result"]["summary"] == "part two"
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            reset_script_index()

    asyncio.run(check())


def test_rpc_set_auto_retry_toggles_policy():
    async def check():
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output))
        assert agent.retry is None
        await server.dispatch({"id": "on", "type": "set_auto_retry", "enabled": True})
        assert agent.retry is not None and agent.retry.enabled and agent.compactor.retry is agent.retry
        await server.dispatch({"id": "off", "type": "set_auto_retry", "enabled": False})
        assert agent.retry.enabled is False
        await server.dispatch({"id": "bad", "type": "set_auto_retry", "enabled": "yes"})
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        by_id = {record.get("id"): record for record in records}
        assert by_id["on"]["success"] and by_id["off"]["success"]
        assert not by_id["bad"]["success"]

    asyncio.run(check())


def test_rpc_abort_retry_cancels_pending_backoff():
    async def check():
        os.environ["FAUX_SCRIPT"] = json.dumps([{"error": "terminated"}, {"text": "recovered"}])
        reset_script_index()
        try:
            output = io.StringIO()
            agent = Agent(resolve_model("faux"), retry=RetryPolicy(base_delay_ms=60_000))
            server = RpcServer(agent, JsonWriter(output))
            await server.dispatch({"id": "prompt", "type": "prompt", "message": "hello"})
            for _ in range(200):
                if any('"auto_retry_start"' in line for line in output.getvalue().splitlines()):
                    break
                await asyncio.sleep(0.01)
            await server.dispatch({"id": "abort", "type": "abort_retry"})
            await asyncio.wait_for(server.task, 5)
            records = [json.loads(line) for line in output.getvalue().splitlines() if line]
            ends = [record for record in records if record["type"] == "auto_retry_end"]
            assert ends and ends[-1]["success"] is False and ends[-1]["finalError"] == "Retry cancelled"
            assert not any("recovered" in line for line in output.getvalue().splitlines())
            assert records[-1] == {"type": "agent_settled", "aborted": False}
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            reset_script_index()

    asyncio.run(check())


def test_rpc_new_session_forks_from_parent(tmp_path):
    async def check():
        parent = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        parent.start_new()
        parent.append(UserMessage(content="parent question", timestamp=0))
        parent.append(AssistantMessage(content=[TextContent(text="parent answer")], api="faux",
                                       provider="faux", model="faux", usage=Usage(),
                                       stop_reason="stop", timestamp=1))
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        server = RpcServer(agent, JsonWriter(output), session=SessionManager(
            session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path)))
        await server.dispatch({"id": "n", "type": "new_session", "parentSession": parent.session_file})
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        reply = records[-1]
        assert reply["id"] == "n" and reply["success"] and reply["data"] == {"cancelled": False}
        texts = [get_message_text(m.content) for m in agent.messages if m.role in ("user", "assistant")]
        assert texts == ["parent question", "parent answer"]
        assert server.session.session_file != parent.session_file
        assert server.session.leaf_id == parent.leaf_id

    asyncio.run(check())


def test_rpc_get_commands_includes_prompts_and_skills():
    async def check():
        output = io.StringIO()

        class FakeResources:
            class config:
                cwd = "/proj"

            prompts = {"hi": SimpleNamespace(name="hi", description="Hi there",
                                             path="/proj/.pi/prompts/hi.md")}
            skills = {"greet": SimpleNamespace(name="greet", description="Greets",
                                               path="/proj/.pi/skills/greet")}

        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), resources=FakeResources())
        await server.dispatch({"id": "c", "type": "get_commands"})
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        commands = records[-1]["data"]["commands"]
        assert {"name": "hi", "description": "Hi there", "source": "prompt",
                "sourceInfo": {"path": "/proj/.pi/prompts/hi.md", "source": "local",
                               "scope": "project", "origin": "top-level"}} in commands
        assert {"name": "skill:greet", "description": "Greets", "source": "skill",
                "sourceInfo": {"path": "/proj/.pi/skills/greet", "source": "local",
                               "scope": "project", "origin": "top-level"}} in commands

    asyncio.run(check())


def test_new_session_keeps_system_messages_and_stays_in_dispatch():
    """new_session must not raise NameError out of dispatch (killing the server)."""

    async def check():
        output = io.StringIO()
        agent = Agent(resolve_model("faux"))
        agent.messages = [
            SystemMessage(content="sys", timestamp=0),
            UserMessage(content="hi", timestamp=0),
        ]
        server = RpcServer(agent, JsonWriter(output))
        await server.dispatch({"id": "n", "type": "new_session"})
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        assert records[-1]["id"] == "n"
        assert records[-1]["success"]
        assert [message.role for message in agent.messages] == ["system"]

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


def test_serve_shutdown_does_not_wait_for_blocked_stdin():
    """Cancelling serve() must not wedge asyncio.run()'s executor shutdown.

    stdin stays open (no EOF) so readline() blocks forever; if it were parked in
    the loop's default executor, process exit would hang waiting for that thread.
    """
    script = textwrap.dedent(
        """
        import asyncio, io
        from pi.agent.agent import Agent
        from pi.ai.models import resolve_model
        from pi.coding_agent.rpc import JsonWriter, RpcServer

        async def main():
            server = RpcServer(Agent(resolve_model("faux")), JsonWriter(io.StringIO()))
            serving = asyncio.create_task(server.serve())
            await asyncio.sleep(0.2)
            serving.cancel()
            try:
                await serving
            except asyncio.CancelledError:
                pass

        asyncio.run(main())
        """
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=root)
    try:
        returncode = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise AssertionError("serve() hung on a blocked stdin readline during shutdown")
    assert returncode == 0, process.stderr.read().decode()


def test_extension_ui_response_is_read_while_command_is_running():
    class Source:
        def __init__(self):
            self.lines = queue.Queue()

        def readline(self):
            return self.lines.get()

    class Extensions:
        server = None

        async def process_input(self, text):
            confirmed = await self.server.ui_sink({"method": "confirm", "title": "Continue?"})
            assert confirmed is True
            return None

    async def check():
        output = io.StringIO()
        source = Source()
        extensions = Extensions()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), extensions=extensions)
        extensions.server = server
        serving = asyncio.create_task(server.serve(source))
        source.lines.put(b'{"id":"prompt","type":"prompt","message":"hello"}\n')

        request = None
        for _ in range(100):
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            request = next((record for record in records
                            if record.get("type") == "extension_ui_request"), None)
            if request is not None:
                break
            await asyncio.sleep(0.01)
        assert request is not None

        reply = {"type": "extension_ui_response", "id": request["id"], "value": True}
        source.lines.put((json.dumps(reply) + "\n").encode())
        source.lines.put(b'{"id":"state","type":"get_state"}\n')
        source.lines.put(b"")
        await asyncio.wait_for(serving, 2)

        records = [json.loads(line) for line in output.getvalue().splitlines()]
        responses = {record.get("id"): record for record in records if record.get("type") == "response"}
        assert responses["prompt"]["data"] == {"disposition": "handled"}
        assert responses["state"]["success"]

    asyncio.run(check())


def test_extension_ui_request_times_out(monkeypatch):
    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output))
        monkeypatch.setattr("pi.coding_agent.rpc._UI_RESPONSE_TIMEOUT_SECONDS", 0.01)
        with pytest.raises(asyncio.TimeoutError, match="Extension UI request timed out"):
            await server.ui_sink({"method": "input", "message": "Value"})
        assert not server.ui_futures

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
                        {"id": "m", "type": "set_model", "provider": "faux", "modelId": "faux"}):
            await server.dispatch(command)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        by_id = {record.get("id"): record for record in records}
        assert by_id["st"]["data"] == {"disposition": "queued"}
        assert by_id["fu"]["data"] == {"disposition": "queued"}
        assert by_id["lv"]["data"] == {"level": "minimal"}
        assert by_id["g"]["data"]["pendingMessageCount"] == 2
        assert by_id["g"]["data"]["steeringMode"] == "one-at-a-time"
        assert by_id["c"]["data"] == {"steering": ["later"], "followUp": ["after"]}
        updates = [record for record in records if record["type"] == "queue_update"]
        assert updates[0]["steering"] == ["later"] and updates[0]["followUp"] == []
        assert by_id["u"]["error"] == "Unknown command: nope"
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
