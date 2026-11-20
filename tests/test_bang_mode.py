"""``!`` / ``!!`` shell escape: execution, session record, projection, RPC (F-TUI-04)."""

import asyncio
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.agent.agent import Agent
from pi.agent.agent_loop import default_convert_to_llm
from pi.agent.types import AgentTool, AgentToolResult, BashExecutionMessage, bash_execution_to_text
from pi.ai.models import resolve_model
from pi.ai.types import TextContent
from pi.coding_agent.bash_mode import execute_bang, render_bang_output
from pi.coding_agent.rpc import JsonWriter, RpcServer
from pi.coding_agent.session import SessionManager


def stub_bash_tool(monkeypatch, output="hello", exit_code=0, commands=None):
    """Replace the shared runner with a canned AgentToolResult."""

    def fake_create(cwd):
        async def execute(call_id, params, signal=None, on_update=None):
            if commands is not None:
                commands.append(params["command"])
            return AgentToolResult(content=[TextContent(text=output)],
                                   structured_content={"output": output, "truncated": False,
                                                       "exit_code": exit_code})

        return AgentTool(name="bash", label="bash", description="", parameters={}, execute=execute)

    monkeypatch.setattr("pi.coding_agent.bash_mode.create_bash_tool", fake_create)


def test_bang_execution_records_session_and_projects(tmp_path, monkeypatch):
    """`!` output enters the transcript and projects to user text; `!!` does not."""
    stub_bash_tool(monkeypatch)
    session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    session.start_new()

    async def run():
        visible = await execute_bang("! echo hello", str(tmp_path))
        hidden = await execute_bang("!! echo secret", str(tmp_path))
        for message in (visible, hidden):
            session.append_bash_execution(message)

    asyncio.run(run())
    agent_messages = session.load(session.session_file)
    assert [m.role for m in agent_messages] == ["bashExecution", "bashExecution"]
    projected = default_convert_to_llm(agent_messages)
    assert [m.role for m in projected] == ["user"]
    assert projected[0].content == "Ran `echo hello`\n```\nhello\n```"


def test_bang_message_text_covers_exit_and_truncation():
    message = BashExecutionMessage(command="ls", output="", exit_code=2, timestamp=0)
    assert bash_execution_to_text(message) == "Ran `ls`\n(no output)\n\nCommand exited with code 2"
    message = BashExecutionMessage(command="ls", output="x", exit_code=0, truncated=True,
                                   full_output_path="/tmp/full.log", timestamp=0)
    assert "[Output truncated. Full output: /tmp/full.log]" in bash_execution_to_text(message)
    cancelled = BashExecutionMessage(command="ls", output="x", exit_code=None, cancelled=True, timestamp=0)
    assert "(command cancelled)" in bash_execution_to_text(cancelled)


def test_render_marks_excluded_header():
    assert render_bang_output(BashExecutionMessage(command="ls", output="", timestamp=0)).startswith("! ls")
    assert render_bang_output(BashExecutionMessage(command="ls", output="",
                                                   exclude_from_context=True, timestamp=0)).startswith("!! ls")


def test_rpc_bash_records_and_returns_result(tmp_path, monkeypatch):
    """The RPC `bash` command runs, records a session entry, and returns the shape."""
    commands = []
    stub_bash_tool(monkeypatch, output="listed", exit_code=0, commands=commands)

    async def check():
        session = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
        session.start_new()
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), session=session)
        await server.dispatch({"id": "b", "type": "bash", "command": "ls -la"})
        await server._bash_task
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        reply = next(record for record in records if record.get("id") == "b")
        assert reply["success"]
        assert reply["data"] == {"output": "listed", "exitCode": 0, "cancelled": False, "truncated": False}
        appended = next(record for record in records if record.get("type") == "entry_appended")
        assert appended["entry"]["message"]["role"] == "bashExecution"
        assert [m.role for m in server.agent.messages] == ["bashExecution"]
        # The RPC payload is a raw command: no bang prefix character is eaten.
        assert commands == ["ls -la"]

    asyncio.run(check())


def test_rpc_bash_exclude_from_context(monkeypatch):
    stub_bash_tool(monkeypatch)

    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output))
        await server.dispatch({"id": "b", "type": "bash", "command": "top", "excludeFromContext": True})
        await server._bash_task
        await server.dispatch({"id": "x", "type": "abort_bash"})
        message = server.agent.messages[0]
        assert message.exclude_from_context is True
        assert default_convert_to_llm(server.agent.messages) == []

    asyncio.run(check())


def test_rpc_abort_bash_reaches_a_running_command(monkeypatch):
    """dispatch() does not block on bash, so abort_bash can cancel it."""
    observed = {}

    async def slow_shell(command, cwd, signal=None, exclude_from_context=False):
        observed["signal"] = signal
        while not signal.aborted:
            await asyncio.sleep(0.01)
        return BashExecutionMessage(command=command, output="", exit_code=None, cancelled=True, timestamp=0)

    monkeypatch.setattr("pi.coding_agent.rpc.execute_shell", slow_shell)

    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output))
        await server.dispatch({"id": "b", "type": "bash", "command": "sleep 10"})
        task = server._bash_task
        for _ in range(100):
            if "signal" in observed:
                break
            await asyncio.sleep(0.01)
        assert "signal" in observed
        await server.dispatch({"id": "a", "type": "abort_bash"})
        await asyncio.wait_for(task, 2)
        assert observed["signal"].aborted is True
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        bash_reply = next(record for record in records if record.get("id") == "b")
        assert bash_reply["data"]["cancelled"] is True

    asyncio.run(check())


def test_bang_accepts_leading_whitespace(monkeypatch):
    commands = []
    stub_bash_tool(monkeypatch, commands=commands)

    async def run():
        hidden = await execute_bang("  !! echo secret", "/tmp")
        visible = await execute_bang(" ! echo hello", "/tmp")
        return hidden, visible

    hidden, visible = asyncio.run(run())
    assert hidden.exclude_from_context is True and hidden.command == "echo secret"
    assert visible.exclude_from_context is False and visible.command == "echo hello"
    assert commands == ["echo secret", "echo hello"]


def test_rpc_bash_empty_command_is_rejected(monkeypatch):
    stub_bash_tool(monkeypatch)

    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output))
        await server.dispatch({"id": "b", "type": "bash", "command": "  "})
        records = [json.loads(line) for line in output.getvalue().splitlines() if line]
        assert not records[-1]["success"]

    asyncio.run(check())
