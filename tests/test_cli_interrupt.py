"""Main-task cancellation must abort a prompt without losing its output."""

import asyncio
import io
import json

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.cli import CliRenderer, async_main, run_prompt
from pi.coding_agent.output_guard import restore_stdout


def test_cancelled_prompt_aborts_and_allows_next_prompt(monkeypatch, capsys):
    async def scenario():
        agent = Agent(resolve_model("faux"))
        started = asyncio.Event()
        aborted = asyncio.Event()
        finished = []

        async def prompt(text, images=None):
            started.set()
            await aborted.wait()
            finished.append(text)

        monkeypatch.setattr(agent, "prompt", prompt)
        monkeypatch.setattr(agent, "abort", aborted.set)
        running = asyncio.create_task(run_prompt(agent, "first", CliRenderer(), None))
        await started.wait()
        running.cancel()
        await asyncio.wait_for(running, 2)
        assert finished == ["first"]
        assert aborted.is_set()
        await run_prompt(agent, "next", CliRenderer(), None)
        assert finished == ["first", "next"]

    asyncio.run(scenario())
    assert "[aborted]" in capsys.readouterr().out


def test_json_cancellation_emits_error_and_settled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PI_OFFLINE", "1")
    output = io.StringIO()
    monkeypatch.setattr("pi.coding_agent.cli.raw_stdout", lambda: output)

    async def scenario():
        started = asyncio.Event()
        aborted = asyncio.Event()

        async def prompt(self, text, images=None):
            started.set()
            await aborted.wait()

        monkeypatch.setattr(Agent, "prompt", prompt)
        monkeypatch.setattr(Agent, "abort", lambda self: aborted.set())
        running = asyncio.create_task(async_main([
            "--provider", "faux", "--mode", "json", "--no-session",
            "--agent-dir", str(tmp_path / "agent"), "hello",
        ]))
        await asyncio.wait_for(started.wait(), 5)
        running.cancel()
        assert await asyncio.wait_for(running, 5) == 1
        assert aborted.is_set()

    try:
        asyncio.run(scenario())
    finally:
        restore_stdout()
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"type": "run_error", "error": "Request aborted"}, {"type": "agent_settled"},
    ]
