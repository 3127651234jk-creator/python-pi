"""Modal dialogs must own input while a fullscreen steering reader is active."""

import asyncio

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.tui import TerminalUI


@pytest.mark.parametrize("kind", ["choose", "input", "cancel"])
def test_dialog_suspends_and_resumes_streaming_reader(kind):
    async def scenario(pipe):
        agent = Agent(resolve_model("faux"))
        ui = TerminalUI(agent, lambda: [], input=pipe, output=DummyOutput(), full_screen=True)
        steered = asyncio.Event()
        messages = []

        def steer(text):
            messages.append(text)
            steered.set()

        agent.steer = steer

        async def work():
            while not ui.session.app.is_running:
                await asyncio.sleep(0.01)
            dialog = asyncio.create_task(ui.choose("Select", [("ok", "OK")]) if kind == "choose"
                                         else ui.input("Value"))
            await asyncio.sleep(0.1)
            assert ui._suspend_read
            assert not ui.session.app.is_running
            pipe.send_text("\r" if kind == "choose" else "\x1b" if kind == "cancel" else "value\r")
            assert await asyncio.wait_for(dialog, 3) == (
                "ok" if kind == "choose" else None if kind == "cancel" else "value")
            while not ui.session.app.is_running:
                await asyncio.sleep(0.01)
            assert not ui._suspend_read
            pipe.send_text("next\r")
            await asyncio.wait_for(steered.wait(), 3)

        await asyncio.wait_for(ui.run(work()), 5)
        assert messages == ["next"]

    with create_pipe_input() as pipe:
        asyncio.run(scenario(pipe))
