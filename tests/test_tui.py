import asyncio

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.tui import AgentCompleter, TerminalUI
from pi.coding_agent.config import validate_settings


def test_command_and_path_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "example.py").write_text("pass", encoding="utf-8")
    completer = AgentCompleter(lambda: ["/model", "/models", "/review", "/skill:demo"])
    event = CompleteEvent(completion_requested=True)
    assert [c.text for c in completer.get_completions(Document("/mo"), event)] == ["/model", "/models"]
    assert any(c.text == "ample.py" for c in completer.get_completions(Document("@ex"), event))


def test_multiline_submit_and_history():
    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe, output=DummyOutput())
        task = asyncio.create_task(ui.read())
        await asyncio.sleep(0.05)
        pipe.send_text("first\x1b\r  second\r")
        assert await asyncio.wait_for(task, 3) == "first\n  second"
        assert "first\n  second" in ui.session.history.get_strings()
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_custom_submit_and_shortcut_validation():
    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], {"keybindings": {"submit": ["c-s"]}}, input=pipe, output=DummyOutput())
        task = asyncio.create_task(ui.read())
        await asyncio.sleep(0.05)
        pipe.send_text("custom\x13")
        assert await asyncio.wait_for(task, 3) == "custom"
        with pytest.raises(ValueError, match="Duplicate"):
            TerminalUI(ui.agent, lambda: [], {"keybindings": {"submit": ["tab"]}}, input=pipe, output=DummyOutput())
        ui.set_theme("light")
        assert ui.session.style == ui.style
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_selector_confirmation_and_cancel():
    async def check(pipe):
        with create_app_session(input=pipe, output=DummyOutput()):
            ui = TerminalUI(Agent(resolve_model("faux")), lambda: [])
            task = asyncio.create_task(ui.choose("Models", [("a", "Model A"), ("b", "Model B")]))
            await asyncio.sleep(0.05)
            pipe.send_text("\t\r")
            assert await asyncio.wait_for(task, 3) == "a"
            task = asyncio.create_task(ui.choose("Models", [("a", "Model A")]))
            await asyncio.sleep(0.05)
            pipe.send_text("\t\t\r")
            assert await asyncio.wait_for(task, 3) is None
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_abort_during_run():
    async def check(pipe):
        agent = Agent(resolve_model("faux"))
        stopped = asyncio.Event()
        agent.abort = stopped.set
        ui = TerminalUI(agent, lambda: [], input=pipe, output=DummyOutput())
        task = asyncio.create_task(ui.run(stopped.wait()))
        await asyncio.sleep(0.05)
        pipe.send_text("\x03")
        await asyncio.wait_for(task, 3)
        assert stopped.is_set() and not ui.running
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_invalid_theme_and_bindings():
    with pytest.raises(ValueError):
        validate_settings({"theme": "invalid"})
    with pytest.raises(ValueError):
        validate_settings({"keybindings": {"unknown": ["c-x"]}})
