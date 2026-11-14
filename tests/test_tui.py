import asyncio

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.tui import AgentCompleter, TerminalUI, _TextDialog, _overlay_bounds
from pi.coding_agent.ts_extension import CustomViewFrame
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


def test_selector_starts_on_the_given_default():
    async def check(pipe):
        with create_app_session(input=pipe, output=DummyOutput()):
            ui = TerminalUI(Agent(resolve_model("faux")), lambda: [])
            task = asyncio.create_task(
                ui.choose("Advanced settings?", [("yes", "Yes"), ("no", "No")], default="no"))
            await asyncio.sleep(0.05)
            pipe.send_text("\r")
            assert await asyncio.wait_for(task, 3) == "no"
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_selector_arrows_and_escape():
    async def check(pipe):
        with create_app_session(input=pipe, output=DummyOutput()):
            ui = TerminalUI(Agent(resolve_model("faux")), lambda: [])
            task = asyncio.create_task(ui.choose("Models", [("a", "Model A"), ("b", "Model B")]))
            await asyncio.sleep(0.05)
            pipe.send_text("\x1b[B\r")
            assert await asyncio.wait_for(task, 3) == "b"
            task = asyncio.create_task(ui.choose("Models", [("a", "Model A")]))
            await asyncio.sleep(0.05)
            pipe.send_text("\x1b")
            assert await asyncio.wait_for(task, 3) is None
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_text_dialog_submit_default_and_cancel():
    async def check(pipe):
        with create_app_session(input=pipe, output=DummyOutput()):
            ui = TerminalUI(Agent(resolve_model("faux")), lambda: [])
            task = asyncio.create_task(ui.input("API Key", password=True))
            await asyncio.sleep(0.1)
            pipe.send_text("secret\r")
            assert await asyncio.wait_for(task, 3) == "secret"
            assert "secret" not in ui.session.history.get_strings()  # no input history
            task = asyncio.create_task(ui.input("Provider ID", default="my-gateway"))
            await asyncio.sleep(0.1)
            pipe.send_text("\r")
            assert await asyncio.wait_for(task, 3) == "my-gateway"
            task = asyncio.create_task(ui.input("Provider ID"))
            await asyncio.sleep(0.1)
            pipe.send_text("\x1b")
            assert await asyncio.wait_for(task, 3) is None
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_text_dialog_rejects_invalid_value_in_place():
    def provider_rule(value):
        text = value.strip()
        if ":" in text:
            raise ValueError("Provider ID must match ^[a-z][a-z0-9-]{0,63}$")
        return text

    async def check(pipe):
        with create_app_session(input=pipe, output=DummyOutput()):
            dialog = _TextDialog("Provider ID", "", False, Style.from_dict({}), provider_rule)
            task = asyncio.create_task(dialog.run())
            await asyncio.sleep(0.1)
            pipe.send_text("cn:deepseek-v4.1-flash\r")
            await asyncio.sleep(0.1)
            # Enter on an invalid value keeps the dialog open, keeps the text
            # and reports the rule instead of closing and re-prompting empty.
            assert not task.done()
            error = dialog.textfield.buffer.validation_error
            assert error is not None and "Provider ID" in error.message
            assert dialog.app.layout.current_buffer is dialog.textfield.buffer
            pipe.send_text("\x1b")
            assert await asyncio.wait_for(task, 3) is None

            dialog = _TextDialog("Provider ID", "  my-gateway  ", False,
                                 Style.from_dict({}), provider_rule)
            task = asyncio.create_task(dialog.run())
            await asyncio.sleep(0.1)
            pipe.send_text("\r")
            assert await asyncio.wait_for(task, 3) == "my-gateway"
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
        validate_settings({"theme": ""})
    with pytest.raises(ValueError):
        validate_settings({"keybindings": {"unknown": ["c-x"]}})


def test_custom_view_forwards_keys_mouse_and_done():
    class FakeSession:
        def __init__(self):
            self.done_event = asyncio.Event()
            self.result = None
            self.render_listener = None
            self.handle_listener = None
            self.options = {}
            self.bounds = None
            self.inputs = []
            self.mouse_events = []

        async def render(self, width, height=24):
            return CustomViewFrame(["component"], (0, 0))

        async def input(self, data, width):
            self.inputs.append(data)
            return CustomViewFrame([f"key:{data}"], None)

        async def mouse(self, event, width):
            self.mouse_events.append(event)
            return CustomViewFrame([f"mouse:{event['type']}"], None)

    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe, output=DummyOutput())
        session = FakeSession()
        task = asyncio.create_task(ui.custom_view(session))
        await asyncio.sleep(0.1)
        pipe.send_text("x")
        await asyncio.sleep(0.1)
        assert session.inputs == ["x"]

        pipe.send_text("\x1b[<0;10;5M")
        await asyncio.sleep(0.1)
        pipe.send_text("\x1b[<0;10;5m")
        await asyncio.sleep(0.1)
        assert [event["type"] for event in session.mouse_events] == ["press", "release", "click"]
        assert session.mouse_events[0]["button"] == "left"
        assert (session.mouse_events[0]["x"], session.mouse_events[0]["y"]) == (9, 4)

        session.result = "finished"
        session.done_event.set()
        assert await asyncio.wait_for(task, 3) == "finished"

    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_custom_view_suspends_the_running_editor():
    class FakeSession:
        def __init__(self):
            self.done_event = asyncio.Event()
            self.result = None
            self.render_listener = None
            self.handle_listener = None
            self.options = {}
            self.bounds = None
            self.inputs = []

        async def render(self, width, height=24):
            return CustomViewFrame(["component"], None)

        async def input(self, data, width):
            self.inputs.append(data)
            return CustomViewFrame([f"key:{data}"], None)

        async def mouse(self, event, width):
            return CustomViewFrame(["mouse"], None)

    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe, output=DummyOutput())
        session = FakeSession()

        async def work():
            await asyncio.sleep(0.05)
            view = asyncio.create_task(ui.custom_view(session))
            await asyncio.sleep(0.1)
            assert not ui.session.app.is_running
            pipe.send_text("y")
            await asyncio.sleep(0.1)
            session.result = "closed"
            session.done_event.set()
            assert await asyncio.wait_for(view, 3) == "closed"

        await asyncio.wait_for(ui.run(work()), 5)
        assert session.inputs == ["y"]

    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_custom_view_coalesces_alt_keys():
    class FakeSession:
        def __init__(self):
            self.done_event = asyncio.Event()
            self.result = None
            self.render_listener = None
            self.handle_listener = None
            self.options = {}
            self.bounds = None
            self.inputs = []

        async def render(self, width, height=24):
            return CustomViewFrame(["component"], None)

        async def input(self, data, width):
            self.inputs.append(data)
            return CustomViewFrame(["component"], None)

        async def mouse(self, event, width):
            return CustomViewFrame(["component"], None)

    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe, output=DummyOutput())
        session = FakeSession()
        task = asyncio.create_task(ui.custom_view(session))
        await asyncio.sleep(0.1)
        pipe.send_text("\x1bb")
        await asyncio.sleep(0.1)
        assert session.inputs == ["\x1bb"]

        pipe.send_text("\x1b")
        # prompt_toolkit holds a lone ESC for its sequence timeout (~0.5s)
        # before delivering it; the forwarder then holds it for another 40ms.
        await asyncio.sleep(0.8)
        assert session.inputs[-1] == "\x1b"

        session.done_event.set()
        await asyncio.wait_for(task, 3)

    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_overlay_bounds_follow_ts_options():
    frame = CustomViewFrame(["abc", "def"], overlay={"width": "50%", "anchor": "center"})
    assert _overlay_bounds(frame, 80, 24) == {"row": 11, "col": 20, "width": 40, "height": 2}

    frame = CustomViewFrame(["x"], overlay={"width": 10, "anchor": "bottom-right", "margin": 2})
    assert _overlay_bounds(frame, 80, 24) == {"row": 21, "col": 68, "width": 10, "height": 1}

    frame = CustomViewFrame(["x"], overlay={"minWidth": 20, "row": "25%", "col": "10%"})
    assert _overlay_bounds(frame, 80, 24) == {"row": 6, "col": 8, "width": 20, "height": 1}


def test_custom_view_overlay_reports_bounds():
    class FakeSession:
        def __init__(self):
            self.done_event = asyncio.Event()
            self.result = None
            self.render_listener = None
            self.handle_listener = None
            self.options = {"overlay": True}
            self.bounds = None

        async def render(self, width, height=24):
            return CustomViewFrame(["abc"], None, {"width": 10, "anchor": "top-left"})

        async def input(self, data, width):
            return await self.render(width, height=24)

        async def mouse(self, event, width):
            return await self.render(width, height=24)

    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe, output=DummyOutput())
        session = FakeSession()
        task = asyncio.create_task(ui.custom_view(session))
        await asyncio.sleep(0.1)
        assert session.bounds == {"row": 0, "col": 0, "width": 10, "height": 1}
        session.done_event.set()
        await asyncio.wait_for(task, 3)

    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_custom_editor_read_submits_text():
    class FakeEditor:
        def __init__(self):
            self.submitted = None
            self.action = None
            self.submit_event = asyncio.Event()
            self.render_listener = None
            self.inputs = []

        def reset(self):
            self.submitted = None
            self.action = None
            self.submit_event.clear()

        async def render(self, width, height=24):
            return CustomViewFrame(["editor"], (0, 0))

        async def input(self, data, width):
            self.inputs.append(data)
            if data == "\r":
                self.submitted = "hello"
                self.submit_event.set()
            return CustomViewFrame([f"editor:{data}"], None)

        async def mouse(self, event, width):
            return CustomViewFrame(["editor"], None)

    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe, output=DummyOutput())
        editor = FakeEditor()
        ui.set_custom_editor(editor)
        task = asyncio.create_task(ui.read())
        await asyncio.sleep(0.1)
        pipe.send_text("h")
        await asyncio.sleep(0.1)
        assert editor.inputs == ["h"]
        pipe.send_text("\r")
        assert await asyncio.wait_for(task, 3) == "hello"

    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))


def test_fullscreen_transcript_widgets_and_input():
    async def check(pipe):
        ui = TerminalUI(Agent(resolve_model("faux")), lambda: [], input=pipe,
                        output=DummyOutput(), full_screen=True)
        ui.write("assistant output\n")
        await ui.ui_request({"method": "setStatus", "text": "extension ready"})
        await ui.ui_request({"method": "setWidget", "placement": "above", "lines": ["progress"]})
        task = asyncio.create_task(ui.read())
        await asyncio.sleep(0.05)
        pipe.send_text("hello\r")
        assert await asyncio.wait_for(task, 3) == "hello"
        assert ui.session.app.full_screen
        assert "assistant output" in ui.transcript and "> hello" in ui.transcript
        assert ui.widgets["above"] == ["progress"]
        assert "extension ready" in ui.toolbar()[0][1]
    with create_pipe_input() as pipe:
        asyncio.run(check(pipe))
