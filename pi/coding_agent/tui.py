"""Multiline terminal editor, completion and modal selectors."""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text

from ..agent.agent import Agent

DEFAULT_KEYBINDINGS = {
    "submit": ["enter"], "newline": ["escape enter"], "abort": ["c-c"],
    "models": ["c-l"], "sessions": ["c-o"], "exit": ["c-d"],
    "complete": ["tab"],
}
THEMES = {
    "dark": {"prompt": "bold ansicyan", "bottom-toolbar": "bg:#202530 #dddddd", "completion-menu.completion.current": "bg:#227799 #ffffff"},
    "light": {"prompt": "bold ansiblue", "bottom-toolbar": "bg:#dddddd #222222", "completion-menu.completion.current": "bg:#227799 #ffffff"},
    "mono": {"prompt": "bold", "bottom-toolbar": "reverse", "completion-menu.completion.current": "reverse"},
}


def load_themes(agent_dir) -> dict:
    """Built-in themes plus user themes from <agent_dir>/themes/*.json."""
    themes = dict(THEMES)
    themes_dir = Path(agent_dir) / "themes"
    if themes_dir.is_dir():
        for path in sorted(themes_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
                themes[path.stem] = data
    return themes


class ExtensionAction:
    """Result marker for an extension-registered shortcut press."""

    def __init__(self, handler: Callable) -> None:
        self.handler = handler


class AgentCompleter(Completer):
    def __init__(self, commands: Callable[[], List[str]]) -> None:
        self.commands = commands
        self.paths = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and not any(char.isspace() for char in text):
            for name in sorted(set(self.commands())):
                if name.startswith(text):
                    yield Completion(name, start_position=-len(text))
            return
        word = text.rsplit(None, 1)[-1] if text.strip() else ""
        word = word[1:] if word.startswith("@") else word
        yield from self.paths.get_completions(Document(word), complete_event)


class TerminalUI:
    def __init__(self, agent: Agent, commands: Callable[[], List[str]], settings: Optional[dict] = None,
                 input=None, output=None, extra_bindings: Optional[List[tuple]] = None,
                 themes: Optional[dict] = None, full_screen: bool = False) -> None:
        self.agent = agent
        self.commands = commands
        self.settings = settings or {}
        self.themes = themes or dict(THEMES)
        self.running = False
        self.full_screen = full_screen
        self.transcript = ""
        self.status = ""
        self.widgets = {"above": [], "below": []}
        self.markdown_transform = None
        self.style = Style.from_dict(self.themes[self.settings.get("theme", "dark")])
        bindings = KeyBindings()
        keys = {**DEFAULT_KEYBINDINGS, **self.settings.get("keybindings", {})}

        def submit(event):
            event.current_buffer.validate_and_handle()

        def newline(event):
            event.current_buffer.insert_text("\n")

        def abort(event):
            if self.running:
                self.agent.abort()
            event.app.exit(result="")

        def models(event):
            if not self.running:
                event.app.exit(result="/model")

        def sessions(event):
            if not self.running:
                event.app.exit(result="/resume")

        def exit_editor(event):
            if self.running:
                self.agent.abort()
                event.app.exit(result="")
            else:
                event.app.exit(exception=EOFError())

        def complete(event):
            buffer = event.current_buffer
            if buffer.complete_state:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=False)

        handlers = {"submit": submit, "newline": newline, "abort": abort, "models": models,
                    "sessions": sessions, "exit": exit_editor, "complete": complete}

        def extension_handler(handler):
            def run(event):
                if not self.running:
                    event.app.exit(result=ExtensionAction(handler))
            return run

        registered = set()
        for action, shortcuts in keys.items():
            for shortcut in shortcuts:
                binding = bindings.add(*shortcut.split(), eager=True)
                binding(handlers[action])
                normalized = bindings.bindings[-1].keys
                if normalized in registered:
                    raise ValueError(f"Duplicate terminal shortcut: {shortcut}")
                registered.add(normalized)
        for _action, shortcuts, handler in (extra_bindings or []):
            for shortcut in shortcuts:
                binding = bindings.add(*shortcut.split(), eager=True)
                binding(extension_handler(handler))
                normalized = bindings.bindings[-1].keys
                if normalized in registered:
                    raise ValueError(f"Extension shortcut collides: {shortcut}")
                registered.add(normalized)
        self.session = PromptSession(
            multiline=True, key_bindings=bindings, history=InMemoryHistory(),
            completer=AgentCompleter(commands), complete_while_typing=True,
            style=self.style, bottom_toolbar=self.toolbar, input=input, output=output,
            prompt_continuation=lambda width, line, wrap: "... ", refresh_interval=0.2,
        )
        if full_screen:
            editor = self.session.app.layout.container
            conversation = FormattedTextControl(
                lambda: to_formatted_text(ANSI(self.transcript)),
                get_cursor_position=lambda: Point(
                    len(self.transcript.rsplit("\n", 1)[-1]), self.transcript.count("\n")))
            self.session.app.layout.container = HSplit([
                Window(conversation, wrap_lines=True),
                Window(FormattedTextControl(lambda: "\n".join(self.widgets["above"])),
                       dont_extend_height=True),
                editor,
                Window(FormattedTextControl(lambda: "\n".join(self.widgets["below"])),
                       dont_extend_height=True),
            ])
            self.session.app.full_screen = True

    def write(self, text: str) -> int:
        self.transcript += text
        self.session.app.invalidate()
        return len(text)

    def flush(self) -> None:
        self.session.app.invalidate()

    def isatty(self) -> bool:
        return True

    async def ui_request(self, request: dict):
        method = request["method"]
        if method == "notify":
            self.write(f"\n[{request.get('level', 'info')}] {request['message']}\n")
        elif method == "setStatus":
            self.status = request["text"]
        elif method == "setWidget":
            self.widgets[request["placement"]] = list(request["lines"])
        elif method == "setTitle":
            self.session.app.output.set_title(request["title"])
        elif method == "select":
            return await self.choose(request["title"], [(item, item) for item in request["options"]])
        elif method == "confirm":
            return await self.choose(request["message"], [("yes", "Yes"), ("no", "No")]) == "yes"
        else:
            raise ValueError(f"Unsupported terminal UI request: {method}")
        self.session.app.invalidate()

    def toolbar(self):
        state = "running; input steers the agent" if self.running else "ready"
        return [("class:bottom-toolbar", f" {self.agent.model.provider}/{self.agent.model.id} | {self.agent.thinking_level} | {state} | {self.status} ")]

    def set_theme(self, name: str):
        if name not in self.themes:
            raise ValueError(f"Unknown theme: {name}")
        self.style = Style.from_dict(self.themes[name])
        self.session.style = self.style

    async def read(self, running=False):
        self.running = running
        with create_app_session(input=self.session.app.input, output=self.session.app.output), \
                (nullcontext() if self.full_screen else patch_stdout()):
            result = await self.session.prompt_async([("class:prompt", "> ")])
        if self.full_screen and isinstance(result, str) and result:
            shown = result
            if self.markdown_transform is not None:
                shown = self.markdown_transform(result)
                if hasattr(shown, "__await__"):
                    shown = await shown
            self.write(f"\n> {shown}\n")
        return result

    async def choose(self, title: str, values: List[Tuple[str, str]]) -> Optional[str]:
        if not values:
            return None
        with create_app_session(input=self.session.app.input, output=self.session.app.output):
            return await radiolist_dialog(title=title, text="Use arrow keys; Tab moves to OK/Cancel; Enter confirms.",
                                          values=values, style=self.style).run_async()

    async def run(self, coroutine):
        task = asyncio.create_task(coroutine)
        reader = None
        try:
            while not task.done():
                reader = asyncio.create_task(self.read(running=True))
                done, _ = await asyncio.wait([task, reader], return_when=asyncio.FIRST_COMPLETED)
                if task in done:
                    break
                try:
                    text = reader.result()
                except (EOFError, KeyboardInterrupt):
                    self.agent.abort()
                    break
                if text:
                    self.agent.steer(text)
            await task
        finally:
            if reader and not reader.done():
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            if not task.done():
                self.agent.abort()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.running = False
