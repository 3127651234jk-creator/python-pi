"""Multiline terminal editor, completion and modal selectors."""

import asyncio
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.validation import ValidationError, Validator
from prompt_toolkit.widgets import Button, Dialog, TextArea, ValidationToolbar

from ..agent.agent import Agent
from .custom_api_form import CustomApiFormApp, CustomApiFormResult, SaveForm, ValidateForm

DEFAULT_KEYBINDINGS = {
    "submit": ["enter"], "newline": ["escape enter"], "abort": ["c-c"],
    "models": ["c-l"], "sessions": ["c-o"], "exit": ["c-d"],
    "complete": ["tab"], "save": ["c-s"],
}
THEMES = {
    "dark": {
        "prompt": "bold ansicyan", "bottom-toolbar": "bg:#202530 #dddddd",
        "completion-menu.completion.current": "bg:#227799 #ffffff",
    },
    "light": {
        "prompt": "bold ansiblue", "bottom-toolbar": "bg:#dddddd #222222",
        "completion-menu.completion.current": "bg:#227799 #ffffff",
        "dialog": "bg:#f7f8fb #23262f",
        "dialog.border": "bg:#f7f8fb #b6bdd0",
        "dialog.title": "bg:#f7f8fb #1d5fd0 bold",
        "dialog.hint": "bg:#f7f8fb #6f7689",
        "dialog.key": "bg:#f7f8fb #34417a bold",
        "dialog.dim": "bg:#f7f8fb #c3c9d8",
        "dialog.marker": "bg:#f7f8fb #6f7689",
        "dialog.text": "bg:#f7f8fb #23262f",
        "dialog.selected": "bg:#dbe6ff #0d1c3d bold",
        "dialog.selected.marker": "bg:#dbe6ff #1d5fd0 bold",
        "dialog.button": "bg:#e8ebf4 #23262f",
        "dialog.button.focused": "bg:#1d5fd0 #ffffff bold",
        "dialog.group": "bg:#f7f8fb #1d5fd0 bold",
        "dialog.label": "bg:#f7f8fb #23262f",
        "dialog.label.focused": "bg:#f7f8fb #0d1c3d bold underline",
        "dialog.field": "bg:#eceff7 #23262f",
        "dialog.field.focused": "bg:#dbe6ff #0d1c3d underline",
        "dialog.error": "bg:#f7f8fb #b3261e",
        "dialog.warning": "bg:#f7f8fb #8a5a00",
        "dialog.status": "bg:#f7f8fb #1d5fd0",
        "dialog.separator": "bg:#f7f8fb #c3c9d8",
        "dialog.checkbox.row": "bg:#f7f8fb #23262f",
        "dialog.checkbox": "#23262f",
        "dialog.checkbox.checked": "#1d5fd0 bold",
        "dialog.checkbox.focused": "bg:#dbe6ff #0d1c3d",
    },
    "mono": {
        "prompt": "bold", "bottom-toolbar": "reverse",
        "completion-menu.completion.current": "reverse",
        "dialog": "", "dialog.border": "", "dialog.title": "bold",
        "dialog.hint": "", "dialog.key": "bold", "dialog.dim": "",
        "dialog.marker": "", "dialog.text": "",
        "dialog.selected": "reverse", "dialog.selected.marker": "reverse bold",
        "dialog.button": "", "dialog.button.focused": "reverse bold",
        "dialog.group": "bold", "dialog.label": "", "dialog.label.focused": "bold underline",
        "dialog.field": "", "dialog.field.focused": "reverse",
        "dialog.error": "bold", "dialog.warning": "", "dialog.status": "",
        "dialog.separator": "", "dialog.checkbox.row": "",
        "dialog.checkbox": "", "dialog.checkbox.checked": "bold",
        "dialog.checkbox.focused": "reverse",
    },
}

# Dialog palette shared by the styled picker. Themes may override these classes;
# user themes written before the picker existed fall back to the dark palette.
DIALOG_STYLES = {
    "dialog": "bg:#191c26 #c6cbe0",
    "dialog.border": "bg:#191c26 #3f4866",
    "dialog.title": "bg:#191c26 #8ab4ff bold",
    "dialog.hint": "bg:#191c26 #6b7391",
    "dialog.key": "bg:#191c26 #aab4d4 bold",
    "dialog.dim": "bg:#191c26 #2c3349",
    "dialog.marker": "bg:#191c26 #6b7391",
    "dialog.text": "bg:#191c26 #c6cbe0",
    "dialog.selected": "bg:#26304a #eef2ff bold",
    "dialog.selected.marker": "bg:#26304a #8ab4ff bold",
    "dialog.button": "bg:#222738 #c6cbe0",
    "dialog.button.focused": "bg:#8ab4ff #14171f bold",
    "dialog.group": "bg:#191c26 #8ab4ff bold",
    "dialog.label": "bg:#191c26 #c6cbe0",
    "dialog.label.focused": "bg:#191c26 #ffffff bold underline",
    "dialog.field": "bg:#222738 #e6eaf5",
    "dialog.field.focused": "bg:#2a3552 #ffffff underline",
    "dialog.error": "bg:#191c26 #ff8f8f",
    "dialog.warning": "bg:#191c26 #ffc66d",
    "dialog.status": "bg:#191c26 #8ab4ff",
    "dialog.separator": "bg:#191c26 #3f4866",
    "dialog.checkbox.row": "bg:#191c26 #c6cbe0",
    "dialog.checkbox": "#c6cbe0",
    "dialog.checkbox.checked": "#8ab4ff bold",
    "dialog.checkbox.focused": "bg:#26304a #eef2ff",
}


def _theme_styles(theme: dict) -> dict:
    """Theme classes with the dialog palette filled in as defaults."""
    return {**DIALOG_STYLES, **theme}


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


_SGR_MOUSE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")


def _mouse_event_from_sequence(data: str, width: int, height: int) -> Optional[dict]:
    """Parse a raw mouse sequence into the TS ``TuiMouseEvent`` shape.

    prompt_toolkit clamps control-level mouse coordinates to the rendered
    content, so interactive views parse the SGR/X10 sequence directly.
    """
    match = _SGR_MOUSE.match(data)
    if match:
        code = int(match.group(1))
        x = max(0, int(match.group(2)) - 1)
        y = max(0, int(match.group(3)) - 1)
        released = match.group(4) == "m"
    elif data.startswith("\x1b[M") and len(data) >= 6:
        code = ord(data[3]) - 32
        x = max(0, ord(data[4]) - 32 - 1)
        y = max(0, ord(data[5]) - 32 - 1)
        released = (code & 3) == 3
    else:
        return None

    base = code & 3
    motion = bool(code & 32)
    wheel = bool(code & 64)
    button = {0: "left", 1: "middle", 2: "right"}.get(base, "none")
    if wheel:
        kind = "wheel"
    elif released:
        kind = "release"
    elif motion:
        kind = "move" if base == 3 else "drag"
    else:
        kind = "press"

    event = {
        "type": kind, "button": button, "x": x, "y": y,
        "screenX": x, "screenY": y, "width": width, "height": height,
        "shift": bool(code & 4), "alt": bool(code & 8), "ctrl": bool(code & 16),
    }
    if wheel:
        event["wheelDelta"] = -1 if base == 0 else 1
    return event


class _KeyForwarder:
    """Coalesce ESC-prefixed keys and push raw key data onto a queue.

    prompt_toolkit reports ESC followed by a character as two separate key
    presses; TS components expect the combined ``\\x1b<char>`` sequence for
    alt+key. Escape alone is delivered after a short hold.
    """

    HOLD_SECONDS = 0.04

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue
        self._timer: Optional[asyncio.TimerHandle] = None

    def feed(self, data: str) -> None:
        if not data:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
            if not data.startswith("\x1b"):
                self.queue.put_nowait(("key", "\x1b" + data))
                return
            self.queue.put_nowait(("key", "\x1b"))
        if data == "\x1b":
            self._timer = asyncio.get_running_loop().call_later(self.HOLD_SECONDS, self._flush)
            return
        self.queue.put_nowait(("key", data))

    def _flush(self) -> None:
        self._timer = None
        self.queue.put_nowait(("key", "\x1b"))

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def _size_value(value, reference: int) -> Optional[int]:
    """Resolve a TS SizeValue (number or "N%") against a reference dimension."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.endswith("%"):
        try:
            return int(float(value[:-1]) * reference / 100)
        except ValueError:
            return None
    return None


def _overlay_margins(value) -> Tuple[int, int, int, int]:
    if isinstance(value, dict):
        return tuple(int(value.get(side) or 0) for side in ("top", "right", "bottom", "left"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        margin = int(value)
        return margin, margin, margin, margin
    return 0, 0, 0, 0


def _overlay_bounds(frame, width: int, height: int) -> dict:
    """Compute the overlay rectangle from TS OverlayOptions."""
    options = frame.overlay or {}
    top, right, bottom, left = _overlay_margins(options.get("margin"))
    available_width = max(1, width - left - right)
    available_height = max(1, height - top - bottom)

    natural_width = max((get_cwidth(line) for line in frame.lines), default=0)
    box_width = _size_value(options.get("width"), width) or natural_width or available_width
    box_width = max(box_width, int(options.get("minWidth") or 0))
    box_width = max(1, min(available_width, box_width))

    box_height = max(1, min(available_height, len(frame.lines) or 1))
    max_height = _size_value(options.get("maxHeight"), height)
    if max_height is not None:
        box_height = max(1, min(box_height, max_height))

    anchor = str(options.get("anchor") or "center")
    col = _size_value(options.get("col"), width)
    if col is None:
        if anchor.endswith("left"):
            col = left
        elif anchor.endswith("right"):
            col = width - right - box_width
        else:
            col = left + max(0, (available_width - box_width) // 2)
    row = _size_value(options.get("row"), height)
    if row is None:
        if anchor.startswith("top"):
            row = top
        elif anchor.startswith("bottom"):
            row = height - bottom - box_height
        else:
            row = top + max(0, (available_height - box_height) // 2)

    col = max(0, min(width - box_width, col + int(options.get("offsetX") or 0)))
    row = max(0, min(height - box_height, row + int(options.get("offsetY") or 0)))
    return {"row": row, "col": col, "width": box_width, "height": box_height}


def _dialog_text(value) -> str:
    """Single-line label text: control characters would break the dialog layout."""
    return "".join(" " if ord(ch) < 32 or ord(ch) == 127 else ch for ch in str(value))


def _clip_text(text: str, width: int) -> str:
    """Truncate to ``width`` columns, appending an ellipsis when cut."""
    if width <= 0:
        return ""
    if get_cwidth(text) <= width:
        return text
    out, used = [], 0
    for char in text:
        size = get_cwidth(char)
        if used + size > width - 1:
            break
        out.append(char)
        used += size
    return "".join(out) + "…"


def _fit_fragments(fragments, width: int):
    """Clip a formatted-text run to ``width`` display columns."""
    out, used = [], 0
    for style, text in fragments:
        room = width - used
        if room <= 0:
            break
        piece = _clip_text(text, room)
        out.append((style, piece))
        used += get_cwidth(piece)
    return out


class _ChoiceDialog:
    """Centered card-style single-choice picker used by :meth:`TerminalUI.choose`.

    Keys mirror the old ``radiolist_dialog``: arrows move inside the list, Tab
    cycles list -> OK -> Cancel, Enter confirms the focused target and Escape
    cancels. Mouse clicks work too.
    """

    FOCUS_ORDER = ("list", "ok", "cancel")

    def __init__(self, title, values, style, default=None) -> None:
        self.title = _dialog_text(title) or "Select"
        self.values = list(values)
        self.style = style
        self.index = next((index for index, (value, _label) in enumerate(self.values)
                           if value == default), 0)
        self.top = 0
        self.focus = "list"
        self._hit: dict = {}
        self._bindings = self._create_bindings()

    def _create_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        def move(step: int) -> None:
            self.focus = "list"
            self.index = max(0, min(len(self.values) - 1, self.index + step))

        def cycle(step: int) -> None:
            position = self.FOCUS_ORDER.index(self.focus)
            self.focus = self.FOCUS_ORDER[(position + step) % len(self.FOCUS_ORDER)]

        @bindings.add("up")
        def _up(event) -> None:
            move(-1)

        @bindings.add("down")
        def _down(event) -> None:
            move(1)

        @bindings.add("pageup")
        def _page_up(event) -> None:
            move(-5)

        @bindings.add("pagedown")
        def _page_down(event) -> None:
            move(5)

        @bindings.add("home")
        def _home(event) -> None:
            move(-len(self.values))

        @bindings.add("end")
        def _end(event) -> None:
            move(len(self.values))

        @bindings.add("tab")
        def _tab(event) -> None:
            cycle(1)

        @bindings.add("s-tab")
        def _shift_tab(event) -> None:
            cycle(-1)

        @bindings.add("left")
        @bindings.add("right")
        def _nudge_focus(event) -> None:
            if self.focus in ("ok", "cancel"):
                self.focus = "cancel" if self.focus == "ok" else "ok"

        @bindings.add("enter")
        def _enter(event) -> None:
            if self.focus == "cancel":
                self._finish(None)
            else:
                self._finish(self.values[self.index][0])

        @bindings.add("escape")
        @bindings.add("c-c")
        def _cancel(event) -> None:
            self._finish(None)

        @bindings.add(Keys.Vt100MouseEvent)
        def _mouse(event) -> None:
            size = get_app().output.get_size()
            payload = _mouse_event_from_sequence(event.data, size.columns, size.rows)
            if payload is None or payload["type"] != "press":
                return
            row, col = payload["y"], payload["x"]
            hit = self._hit
            if hit.get("list_top") is not None and hit["list_top"] <= row < hit["list_bottom"]:
                self.focus = "list"
                self.index = min(len(self.values) - 1, self.top + row - hit["list_top"])
            elif hit.get("ok_row") == row and hit["ok_start"] <= col < hit["ok_end"]:
                self._finish(self.values[self.index][0])
            elif hit.get("cancel_row") == row and hit["cancel_start"] <= col < hit["cancel_end"]:
                self._finish(None)

        return bindings

    def _finish(self, result) -> None:
        get_app().exit(result=result)

    def _content_line(self, fragments, content_width: int, pad_style: str = "class:dialog"):
        used = sum(get_cwidth(text) for _, text in fragments)
        line = [("class:dialog.border", "│"), (pad_style, " ")]
        line.extend(fragments)
        line.append((pad_style, " " * max(0, content_width + 1 - used)))
        line.append(("class:dialog.border", "│"))
        return line

    def _render(self):
        size = get_app().output.get_size()
        columns, rows = max(20, size.columns), max(10, size.rows)

        labels = [_dialog_text(label) for _, label in self.values]
        hint = [
            ("class:dialog.key", "↑↓"), ("class:dialog.hint", " select   "),
            ("class:dialog.key", "Tab"), ("class:dialog.hint", " focus   "),
            ("class:dialog.key", "Enter"), ("class:dialog.hint", " confirm   "),
            ("class:dialog.key", "Esc"), ("class:dialog.hint", " cancel"),
        ]
        hint_width = sum(get_cwidth(text) for _, text in hint)
        content_width = max(46, get_cwidth(self.title) + 1,
                            max((get_cwidth(label) for label in labels), default=0) + 5,
                            hint_width + 2)
        content_width = min(content_width, 76, columns - 4)
        width = content_width + 4

        visible = max(1, min(len(self.values), rows - 9))
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + visible:
            self.top = self.index - visible + 1
        self.top = max(0, min(self.top, len(self.values) - visible))

        lines = [[("class:dialog.border", "╭" + "─" * (width - 2) + "╮")]]
        lines.append(self._content_line([], content_width))
        count = f"{self.index + 1}/{len(self.values)}" if len(self.values) > visible else ""
        title = _clip_text(self.title, content_width - (get_cwidth(count) + 2 if count else 0))
        title_fragments = [("class:dialog.title", title)]
        if count:
            title_fragments.append(("class:dialog", " " * (content_width - get_cwidth(title) - get_cwidth(count))))
            title_fragments.append(("class:dialog.hint", count))
        lines.append(self._content_line(title_fragments, content_width))
        lines.append(self._content_line([("class:dialog.dim", "─" * content_width)], content_width))
        lines.append(self._content_line([], content_width))

        for offset, label in enumerate(labels[self.top:self.top + visible]):
            selected = self.top + offset == self.index
            marker_style = "class:dialog.selected.marker" if selected else "class:dialog.marker"
            text_style = "class:dialog.selected" if selected else "class:dialog.text"
            text = _clip_text(label, content_width - 4)
            lines.append(self._content_line(
                [(marker_style, " ●  " if selected else " ○  "), (text_style, text)],
                content_width, pad_style=text_style))

        lines.append(self._content_line([], content_width))
        if hint_width > content_width:
            hint = _fit_fragments(hint, content_width)
        lines.append(self._content_line(hint, content_width))

        ok_style = "class:dialog.button.focused" if self.focus == "ok" else "class:dialog.button"
        cancel_style = "class:dialog.button.focused" if self.focus == "cancel" else "class:dialog.button"
        ok_text, cancel_text = "[ OK ]", "[ Cancel ]"
        buttons_width = get_cwidth(ok_text) + 2 + get_cwidth(cancel_text)
        lines.append(self._content_line([
            ("class:dialog", " " * max(0, content_width - buttons_width)),
            (ok_style, ok_text), ("class:dialog", "  "), (cancel_style, cancel_text),
        ], content_width))
        lines.append([("class:dialog.border", "╰" + "─" * (width - 2) + "╯")])

        left = max(0, (columns - width) // 2)
        top = max(0, (rows - len(lines)) // 2)
        list_top = top + 5
        ok_row = top + len(lines) - 2
        ok_start = left + 2 + content_width - buttons_width
        self._hit = {
            "list_top": list_top, "list_bottom": list_top + visible,
            "ok_row": ok_row, "ok_start": ok_start, "ok_end": ok_start + get_cwidth(ok_text),
            "cancel_row": ok_row, "cancel_start": ok_start + get_cwidth(ok_text) + 2,
            "cancel_end": ok_start + buttons_width,
        }

        screen = [[("", " " * columns)] for _ in range(top)]
        for line in lines:
            screen.append([("", " " * left)] + line
                          + [("", " " * max(0, columns - left - width))])
        while len(screen) < rows:
            screen.append([("", " " * columns)])

        formatted = []
        for number, line in enumerate(screen):
            if number:
                formatted.append(("", "\n"))
            formatted.extend(line)
        return formatted

    async def run(self):
        app = Application(
            layout=Layout(HSplit([
                Window(FormattedTextControl(self._render), wrap_lines=False, dont_extend_height=True),
            ])),
            key_bindings=self._bindings, full_screen=True, mouse_support=True, style=self.style,
        )
        return await app.run_async()


class _FormValidator(Validator):
    """Adapt a form rule to prompt_toolkit.

    The rule returns the normalized value and raises ``ValueError`` with a
    user-facing message; the message is rendered inside the dialog instead of
    being printed behind it while the dialog reopens.
    """

    def __init__(self, check: Callable[[str], str]) -> None:
        self.check = check

    def validate(self, document: Document) -> None:
        try:
            self.check(document.text)
        except ValueError as exc:
            raise ValidationError(message=str(exc))


class _TextDialog:
    """Centered card-style single-line input used by :meth:`TerminalUI.input`.

    Enter submits, Escape or Ctrl+C cancels, Cancel is clickable. The buffer
    carries no history, so password entry never lands in an input history.
    """

    def __init__(self, title, default: str, password: bool, style, validate=None) -> None:
        self.validate = validate
        self.textfield = TextArea(
            text=default, multiline=False, password=password,
            validator=_FormValidator(validate) if validate else None)
        cancel_button = Button(text="Cancel", handler=self._cancel)
        ok_button = Button(text="OK", handler=self._accept)
        body = HSplit([self.textfield, ValidationToolbar()] if validate else [self.textfield])
        dialog = Dialog(title=_dialog_text(title), body=body,
                        buttons=[ok_button, cancel_button], with_background=True)
        bindings = KeyBindings()

        @bindings.add("enter", eager=True)
        def _submit(event) -> None:
            if event.app.layout.has_focus(cancel_button):
                self._cancel()
            else:
                self._accept()

        @bindings.add("escape")
        @bindings.add("c-c")
        def _cancel(event) -> None:
            self._cancel()

        self.app = Application(layout=Layout(dialog), key_bindings=bindings,
                               mouse_support=True, style=style, full_screen=True)

    def _accept(self) -> None:
        if self.validate is not None:
            if not self.textfield.buffer.validate():
                # Keep the value on screen and show the reason under the field.
                self.app.layout.focus(self.textfield)
                self.app.invalidate()
                return
            get_app().exit(result=self.validate(self.textfield.text))
            return
        get_app().exit(result=self.textfield.text)

    def _cancel(self) -> None:
        get_app().exit(result=None)

    async def run(self):
        return await self.app.run_async()


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
        self._transcript_version = 0
        self._formatted_version = -1
        self._formatted_transcript = []
        # True while an extension custom view owns the terminal: the input loop
        # parks instead of restarting the editor prompt.
        self._suspend_read = False
        self._resume_event = asyncio.Event()
        self.custom_editor = None
        self.status = ""
        self.widgets = {"above": [], "below": []}
        self.widget_entries: dict = {}  # key -> (placement, lines)
        self.header_lines: List[str] = []
        self.footer_lines: List[str] = []
        self.markdown_transform = None
        self.style = Style.from_dict(_theme_styles(self.themes[self.settings.get("theme", "dark")]))
        bindings = KeyBindings()
        keys = {**DEFAULT_KEYBINDINGS, **self.settings.get("keybindings", {})}
        # "save" belongs to modal forms (the custom API page); the prompt has
        # no save target, so it is not registered here.
        self.form_keybindings = keys

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
            if action == "save":
                continue
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
                self.formatted_transcript,
                get_cursor_position=lambda: Point(
                    len(self.transcript.rsplit("\n", 1)[-1]), self.transcript.count("\n")))
            self.session.app.layout.container = HSplit([
                Window(FormattedTextControl(lambda: "\n".join(self.header_lines)),
                       dont_extend_height=True),
                Window(conversation, wrap_lines=True),
                Window(FormattedTextControl(lambda: "\n".join(self.widgets["above"])),
                       dont_extend_height=True),
                editor,
                Window(FormattedTextControl(lambda: "\n".join(self.widgets["below"])),
                       dont_extend_height=True),
                Window(FormattedTextControl(lambda: "\n".join(self.footer_lines)),
                       dont_extend_height=True),
            ])
            self.session.app.full_screen = True

    def formatted_transcript(self):
        """Parsed transcript, rebuilt only when new text was written."""
        if self._formatted_version != self._transcript_version:
            self._formatted_transcript = to_formatted_text(ANSI(self.transcript))
            self._formatted_version = self._transcript_version
        return self._formatted_transcript

    def _sync_widgets(self) -> None:
        widgets = {"above": [], "below": []}
        for placement, lines in self.widget_entries.values():
            widgets[placement].extend(lines)
        self.widgets = widgets

    def write(self, text: str) -> int:
        self.transcript += text
        self._transcript_version += 1
        self.session.app.invalidate()
        return len(text)

    def width(self) -> int:
        """Terminal width used to size rendered images and extension views."""
        return self.session.app.output.get_size().columns

    def height(self) -> int:
        """Terminal height used by extension custom views."""
        return self.session.app.output.get_size().rows

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
            placement = request.get("placement", "above")
            if placement not in ("above", "below"):
                raise ValueError(f"Unsupported widget placement: {placement}")
            key = str(request.get("key", ""))
            lines = list(request.get("lines") or [])
            if lines:
                self.widget_entries[key] = (placement, lines)
            else:
                self.widget_entries.pop(key, None)
            self._sync_widgets()
        elif method == "setHeader":
            self.header_lines = list(request.get("lines") or [])
        elif method == "setFooter":
            self.footer_lines = list(request.get("lines") or [])
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
        self.style = Style.from_dict(_theme_styles(self.themes[name]))
        self.session.style = self.style

    async def read(self, running=False):
        self.running = running
        used_custom = self.custom_editor is not None
        if used_custom:
            try:
                result = await self._read_custom_editor()
            except (OSError, RuntimeError) as exc:
                print(f"Custom editor failed ({exc}); using the default editor", file=sys.stderr)
                self.custom_editor = None
                used_custom = False
                result = await self._read_prompt()
        else:
            result = await self._read_prompt()
        if isinstance(result, str) and result:
            shown = result
            if self.markdown_transform is not None:
                shown = self.markdown_transform(result)
                if hasattr(shown, "__await__"):
                    shown = await shown
            if self.full_screen:
                self.write(f"\n> {shown}\n")
            elif used_custom:
                print(f"\n> {shown}\n", flush=True)
        return result

    async def _read_prompt(self):
        with create_app_session(input=self.session.app.input, output=self.session.app.output), \
                (nullcontext() if self.full_screen else patch_stdout()):
            return await self.session.prompt_async([("class:prompt", "> ")])

    def set_custom_editor(self, session) -> None:
        """Swap the prompt editor for an extension component; None restores it."""
        self.custom_editor = session
        if self.session.app.is_running:
            self.session.app.exit(result="")
        self.session.app.invalidate()

    async def _read_custom_editor(self) -> str:
        """One input read driven by an extension's ``setEditorComponent``."""
        session = self.custom_editor
        session.reset()
        if self.session.app.is_running:
            self.session.app.exit(result="")
            while self.session.app.is_running:
                await asyncio.sleep(0.01)
        width, height = self.width(), self.height()
        initial = await session.render(width, height)
        state = {"frame": initial, "fragments": []}
        app_ref: dict = {}

        def text():
            return state["fragments"]

        def cursor_position():
            cursor = state["frame"].cursor
            return Point(cursor[0], cursor[1]) if cursor else None

        def apply(frame) -> None:
            state["frame"] = frame
            state["fragments"] = to_formatted_text(ANSI("\n".join(frame.lines)))
            if "app" in app_ref:
                app_ref["app"].invalidate()

        queue: asyncio.Queue = asyncio.Queue()

        async def worker() -> None:
            while True:
                kind, payload = await queue.get()
                try:
                    if kind == "render":
                        frame = await session.render(width, height)
                    elif kind == "key":
                        frame = await session.input(payload, width)
                    else:
                        frame = await session.mouse(payload, width)
                except Exception:  # noqa: BLE001 - a dead bridge must not trap the user
                    if "app" in app_ref:
                        app_ref["app"].exit(result=None)
                    return
                apply(frame)

        async def watch_submit() -> None:
            await session.submit_event.wait()
            if "app" in app_ref:
                app_ref["app"].exit(result=True)

        session.render_listener = lambda: queue.put_nowait(("render", None))

        bindings = KeyBindings()
        forwarder = _KeyForwarder(queue)

        @bindings.add(Keys.Any)
        def forward(event) -> None:
            forwarder.feed(
                event.data or "".join(key.data for key in event.key_sequence if key.data))

        @bindings.add(Keys.Vt100MouseEvent)
        def forward_mouse(event) -> None:
            payload = _mouse_event_from_sequence(event.data, width, height)
            if payload is None:
                return
            editor_rows = max(1, len(state["frame"].lines))
            if payload["y"] < height - editor_rows:
                return
            payload["y"] -= height - editor_rows
            payload["screenY"] = payload["y"]
            payload["height"] = editor_rows
            queue.put_nowait(("mouse", payload))

        editor_control = FormattedTextControl(text, focusable=True, get_cursor_position=cursor_position)
        layout = Layout(HSplit([
            Window(FormattedTextControl(self.formatted_transcript), wrap_lines=True),
            Window(editor_control, wrap_lines=False, dont_extend_height=True),
        ]))
        apply(initial)
        app = Application(
            layout=layout, key_bindings=bindings, full_screen=True, mouse_support=True,
            style=self.style, input=self.session.app.input, output=self.session.app.output,
        )
        app_ref["app"] = app
        workers = [asyncio.create_task(worker()), asyncio.create_task(watch_submit())]
        try:
            with create_app_session(input=self.session.app.input, output=self.session.app.output):
                await app.run_async(
                    pre_run=lambda: app.exit(result=True) if session.submit_event.is_set() else None)
        finally:
            forwarder.close()
            session.render_listener = None
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        if session.action == "exit":
            if self.running:
                self.agent.abort()
                return ""
            raise EOFError()
        if session.action == "escape":
            if self.running:
                self.agent.abort()
            return ""
        return session.submitted or ""

    async def custom_view(self, session) -> Any:
        """Host one ``ctx.ui.custom`` component.

        Renders the component in a dedicated full-screen view, forwards keys and
        mouse events to it, and closes when the extension calls ``done``. When
        the editor prompt is active (extension commands run while ``ui.run``
        streams), the prompt is suspended until the view closes.
        """
        suspended = self.session.app.is_running
        if suspended:
            self._suspend_read = True
            self._resume_event.clear()
            self.session.app.exit(result="")
            while self.session.app.is_running:
                await asyncio.sleep(0.01)
        try:
            return await self._run_custom_view(session)
        finally:
            if suspended:
                self._suspend_read = False
                self._resume_event.set()

    async def _run_custom_view(self, session) -> Any:
        width = self.width()
        height = self.height()
        initial = await session.render(width, height)
        state = {"frame": initial, "fragments": []}
        overlay_mode = bool(session.options.get("overlay"))

        def text():
            return state["fragments"]

        def cursor_position():
            cursor = state["frame"].cursor
            return Point(cursor[0], cursor[1]) if cursor else None

        control = FormattedTextControl(text, get_cursor_position=cursor_position)
        floating = None
        float_container = None
        if overlay_mode:
            floating = Float(Window(control, wrap_lines=False), top=0, left=0,
                             width=width, height=height)
            float_container = FloatContainer(content=Window(FormattedTextControl("")), floats=[floating])
            layout = Layout(float_container)
        else:
            layout = Layout(HSplit([Window(control, wrap_lines=False)]))

        queue: asyncio.Queue = asyncio.Queue()
        app_ref: dict = {}

        def apply(frame) -> None:
            state["frame"] = frame
            state["fragments"] = to_formatted_text(ANSI("\n".join(frame.lines)))
            if float_container is not None:
                bounds = _overlay_bounds(frame, width, height)
                session.bounds = bounds
                floating.top = bounds["row"]
                floating.left = bounds["col"]
                floating.width = bounds["width"]
                floating.height = bounds["height"]
                float_container.floats = [] if frame.hidden else [floating]
            if "app" in app_ref:
                app_ref["app"].invalidate()

        apply(initial)
        session.handle_listener = lambda action: session.finish(None) if action == "hide" else None

        async def worker() -> None:
            while True:
                kind, payload = await queue.get()
                try:
                    if kind == "render":
                        frame = await session.render(width, height)
                    elif kind == "key":
                        frame = await session.input(payload, width)
                    else:
                        frame = await session.mouse(payload, width)
                except Exception:  # noqa: BLE001 - a dead bridge must not trap the user
                    session.finish(session.result)
                    return
                apply(frame)

        async def watch_done() -> None:
            await session.done_event.wait()
            if "app" in app_ref:
                app_ref["app"].exit(result=session.result)

        session.render_listener = lambda: queue.put_nowait(("render", None))

        press: dict = {"point": None, "moved": False}

        bindings = KeyBindings()
        forwarder = _KeyForwarder(queue)

        @bindings.add(Keys.Any)
        def forward(event) -> None:
            forwarder.feed(
                event.data or "".join(key.data for key in event.key_sequence if key.data))

        @bindings.add(Keys.Vt100MouseEvent)
        def forward_mouse(event) -> None:
            payload = _mouse_event_from_sequence(
                event.data, width, max(1, len(state["frame"].lines)))
            if payload is None:
                return
            point = (payload["x"], payload["y"])
            if payload["type"] == "press":
                press["point"] = point
                press["moved"] = False
            elif payload["type"] in ("move", "drag") and press["point"] is not None:
                if press["point"] != point:
                    press["moved"] = True
            elif payload["type"] == "release":
                pressed, moved = press["point"], press["moved"]
                press["point"], press["moved"] = None, False
                queue.put_nowait(("mouse", payload))
                if pressed is not None and pressed == point and not moved:
                    queue.put_nowait(("mouse", dict(payload, type="click", clickCount=1)))
                return
            queue.put_nowait(("mouse", payload))

        app = Application(
            layout=layout,
            key_bindings=bindings, full_screen=True, mouse_support=True,
            style=self.style, input=self.session.app.input, output=self.session.app.output,
        )
        app_ref["app"] = app
        workers = [asyncio.create_task(worker()), asyncio.create_task(watch_done())]
        try:
            with create_app_session(input=self.session.app.input, output=self.session.app.output):
                return await app.run_async(
                    pre_run=lambda: app.exit(result=session.result) if session.done_event.is_set() else None)
        finally:
            forwarder.close()
            session.render_listener = None
            session.handle_listener = None
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _run_dialog(self, create_dialog: Callable) -> Any:
        """Give a modal dialog exclusive input while the steering reader waits."""
        suspended = self.session.app.is_running
        if suspended:
            self._suspend_read = True
            self._resume_event.clear()
        try:
            if suspended:
                self.session.app.exit(result="")
                while self.session.app.is_running:
                    await asyncio.sleep(0.01)
            with create_app_session(input=self.session.app.input, output=self.session.app.output):
                return await create_dialog().run()
        finally:
            if suspended:
                self._suspend_read = False
                self._resume_event.set()

    async def choose(self, title: str, values: List[Tuple[str, str]],
                     default: Optional[str] = None) -> Optional[str]:
        if not values:
            return None
        return await self._run_dialog(lambda: _ChoiceDialog(title, values, self.style, default))

    async def input(self, title: str, default: str = "", password: bool = False,
                    validate: Optional[Callable[[str], str]] = None) -> Optional[str]:
        """Modal single-line text entry; None when cancelled. The buffer keeps
        no history, so password entry never lands in an input history. With
        ``validate`` the dialog rejects an invalid value in place, keeps the
        text and shows the rule's error message."""
        return await self._run_dialog(lambda: _TextDialog(title, default, password, self.style, validate))

    async def custom_api_form(self, validate: ValidateForm,
                              save: SaveForm) -> Optional[CustomApiFormResult]:
        """Run the single-page custom API form.

        Uses the same suspend/resume path as extension views so the main
        editor prompt is never running underneath the form.
        """
        app = CustomApiFormApp(
            validate=validate, save=save, style=self.style,
            shortcuts=self.form_keybindings.get("save", ["c-s"]),
            mono=self.settings.get("theme", "dark") == "mono",
            input=self.session.app.input, output=self.session.app.output,
        )
        suspended = self.session.app.is_running
        if suspended:
            self._suspend_read = True
            self._resume_event.clear()
            self.session.app.exit(result="")
            while self.session.app.is_running:
                await asyncio.sleep(0.01)
        try:
            with create_app_session(input=self.session.app.input, output=self.session.app.output):
                return await app.run()
        finally:
            if suspended:
                self._suspend_read = False
                self._resume_event.set()

    async def run(self, coroutine):
        task = asyncio.create_task(coroutine)
        reader = None
        try:
            while not task.done():
                if self._suspend_read:
                    await self._resume_event.wait()
                    continue
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
