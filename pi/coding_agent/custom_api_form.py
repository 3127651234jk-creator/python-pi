"""Single-page "Add custom API model" form for the enhanced terminal.

Implements docs/specs/custom-api-single-page.md. Every field lives on one
scrollable page; the form owns raw strings, touched flags and per-field
errors, and never writes configuration. An injected ``save`` callback runs
the existing saver transaction plus the post-save choices.

The page is a plain prompt_toolkit Application rather than a general form
framework: TextArea, CheckboxList, the layout containers and the existing
``dialog.*`` theme classes do the work.
"""

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    ConditionalContainer, DynamicContainer, HSplit, Layout, VSplit, Window, WindowAlign,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor
from prompt_toolkit.layout.scrollable_pane import ScrollablePane
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.output import Output
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import CheckboxList, TextArea

from ..ai.models import ModelError
from .custom_api import (
    API_FORMATS, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS,
    ConcurrentConfigChange, CustomApiForm, FormValidation, SaveError,
    is_local_base_url, is_plain_http, validate_base_url,
)

MIN_COLUMNS = 60
MIN_ROWS = 18
WIDE_COLUMNS = 100
LABEL_COLUMNS = 80
CARD_MAX_WIDTH = 96
LABEL_WIDTH = 18

FIELD_LABELS: Dict[str, str] = {
    "provider_id": "Provider ID *",
    "base_url": "Base URL *",
    "api_key": "API Key *",
    "model_id": "Model ID *",
    "display_name": "Display name",
    "context_window": "Context window",
    "max_tokens": "Max output tokens",
}

# Visual order used when locating the first field with an error.
ERROR_ORDER = ("api", "provider_id", "base_url", "allow_http", "api_key",
               "model_id", "display_name", "context_window", "max_tokens")


@dataclass
class FormDraft:
    """Raw strings and UI state. ``api_key`` is never rendered by ``repr``."""

    api: str = "openai-completions"
    provider_id: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model_id: str = ""
    display_name: str = ""
    context_window: str = str(DEFAULT_CONTEXT_WINDOW)
    max_tokens: str = str(DEFAULT_MAX_TOKENS)
    image_input: bool = False
    reasoning: bool = False
    use_now: bool = True
    set_default: bool = False
    allow_http: bool = False
    touched: Dict[str, bool] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class SaveResult:
    """What the save service wants shown after the page closes."""

    messages: List[str] = field(default_factory=list)


@dataclass
class CustomApiFormResult:
    """A saved model plus the two post-save choices."""

    form: CustomApiForm
    use_now: bool
    set_default: bool
    messages: List[str] = field(default_factory=list)


ValidateForm = Callable[[Dict[str, object]], FormValidation]
SaveForm = Callable[[CustomApiForm, bool, bool], Awaitable[SaveResult]]


def _clip(text: str, width: int) -> str:
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


def _fit(fragments: List[Tuple[str, str]], width: int) -> StyleAndTextTuples:
    out: StyleAndTextTuples = []
    used = 0
    for style, text in fragments:
        room = width - used
        if room <= 0:
            break
        piece = _clip(text, room)
        out.append((style, piece))
        used += get_cwidth(piece)
    return out


def _shortcut_label(shortcut: str) -> str:
    parts = shortcut.split()
    labels = []
    for part in parts:
        if part.startswith("c-") and len(part) == 3:
            labels.append("Ctrl+" + part[2].upper())
        elif part == "escape":
            labels.append("Esc")
        elif part == "s-tab":
            labels.append("Shift+Tab")
        else:
            labels.append(part.capitalize())
    return "+".join(labels)


class _Selector:
    """Focusable API-format value; left/right cycles without a dialog."""

    def __init__(self, options: List[Tuple[str, str]], index: int, on_change) -> None:
        self.options = options
        self.index = index
        self.on_change = on_change
        self.control = FormattedTextControl(self._fragments, focusable=True,
                                            key_bindings=self._bindings())
        self.window = Window(self.control, height=1, style="class:dialog.field")

    def _fragments(self) -> StyleAndTextTuples:
        def click(event) -> None:
            if event.event_type == MouseEventType.MOUSE_UP:
                self._cycle(1)

        return [("", f" {self.options[self.index][1]} ▾", click)]

    def _cycle(self, step: int) -> None:
        self.index = (self.index + step) % len(self.options)
        self.on_change(self.options[self.index][0])
        get_app().invalidate()

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("left")
        def _left(event) -> None:
            self._cycle(-1)

        @bindings.add("right")
        def _right(event) -> None:
            self._cycle(1)

        @bindings.add("enter")
        def _enter(event) -> None:
            event.app.layout.focus_next()

        return bindings

    def __pt_container__(self):
        return self.window


class _TextButton:
    """Borderless button that uses the dialog palette and fills its row."""

    def __init__(self, text: str, handler: Callable[[], None], width: int) -> None:
        self.text = text
        self.handler = handler
        self.width = width
        self.control = FormattedTextControl(self._fragments, focusable=True,
                                            key_bindings=self._bindings())
        self.window = Window(self.control, height=1, width=width,
                             align=WindowAlign.CENTER, style="class:dialog.button")

    def _fragments(self) -> StyleAndTextTuples:
        def click(event) -> None:
            if event.event_type == MouseEventType.MOUSE_UP:
                self.handler()

        return [("", f" {self.text} ", click)]

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add(" ")
        @bindings.add("enter")
        def _press(event) -> None:
            self.handler()

        return bindings

    def __pt_container__(self):
        return self.window


class _Checkbox(CheckboxList):
    """Single ``[ ]``/``[x]`` checkbox with the dialog palette."""

    show_scrollbar = False

    def __init__(self, text: str, checked: bool) -> None:
        super().__init__(
            [("value", text)],
            default_values=["value"] if checked else [],
            open_character="[", select_character="x", close_character="]",
            container_style="class:dialog.checkbox.row",
            default_style="class:dialog.checkbox",
            selected_style="class:dialog.checkbox",
            checked_style="class:dialog.checkbox.checked",
        )
        self.show_scrollbar = False

    @property
    def checked(self) -> bool:
        return "value" in self.current_values

    @checked.setter
    def checked(self, value: bool) -> None:
        self.current_values = ["value"] if value else []


class CustomApiFormApp:
    """The single page. ``run`` returns a saved result or None on cancel."""

    def __init__(
        self,
        *,
        validate: ValidateForm,
        save: SaveForm,
        style: BaseStyle,
        shortcuts: Optional[List[str]] = None,
        initial: Optional[FormDraft] = None,
        mono: bool = False,
        input: Optional[Input] = None,
        output: Optional[Output] = None,
    ) -> None:
        self.validate = validate
        self.save = save
        self.mono = mono
        self.shortcuts = list(shortcuts or ["c-s"])
        self._save_shortcuts = [
            shortcut for shortcut in self.shortcuts
            if tuple(shortcut.split()) not in (("tab",), ("s-tab",), ("escape",), ("c-c",))
        ]
        self.draft = initial or FormDraft()
        self._validation = FormValidation(None, {})
        self.saving = False
        self._status = ""
        self._status_kind = "status"
        self._last_focus: Optional[str] = None

        self._build_widgets()
        self._checkbox_state = {key: box.checked for key, box in self.checkboxes.items()}
        self._build_layout()
        self.app = Application(
            layout=self._layout,
            key_bindings=self._create_bindings(),
            style=style,
            full_screen=True,
            mouse_support=True,
            input=input,
            output=output,
            before_render=self._before_render,
        )
        self._focus("provider_id")
        self._revalidate()

    # -- widgets ---------------------------------------------------------

    def _build_widgets(self) -> None:
        self.inputs: Dict[str, TextArea] = {}
        for key in ("provider_id", "base_url", "api_key", "model_id",
                    "display_name", "context_window", "max_tokens"):
            area = TextArea(
                text=getattr(self.draft, key),
                multiline=False,
                password=(key == "api_key"),
                wrap_lines=False,
                focus_on_click=True,
                style="class:dialog.field",
                input_processors=[BeforeInput(" ")],
            )
            area.accept_handler = self._accept
            area.buffer.on_text_changed += (
                lambda _buffer, field=key: self._on_text_changed(field))
            if key == "api_key":
                placeholder = ConditionalProcessor(
                    BeforeInput([("class:dialog.hint", "Required")]),
                    Condition(lambda area=area: not area.text and not self._is_focused(area)),
                )
                area.control.input_processors.append(placeholder)
            elif key == "display_name":
                placeholder = ConditionalProcessor(
                    BeforeInput(lambda: [("class:dialog.hint", self.inputs["model_id"].text or "Optional")]),
                    Condition(lambda area=area: not area.text and not self._is_focused(area)),
                )
                area.control.input_processors.append(placeholder)
            self.inputs[key] = area

        self.api_selector = _Selector(list(API_FORMATS), self._api_index(), self._set_api)
        self.checkboxes: Dict[str, _Checkbox] = {
            "image_input": _Checkbox("Image input", self.draft.image_input),
            "reasoning": _Checkbox("Reasoning", self.draft.reasoning),
            "use_now": _Checkbox("Use this model now", self.draft.use_now),
            "set_default": _Checkbox("Set as default", self.draft.set_default),
            "allow_http": _Checkbox("Allow sending the API key over HTTP", self.draft.allow_http),
        }
        self.cancel_button = _TextButton("Cancel", self._cancel, 12)
        self.save_button = _TextButton("Save model", self._save, 14)
        self.use_root_button = _TextButton("Use suggested root", self._apply_root, 22)
        self.use_format_button = _TextButton("Use that format", self._apply_format, 16)
        self._update_action_buttons()

    def _api_index(self) -> int:
        return next((index for index, (value, _label) in enumerate(API_FORMATS)
                     if value == self.draft.api), 0)

    # -- layout ----------------------------------------------------------

    def _build_layout(self) -> None:
        too_small = Condition(self._too_small)

        body = HSplit([
            Window(FormattedTextControl([(
                "class:dialog.hint",
                "Configure your endpoint and model. Connection is not verified on save.",
            )]), wrap_lines=True, dont_extend_height=True),
            Window(height=1),
            self._group("CONNECTION"),
            self._selector_field(),
            self._field("provider_id"),
            self._field("base_url"),
            self._preview_line(),
            self._suggestion_area(),
            self._http_area(),
            self._field("api_key"),
            Window(height=1),
            self._group("MODEL"),
            self._field("model_id"),
            self._field("display_name"),
            Window(height=1),
            self._group("CAPABILITIES & LIMITS"),
            self._pair(self._field("context_window"), self._field("max_tokens"),
                       self._wide_side_by_side),
            self._pair(self.checkboxes["image_input"], self.checkboxes["reasoning"],
                       self._checkboxes_side_by_side),
            Window(FormattedTextControl([(
                "class:dialog.hint",
                "Limits describe client settings; confirm them with your service docs.",
            )]), wrap_lines=True, dont_extend_height=True),
            Window(height=1),
            self._group("AFTER SAVING"),
            self._pair(self.checkboxes["use_now"], self.checkboxes["set_default"],
                       self._wide_side_by_side),
            Window(height=1),
            Window(FormattedTextControl([(
                "class:dialog.hint",
                "Required fields are marked *. Costs are not configured.",
            )]), wrap_lines=True, dont_extend_height=True),
        ], style="class:dialog")

        self._scroll = ScrollablePane(
            body, keep_cursor_visible=True, keep_focused_window_visible=True,
            show_scrollbar=True, display_arrows=False,
        )
        self._saving_message = Window(
            FormattedTextControl([("class:dialog.status", " Saving…")]),
            height=1, dont_extend_height=True,
        )
        self._body_area = DynamicContainer(
            lambda: self._saving_message if self.saving else self._scroll)

        buttons = VSplit([
            self.cancel_button,
            Window(width=2),
            self.save_button,
        ], height=1)
        self._buttons_area = ConditionalContainer(buttons, filter=~Condition(lambda: self.saving))

        footer = HSplit([
            Window(FormattedTextControl(self._separator_fragments), height=1),
            ConditionalContainer(
                Window(FormattedTextControl(self._status_fragments),
                       wrap_lines=True, dont_extend_height=True),
                filter=Condition(lambda: bool(self._status)),
            ),
            VSplit([
                Window(FormattedTextControl(self._hint_fragments), height=1),
                self._buttons_area,
            ], height=1),
        ], style="class:dialog")

        card = HSplit([
            Window(FormattedTextControl(self._top_line), height=1, style="class:dialog"),
            VSplit([
                Window(width=1, char="│", style="class:dialog.border"),
                HSplit([
                    Window(height=1),
                    VSplit([
                        Window(width=2),
                        HSplit([
                            ConditionalContainer(
                                Window(FormattedTextControl(self._summary_fragments),
                                       wrap_lines=True, dont_extend_height=True),
                                filter=Condition(lambda: bool(self._summary_fragments())),
                            ),
                            self._body_area,
                        ], style="class:dialog"),
                        Window(width=2),
                    ], style="class:dialog"),
                    footer,
                    Window(height=1),
                ], style="class:dialog"),
                Window(width=1, char="│", style="class:dialog.border"),
            ], style="class:dialog"),
            Window(FormattedTextControl(self._bottom_line), height=1, style="class:dialog"),
        ], width=self._card_width, style="class:dialog")

        resize = HSplit([
            Window(),
            Window(FormattedTextControl([(
                "class:dialog.warning", "Resize terminal to at least 60 × 18",
            )]), height=1, align=WindowAlign.CENTER),
            Window(),
        ], style="class:dialog")

        margin = lambda: max(0, (self._columns() - self._card_width()) // 2)
        centered = VSplit([
            Window(width=margin),
            card,
            Window(width=lambda: max(0, self._columns() - self._card_width() - margin())),
        ])
        self._layout = Layout(HSplit([
            ConditionalContainer(centered, filter=~too_small),
            ConditionalContainer(resize, filter=too_small),
        ]))

    def _group(self, text: str) -> Window:
        return Window(FormattedTextControl([("class:dialog.group", text)]),
                      height=1, dont_extend_height=True)

    def _label(self, text: str, target) -> Window:
        return Window(
            FormattedTextControl(lambda: self._label_fragments(text, target)),
            width=lambda: Dimension.exact(LABEL_WIDTH) if self._wide() else Dimension(),
            height=1, dont_extend_height=True,
        )

    def _field(self, key: str) -> HSplit:
        label = self._label(FIELD_LABELS[key], self.inputs[key].control)
        inline = VSplit([label, self.inputs[key]], height=1)
        stacked = HSplit([label, self.inputs[key]])
        return HSplit([
            DynamicContainer(lambda: inline if self._wide() else stacked),
            self._error_line(key),
        ])

    def _selector_field(self) -> HSplit:
        label = self._label("API format", self.api_selector.control)
        inline = VSplit([label, self.api_selector], height=1)
        stacked = HSplit([label, self.api_selector])
        return HSplit([DynamicContainer(lambda: inline if self._wide() else stacked)])

    def _error_line(self, key: str) -> ConditionalContainer:
        return ConditionalContainer(
            Window(FormattedTextControl(lambda: self._error_fragments(key)),
                   wrap_lines=True, dont_extend_height=True),
            filter=Condition(lambda key=key: bool(self.draft.errors.get(key))),
        )

    def _preview_line(self) -> Window:
        return Window(FormattedTextControl(self._preview_fragments),
                      wrap_lines=True, dont_extend_height=True)

    def _suggestion_area(self) -> ConditionalContainer:
        row = VSplit([
            ConditionalContainer(self.use_root_button,
                                 filter=Condition(lambda: self._has_root_action())),
            Window(width=2),
            ConditionalContainer(self.use_format_button,
                                 filter=Condition(lambda: self._has_format_action())),
        ], height=1)
        return ConditionalContainer(
            row, filter=Condition(lambda: self._has_root_action() or self._has_format_action()))

    def _http_area(self) -> ConditionalContainer:
        return ConditionalContainer(
            HSplit([
                Window(FormattedTextControl([(
                    "class:dialog.warning",
                    "Warning: the API key will be sent over an unencrypted connection.",
                )]), wrap_lines=True, dont_extend_height=True),
                self.checkboxes["allow_http"],
                self._error_line("allow_http"),
            ], style="class:dialog"),
            filter=Condition(self._http_required),
        )

    def _pair(self, left, right, side_by_side: Callable[[], bool]) -> DynamicContainer:
        inline = VSplit([left, Window(width=2), right])
        stacked = HSplit([left, right])
        return DynamicContainer(lambda: inline if side_by_side() else stacked)

    # -- sizes -----------------------------------------------------------

    def _size(self) -> Tuple[int, int]:
        size = get_app().output.get_size()
        return max(1, size.columns), max(1, size.rows)

    def _columns(self) -> int:
        return self._size()[0]

    def _card_width(self) -> int:
        return min(CARD_MAX_WIDTH, max(1, self._columns() - 4))

    def _content_width(self) -> int:
        return max(1, self._card_width() - 6)

    def _wide(self) -> bool:
        return self._columns() >= LABEL_COLUMNS

    def _wide_side_by_side(self) -> bool:
        return self._columns() >= WIDE_COLUMNS

    def _checkboxes_side_by_side(self) -> bool:
        return self._columns() >= LABEL_COLUMNS

    def _too_small(self) -> bool:
        columns, rows = self._size()
        return columns < MIN_COLUMNS or rows < MIN_ROWS

    # -- text ------------------------------------------------------------

    def _top_line(self) -> StyleAndTextTuples:
        width = self._card_width()
        title = "Add custom API model"
        fill = max(0, width - 5 - get_cwidth(title))
        return [
            ("class:dialog.border", "╭─ "),
            ("class:dialog.title", _clip(title, max(0, width - 5))),
            ("class:dialog.border", " " + "─" * fill + "╮"),
        ]

    def _bottom_line(self) -> StyleAndTextTuples:
        width = self._card_width()
        return [("class:dialog.border", "╰" + "─" * max(0, width - 2) + "╯")]

    def _separator_fragments(self) -> StyleAndTextTuples:
        return [("class:dialog.separator", "─" * max(1, self._card_width() - 2))]

    def _label_fragments(self, text: str, target) -> StyleAndTextTuples:
        focused = get_app().layout.has_focus(target)
        style = "class:dialog.label.focused" if focused else "class:dialog.label"
        padded = text.ljust(LABEL_WIDTH) if self._wide() else text
        return [(style, padded)]

    def _error_fragments(self, key: str) -> StyleAndTextTuples:
        message = self.draft.errors.get(key, "")
        prefix = "Error: " if self.mono else ""
        return [("class:dialog.error", f"  {prefix}{message}")]

    def _preview_fragments(self) -> StyleAndTextTuples:
        if self._validation.request_url:
            return [
                ("class:dialog.hint", "Request URL: "),
                ("class:dialog.text", self._validation.request_url),
            ]
        return [("class:dialog.hint", "Request URL: Enter a valid base URL")]

    def _summary_fragments(self) -> StyleAndTextTuples:
        count = len(self.draft.errors)
        if not count:
            return []
        word = "field needs" if count == 1 else "fields need"
        prefix = "Error: " if self.mono else ""
        return [("class:dialog.error", f"{prefix}{count} {word} attention")]

    def _status_fragments(self) -> StyleAndTextTuples:
        if not self._status:
            return []
        prefix = ""
        if self.mono:
            prefix = {"error": "Error: ", "warning": "Warning: "}.get(self._status_kind, "")
        style = {
            "error": "class:dialog.error",
            "warning": "class:dialog.warning",
        }.get(self._status_kind, "class:dialog.status")
        return [(style, prefix + self._status)]

    def _hint_fragments(self) -> StyleAndTextTuples:
        parts: StyleAndTextTuples = [
            ("class:dialog.key", "Tab"), ("class:dialog.hint", " next · "),
            ("class:dialog.key", "Shift+Tab"), ("class:dialog.hint", " back · "),
            ("class:dialog.key", "Esc"), ("class:dialog.hint", " cancel"),
        ]
        if self._save_shortcuts:
            parts.extend([
                ("class:dialog.hint", " · "),
                ("class:dialog.key", _shortcut_label(self._save_shortcuts[0])),
                ("class:dialog.hint", " save"),
            ])
        buttons_width = self.cancel_button.width + 2 + self.save_button.width
        return _fit(parts, max(1, self._content_width() - buttons_width - 1))

    # -- state -----------------------------------------------------------

    def _values(self) -> Dict[str, object]:
        return {
            "api": self.api_selector.options[self.api_selector.index][0],
            "provider_id": self.inputs["provider_id"].text,
            "base_url": self.inputs["base_url"].text,
            "api_key": self.inputs["api_key"].text,
            "model_id": self.inputs["model_id"].text,
            "display_name": self.inputs["display_name"].text,
            "context_window": self.inputs["context_window"].text,
            "max_tokens": self.inputs["max_tokens"].text,
            "image_input": self.checkboxes["image_input"].checked,
            "reasoning": self.checkboxes["reasoning"].checked,
        }

    def _revalidate(self) -> None:
        self._validation = self.validate(self._values())
        self.draft.errors = {
            key: message for key, message in self._validation.errors.items()
            if self.draft.touched.get(key)
        }
        self._update_action_buttons()

    def _set_api(self, value: str) -> None:
        self.draft.api = value
        self.draft.touched["api"] = True
        self.checkboxes["allow_http"].checked = False
        self._revalidate()

    def _on_text_changed(self, key: str) -> None:
        if key == "base_url":
            self.checkboxes["allow_http"].checked = False
        self._revalidate()

    def _http_required(self) -> bool:
        try:
            url = validate_base_url(self.inputs["base_url"].text)
        except ValueError:
            return False
        return is_plain_http(url) and not is_local_base_url(url)

    def _has_root_action(self) -> bool:
        suggestion = self._validation.suggestion
        return bool(suggestion and suggestion.suggested_root)

    def _has_format_action(self) -> bool:
        suggestion = self._validation.suggestion
        return bool(suggestion and suggestion.matching_api)

    def _update_action_buttons(self) -> None:
        suggestion = self._validation.suggestion
        if suggestion and suggestion.matching_api:
            label = f"Use {dict(API_FORMATS)[suggestion.matching_api]} format"
            self.use_format_button.text = label
            self.use_format_button.window.width = min(34, max(16, get_cwidth(label) + 2))

    def _apply_root(self) -> None:
        suggestion = self._validation.suggestion
        if suggestion is None or suggestion.suggested_root is None:
            return
        self.inputs["base_url"].text = suggestion.suggested_root
        self.draft.touched["base_url"] = True
        self.checkboxes["allow_http"].checked = False
        self._revalidate()
        self._focus("base_url")

    def _apply_format(self) -> None:
        suggestion = self._validation.suggestion
        if suggestion is None or suggestion.matching_api is None:
            return
        self.api_selector.index = next(
            index for index, (value, _label) in enumerate(API_FORMATS)
            if value == suggestion.matching_api)
        self._set_api(suggestion.matching_api)
        self._focus("base_url")

    # -- focus -----------------------------------------------------------

    def _is_focused(self, area: TextArea) -> bool:
        return get_app().layout.has_focus(area.control)

    def _focus_key(self, window) -> Optional[str]:
        if window is None:
            return None
        for key, area in self.inputs.items():
            if area.window is window:
                return key
        if self.api_selector.window is window:
            return "api"
        for key, checkbox in self.checkboxes.items():
            if checkbox.window is window:
                return f"checkbox:{key}"
        return None

    def _focus(self, key: str) -> None:
        if key == "api":
            self.app.layout.focus(self.api_selector.control)
        elif key == "allow_http":
            self.app.layout.focus(self.checkboxes["allow_http"].control)
        else:
            self.app.layout.focus(self.inputs[key].control)

    def _blur(self) -> None:
        key = self._focus_key(get_app().layout.current_window)
        if key in FIELD_LABELS and not self.draft.touched.get(key):
            self.draft.touched[key] = True
            self._revalidate()

    def _focus_next(self) -> None:
        self._blur()
        get_app().layout.focus_next()

    def _focus_previous(self) -> None:
        self._blur()
        get_app().layout.focus_previous()

    def _focus_first_error(self) -> None:
        for key in ERROR_ORDER:
            if key in self.draft.errors:
                self._focus(key)
                return

    def _sync_focus(self) -> None:
        key = self._focus_key(get_app().layout.current_window)
        if key == self._last_focus:
            return
        previous = self._last_focus
        self._last_focus = key
        if previous in FIELD_LABELS and not self.draft.touched.get(previous):
            self.draft.touched[previous] = True
            self._revalidate()

    def _sync_styles(self) -> None:
        focused = get_app().layout.current_window
        for area in self.inputs.values():
            area.window.style = (
                "class:text-area class:dialog.field.focused"
                if area.window is focused
                else "class:text-area class:dialog.field")
        self.api_selector.window.style = (
            "class:dialog.field.focused" if self.api_selector.window is focused
            else "class:dialog.field")
        for checkbox in self.checkboxes.values():
            checkbox.window.style = (
                "class:dialog.checkbox.focused" if checkbox.window is focused
                else "class:dialog.checkbox.row")
        for button in (self.cancel_button, self.save_button,
                       self.use_root_button, self.use_format_button):
            button.window.style = (
                "class:dialog.button.focused" if button.window is focused
                else "class:dialog.button")

    def _before_render(self, app) -> None:
        self._sync_focus()
        self._sync_checkboxes()
        self._sync_styles()

    def _sync_checkboxes(self) -> None:
        for key, checkbox in self.checkboxes.items():
            checked = checkbox.checked
            if checked != self._checkbox_state[key]:
                self._checkbox_state[key] = checked
                if key == "allow_http":
                    self._revalidate()

    # -- actions ---------------------------------------------------------

    def _accept(self, buffer) -> bool:
        self._focus_next()
        return True

    def _set_status(self, message: str, kind: str) -> None:
        self._status = message
        self._status_kind = kind

    def _cancel(self) -> None:
        if self.saving:
            return
        self.draft.api_key = ""
        self.inputs["api_key"].text = ""
        get_app().exit(result=None)

    def _save(self) -> None:
        if self.saving or self._too_small():
            return
        for key in FIELD_LABELS:
            self.draft.touched[key] = True
        self.draft.touched["api"] = True
        self._revalidate()
        errors = dict(self.draft.errors)
        if self._http_required() and not self.checkboxes["allow_http"].checked:
            errors["allow_http"] = "Confirm that the API key may be sent over this HTTP address"
        self.draft.errors = errors
        if errors:
            word = "field needs" if len(errors) == 1 else "fields need"
            self._set_status(f"{len(errors)} {word} attention", "error")
            self._focus_first_error()
            get_app().invalidate()
            return
        form = self._validation.form
        if form is None:  # defensive: validation agreed there were no errors
            return
        self.saving = True
        self._set_status("", "status")
        get_app().invalidate()
        get_app().create_background_task(self._save_async(form))

    async def _save_async(self, form: CustomApiForm) -> None:
        use_now = self.checkboxes["use_now"].checked
        set_default = self.checkboxes["set_default"].checked
        try:
            result = await self.save(form, use_now, set_default)
        except ConcurrentConfigChange:
            self.saving = False
            self._set_status(
                "Configuration changed on disk; review the values and save again", "warning")
            self.draft.touched["provider_id"] = True
            self._revalidate()
            self._focus_first_error()
            get_app().invalidate()
            return
        except (SaveError, OSError, ValueError, ModelError) as exc:
            self.saving = False
            self._set_status(str(exc) or exc.__class__.__name__, "error")
            self.app.layout.focus(self.save_button.control)
            get_app().invalidate()
            return
        get_app().exit(result=CustomApiFormResult(
            form=form, use_now=use_now, set_default=set_default, messages=result.messages))

    def _create_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("tab", eager=True)
        def _tab(event) -> None:
            self._focus_next()

        @bindings.add("s-tab", eager=True)
        def _shift_tab(event) -> None:
            self._focus_previous()

        @bindings.add("escape")
        @bindings.add("c-c")
        def _cancel(event) -> None:
            self._cancel()

        seen = set()
        for shortcut in self._save_shortcuts:
            keys = tuple(shortcut.split())
            if keys in seen:
                continue
            seen.add(keys)

            @bindings.add(*keys)
            def _submit(event) -> None:
                self._save()

        return bindings

    async def run(self) -> Optional[CustomApiFormResult]:
        return await self.app.run_async()
