"""Token-level ANSI highlighting for fenced code blocks.

The whole block is lexed at once (Pygments), then each token is split on
newlines and styled per line. That keeps multi-line strings and comments
colored while allowing the Markdown renderer to clip or reflow lines without
leaking styles.

Unknown languages and lexer failures return ``None`` so the caller can fall
back to plain code; ``color=False`` never emits ANSI.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

RESET = "\x1b[0m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
MAGENTA = "\x1b[35m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
BOLD = "\x1b[1m"

# Keys are lower-case Pygments token paths. ``string.interpol`` maps to the
# body color (empty style) so f-string/template expressions do not inherit the
# surrounding string color. The lookup walks token parents, so more specific
# entries win.
DEFAULT_THEME: Dict[str, str] = {
    "default": "",
    "comment": DIM,
    "keyword": MAGENTA,
    "keyword.constant": MAGENTA,
    "keyword.type": CYAN,
    "string": GREEN,
    "string.interpol": "",
    "string.escape": YELLOW,
    "string.affix": MAGENTA,
    "number": YELLOW,
    "name.function": CYAN,
    "name.class": CYAN,
    "name.decorator": YELLOW,
    "name.builtin": CYAN,
    "operator": "",
    "punctuation": "",
    "generic.heading": BOLD,
    "error": RED,
}

# The mono theme styles nothing; TUI users choosing the mono palette get plain code.
MONO_THEME: Dict[str, str] = {"default": ""}


def token_name(token) -> str:
    text = str(token)
    return text[6:].lower() if text.startswith("Token.") else ""


@functools.lru_cache(maxsize=256)
def _lex_lines(code: str, language: str) -> Optional[Tuple[Tuple[Tuple[str, str], ...], ...]]:
    """Lex ``code`` into lines of (token name, text) segments.

    Cached per (code, language): streaming re-renders reuse the tokenization.
    """

    try:
        lexer = get_lexer_by_name(language, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return None
    lines: List[List[Tuple[str, str]]] = [[]]
    try:
        for token, value in lex(code, lexer):
            if not value:
                continue
            name = token_name(token)
            # Pygments names strings and numbers ``literal.string`` /
            # ``literal.number``; the short highlight.js-style names are stable.
            if name.startswith("literal."):
                name = name[len("literal."):]
            parts = value.split("\n")
            for index, part in enumerate(parts):
                if index:
                    lines.append([])
                if part:
                    lines[-1].append((name, part))
    except Exception:  # noqa: BLE001 - a lexer bug must not break rendering
        return None
    return tuple(tuple(line) for line in lines)


def highlight_code(code: str, language: str = "", color: bool = True,
                   theme: Optional[Dict[str, str]] = None) -> Optional[List[str]]:
    """Render ``code`` as ANSI lines, or None when it should stay plain.

    The returned list always has one entry per source line and contains no
    character changes beyond style sequences: empty lines stay empty, and a
    token spanning newlines is styled independently on every line.
    """

    if not color:
        return code.split("\n")
    if not language:
        return None
    lexed = _lex_lines(code, language)
    if lexed is None:
        return None
    palette = DEFAULT_THEME if theme is None else theme
    lines: List[str] = []
    for segments in lexed:
        rendered = []
        for name, text in segments:
            style = _style_for_name(name, palette)
            rendered.append(style + text + RESET if style else text)
        lines.append("".join(rendered))
    return lines


def _style_for_name(name: str, theme: Dict[str, str]) -> str:
    """Style for a flattened token path such as ``string.interpol``."""

    parts = name.split(".") if name else []
    while parts:
        key = ".".join(parts)
        if key in theme:
            return theme[key]
        parts.pop()
    return theme.get("default", "")
