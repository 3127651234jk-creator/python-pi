"""Token-level fenced-code highlighting (S01-S07)."""

from __future__ import annotations

import re

from pi.coding_agent.markdown_render import render_markdown
from pi.coding_agent.syntax_highlight import DEFAULT_THEME, MONO_THEME, highlight_code
from pi.coding_agent.tui import TerminalUI

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GREEN = "\x1b[32m"
DIM = "\x1b[2m"
MAGENTA = "\x1b[35m"


def plain(text: str) -> str:
    return ANSI.sub("", text)


def code_lines(markdown: str) -> list:
    """Rendered lines between the fence lines."""
    rendered = render_markdown(markdown).splitlines()
    return rendered[1:-1]


def test_s01_multiline_python_string_keeps_style_per_line():
    code = 'text = """first\n\nthird\nfourth"""\nprint(text)'
    lines = code_lines("```python\n" + code + "\n```")
    assert plain("\n".join(lines)) == code
    assert GREEN in lines[0]
    assert lines[1] == ""  # the empty source line stays empty
    assert GREEN in lines[2]
    assert GREEN in lines[3]  # the last string line is still string-colored
    assert GREEN not in lines[4]  # the statement after the string is not


def test_s02_multiline_comments_do_not_leak_into_following_code():
    code = "/* first\nsecond */\nconst value = 1;"
    lines = code_lines("```javascript\n" + code + "\n```")
    assert DIM in lines[0] and DIM in lines[1]
    assert "const" in plain(lines[2])
    assert DIM not in lines[2]
    assert MAGENTA in lines[2]

    css = "/* one\ntwo */\nbody { color: red; }"
    css_lines = code_lines("```css\n" + css + "\n```")
    assert DIM in css_lines[0] and DIM in css_lines[1]
    assert DIM not in css_lines[2]


def test_s03_interpolation_uses_body_color():
    python = 'value = f"hi {name} there"'
    py_line = code_lines("```python\n" + python + "\n```")[0]
    assert "name" in plain(py_line)
    assert GREEN + "name" not in py_line
    assert GREEN + "hi " in py_line

    js = "const s = `hi ${name} there`;"
    js_line = code_lines("```javascript\n" + js + "\n```")[0]
    assert "name" in plain(js_line)
    assert GREEN + "name" not in js_line
    assert GREEN + "hi " in js_line


def test_s04_unknown_language_unclosed_fence_and_color_off():
    unknown = render_markdown("```unknown-language\nx = 1\n```")
    assert DIM in unknown
    assert "x = 1" in plain(unknown)

    unclosed = render_markdown("```python\nx = 1\n")
    assert "x = 1" in plain(unclosed)
    assert not unclosed.endswith("```")

    plain_mode = render_markdown("```python\ndef f():\n    return 'x'\n```", color=False)
    assert "\x1b[" not in plain_mode
    assert plain_mode == "```python\ndef f():\n    return 'x'\n```"


def test_s05_unicode_tabs_and_backticks_survive():
    code = 's = "中文😀\tvalue `tick`"\n\tindented = 1'
    markdown = "```python\n" + code + "\n```"
    rendered = render_markdown(markdown)
    assert plain(rendered) == markdown
    assert "中文😀" in rendered
    assert "\t" in rendered
    assert "`tick`" in rendered


def test_s06_themes_and_both_display_paths():
    code = "```python\ndef f():\n    return 1\n```"
    default = render_markdown(code)
    mono = render_markdown(code, theme=MONO_THEME)
    custom = render_markdown(code, theme={**DEFAULT_THEME, "keyword": "\x1b[31m"})
    assert default != mono
    assert default != custom
    assert plain(default) == plain(custom) == plain(mono) == "```python\ndef f():\n    return 1\n```"
    assert "\x1b[0m" in default and "\x1b[0m" in custom
    # Mono leaves code tokens unstyled; only the fence lines keep their dim marker.
    assert mono.splitlines()[1] == "def f():"
    assert mono.splitlines()[2] == "    return 1"

    # The full-screen transcript parses the same ANSI output the CLI prints.
    dummy = object.__new__(TerminalUI)
    dummy.transcript = default
    dummy._transcript_version = 1
    dummy._formatted_version = 0
    dummy._formatted_transcript = []
    formatted = TerminalUI.formatted_transcript(dummy)
    text = "".join(piece for _style, piece, *_ in formatted)
    assert "def f():" in text and "return 1" in text


def test_s07_chunked_prefixes_match_one_shot_and_do_not_leak():
    markdown = ("Intro\n\n```python\nvalue = '''a\n\nb'''\nprint(value)\n```\n\nAfter bold.")
    one_shot = render_markdown(markdown)
    for end in range(1, len(markdown) + 1):
        prefix = render_markdown(markdown[:end])
        # The renderer joins lines, so it drops a single trailing newline but
        # keeps an empty line before the end.
        assert plain(prefix) == "\n".join(markdown[:end].splitlines())
        # Every style sequence opened in a prefix is closed in that prefix.
        opens = [code for code in ANSI.findall(prefix) if code != "\x1b[0m"]
        assert len(opens) == prefix.count("\x1b[0m")
    assert plain(one_shot) == markdown


def test_highlight_code_returns_none_for_unknown_or_plain_mode():
    assert highlight_code("x = 1", "no-such-lexer") is None
    assert highlight_code("x = 1", "", True) is None
    assert highlight_code("x = 1", "python", False) == ["x = 1"]
