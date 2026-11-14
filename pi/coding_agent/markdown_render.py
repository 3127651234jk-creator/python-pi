"""Minimal ANSI Markdown renderer for terminal output.

Covers the constructs that appear in assistant replies: headings, bold,
italic, inline code, fenced code blocks, ordered/unordered lists, block
quotes and horizontal rules. Unknown lines pass through unchanged.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .syntax_highlight import highlight_code

BOLD = "\x1b[1m"
ITALIC = "\x1b[3m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_RULE = re.compile(r"^\s*([-*_]\s*){3,}$")


def _inline(text: str, color: bool) -> str:
    if not color:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", BOLD + r"\1" + RESET, text, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", ITALIC + r"\1" + RESET, text)
    text = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", YELLOW + r"\1" + RESET, text)
    return text


def _code_line(line: str, color: bool) -> str:
    return (DIM + line + RESET) if color else line


def render_markdown(text: str, color: bool = True,
                    theme: Optional[Dict[str, str]] = None) -> str:
    """Render Markdown source to an ANSI string (plain when color is off).

    Fenced blocks are collected whole and lexed at once, so multi-line strings
    and comments keep their style on every line. Unknown languages and
    unterminated fences fall back to dim plain code.
    """

    out: List[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.strip().startswith("```"):
            info = line.strip()[3:].strip()
            language = info.split()[0] if info else ""
            out.append(_code_line(line, color))
            block: List[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            highlighted = highlight_code("\n".join(block), language, color, theme) if block else []
            if highlighted is None:
                out.extend(_code_line(code_line, color) for code_line in block)
            else:
                out.extend(highlighted)
            if index < len(lines):
                out.append(_code_line(lines[index], color))
                index += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            marker = (BOLD + CYAN) if color else ""
            out.append(("#" * min(level, 3) + " " if not color else "") + marker + heading.group(2) + (RESET if color else ""))
            continue
        bullet = _BULLET.match(line)
        if bullet:
            marker = "• " if color else "- "
            out.append(bullet.group(1) + marker + _inline(bullet.group(2), color))
            continue
        numbered = _NUMBERED.match(line)
        if numbered:
            out.append(numbered.group(1) + numbered.group(2) + " " + _inline(numbered.group(3), color))
            continue
        quote = _QUOTE.match(line)
        if quote:
            bar = "│ " if color else "> "
            body = (DIM + quote.group(1) + RESET) if color else quote.group(1)
            out.append(bar + body)
            continue
        if _RULE.fullmatch(line):
            out.append("─" * 40 if color else "-" * 40)
            continue
        out.append(_inline(line, color))
    return "\n".join(out)
