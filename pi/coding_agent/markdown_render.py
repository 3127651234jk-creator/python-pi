"""Minimal ANSI Markdown renderer for terminal output.

Covers the constructs that appear in assistant replies: headings, bold,
italic, inline code, fenced code blocks, ordered/unordered lists, block
quotes and horizontal rules. Unknown lines pass through unchanged.
"""

from __future__ import annotations

import re
from typing import List

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


def render_markdown(text: str, color: bool = True) -> str:
    """Render Markdown source to an ANSI string (plain when color is off)."""
    out: List[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            if color:
                out.append(DIM + line + RESET)
            else:
                out.append(line)
            continue
        if in_code:
            out.append((DIM + line + RESET) if color else line)
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
