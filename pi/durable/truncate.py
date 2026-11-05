"""Shared truncation utilities for tool outputs.

Python port of ``packages/durable/src/truncate.ts``. Truncation is based on two
independent limits - whichever is hit first wins: 2000 lines or 50KB. Never
returns partial lines. Tool output streams are bounded by ``output.py`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Optional[str]  # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def utf8_byte_length(content: str) -> int:
    return len(content.encode("utf-8"))


def split_lines_for_counting(content: str) -> List[str]:
    """Lines counted like ``truncate_head``: a trailing newline adds no line."""
    if len(content) == 0:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(
    content: str, max_lines: Optional[int] = None, max_bytes: Optional[int] = None
) -> TruncationResult:
    """Truncate content from the head (keep first N lines/bytes)."""
    return truncate_head_of(
        content,
        {"lines": len(split_lines_for_counting(content)), "bytes": utf8_byte_length(content)},
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_head_of(
    prefix: str,
    totals: dict,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> TruncationResult:
    """``truncate_head`` of a text known by a prefix and its totals.

    The prefix must be the whole text, or longer than ``max_bytes + 1`` UTF-8
    bytes, or hold at least ``max_lines`` newlines; then the result equals
    ``truncate_head`` of the whole text.
    """
    max_lines = DEFAULT_MAX_LINES if max_lines is None else max_lines
    max_bytes = DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
    total_bytes = totals["bytes"]
    lines = split_lines_for_counting(prefix)
    total_lines = totals["lines"]

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=prefix,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = utf8_byte_length(lines[0])
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines_list: List[str] = []
    output_bytes_count = 0
    truncated_by = "lines"
    for i, line in enumerate(lines[:max_lines]):
        line_bytes = utf8_byte_length(line) + (1 if i > 0 else 0)  # +1 for the newline
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_list.append(line)
        output_bytes_count += line_bytes

    # Without a byte break, only omitted lines prove the line limit was reached.
    if truncated_by != "bytes":
        truncated_by = "lines" if len(output_lines_list) < total_lines else "bytes"

    output_content = "\n".join(output_lines_list)
    final_output_bytes = utf8_byte_length(output_content)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_list),
        output_bytes=final_output_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )
