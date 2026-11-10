"""Shared truncation utilities.

Python port of ``packages/coding-agent/src/core/tools/truncate.ts``: head
truncation for file reads, tail truncation for shell output, both bounded by
independent line and byte limits (whichever is hit first wins).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: Optional[str]  # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False
    max_lines: int = DEFAULT_MAX_LINES
    max_bytes: int = DEFAULT_MAX_BYTES


def split_lines_for_counting(content: str) -> List[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def byte_length(text: str) -> int:
    return len(text.encode("utf-8"))


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Keep the first N lines/bytes; never returns partial lines."""
    total_bytes = byte_length(content)
    lines = split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    first_line_bytes = byte_length(lines[0])
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: List[str] = []
    output_bytes_count = 0
    truncated_by = "lines"

    for i, line in enumerate(lines[:max_lines]):
        line_bytes = byte_length(line) + (1 if i > 0 else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(line)
        output_bytes_count += line_bytes

    if truncated_by != "bytes":
        truncated_by = "lines" if len(output_lines) < total_lines else "bytes"

    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=byte_length(output_content),
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Keep the last N lines/bytes; may return a partial first line."""
    total_bytes = byte_length(content)
    lines = split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    output_lines: List[str] = []
    output_bytes_count = 0
    truncated_by = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(output_lines) >= max_lines:
            break
        line = lines[i]
        line_bytes = byte_length(line) + (1 if output_lines else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines:
                truncated_line = truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines.insert(0, truncated_line)
                output_bytes_count = byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines.insert(0, line)
        output_bytes_count += line_bytes

    if truncated_by != "bytes":
        truncated_by = "lines" if len(output_lines) < total_lines else "bytes"

    output_content = "\n".join(output_lines)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines),
        output_bytes=byte_length(output_content),
        last_line_partial=last_line_partial,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """Truncate to the last ``max_bytes`` bytes, respecting UTF-8 boundaries."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    start = len(data) - max_bytes
    while start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    return data[start:].decode("utf-8", errors="replace")


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH):
    if len(line) <= max_chars:
        return {"text": line, "was_truncated": False}
    return {"text": f"{line[:max_chars]}... [truncated]", "was_truncated": True}
