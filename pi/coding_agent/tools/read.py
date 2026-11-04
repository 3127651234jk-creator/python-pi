"""read tool: read file contents as text or image.

Python port of ``packages/coding-agent/src/core/tools/read.ts``.
"""

from __future__ import annotations

import base64
import asyncio
import os
from typing import Any, List, Optional, Union

from ...ai.types import ImageContent, TextContent
from ...agent.types import AgentTool, AgentToolResult
from .common import detect_image_mime_type, resolve_to_cwd
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_head

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}

READ_DESCRIPTION = (
    f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
    f"Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines "
    f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. "
    "When you need the full file, continue with offset until complete."
)

READ_PROMPT_SNIPPET = "Read file contents"
READ_PROMPT_GUIDELINES = ["Use read to examine files instead of cat or sed."]


def create_read_tool(cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: Any,
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        path: str = params["path"]
        offset: Optional[int] = params.get("offset")
        limit: Optional[int] = params.get("limit")
        absolute_path = resolve_to_cwd(path, cwd)

        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(f"File not found: {path}")

        mime_type = detect_image_mime_type(absolute_path)
        content: List[Union[TextContent, ImageContent]]

        if mime_type:
            with open(absolute_path, "rb") as handle:
                data = base64.b64encode(handle.read()).decode("ascii")
            content = [
                TextContent(text=f"Read image file [{mime_type}]"),
                ImageContent(data=data, mime_type=mime_type),
            ]
            return AgentToolResult(content=content)

        with open(absolute_path, "r", encoding="utf-8", errors="replace") as handle:
            text_content = handle.read()

        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, (offset or 1) - 1)
        start_line_display = start_line + 1
        if start_line >= len(all_lines):
            raise ValueError(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")

        user_limited_lines: Optional[int] = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])

        truncation = truncate_head(selected)
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(byte_length(all_lines[start_line]))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} "
                f"limit. Use bash to inspect this line.]"
            )
            details: Optional[TruncationResult] = truncation
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = truncation
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            details = None
        else:
            output_text = truncation.content
            details = None

        return AgentToolResult(content=[TextContent(text=output_text)], details=details)

    return AgentTool(
        name="read",
        label="read",
        description=READ_DESCRIPTION,
        parameters=READ_SCHEMA,
        execute=execute,
    )


def byte_length(text: str) -> int:
    return len(text.encode("utf-8"))
