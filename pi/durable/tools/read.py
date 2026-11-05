"""read tool (port of ``tools/read.ts``).

Reads text files through the environment's ``BinaryReader``; remarks about
truncation and continuation are diagnostics, the content is only file text.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..env import BinaryReader, Context, FileInfo, get_or_throw, range_decoder, starts_with_bom
from ..output import character_end
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size, truncate_head_of, utf8_byte_length
from .base import ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration, define_tool, require_env
from .image import detect_supported_image_mime_type_of
from .path_utils import resolve_read_tool_path

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}

READ_CHUNK = 64 * 1024

READ_DESCRIPTION = (
    f"Read the contents of a text file. Output is truncated to {DEFAULT_MAX_LINES} lines or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. "
    "When you need the full file, continue with offset until complete."
)


def _slice_index(value: float) -> int:
    """``Array.prototype.slice``'s index conversion: NaN is 0, other values truncate toward zero."""
    if isinstance(value, float) and math.isnan(value):
        return 0
    return math.trunc(value)


async def _read_head(reader: BinaryReader, start: int, end: int, skip_bom: bool, context: Context) -> str:
    """The decoded start of bytes ``[start, end)``, decoded as part of the whole file."""
    decoder = range_decoder()
    text = ""
    newlines = 0
    position = 3 if (skip_bom and start == 0) else start
    while position < end:
        data: bytes = get_or_throw(await reader.read(position, min(READ_CHUNK, end - position), context))
        if len(data) == 0:
            break
        position += len(data)
        decoded = decoder.decode(data)
        text += decoded
        newlines += decoded.count("\n")
        if newlines >= DEFAULT_MAX_LINES or utf8_byte_length(text) > DEFAULT_MAX_BYTES + 1:
            return text
    return text + decoder.decode()


def create_read_tool() -> ToolRegistration:
    async def execute(args: dict, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        path: str = args["path"]
        offset = args.get("offset")
        limit = args.get("limit")
        env = require_env(api)
        absolute_path = await resolve_read_tool_path(env, path, context)
        reader: BinaryReader = get_or_throw(await env.open_binary_reader(absolute_path, None, context))
        try:
            # A concurrent writer can change the file between the scan and the reads; retry once from a fresh scan.
            attempt = 0
            while True:
                before: FileInfo = get_or_throw(await reader.info(context))
                result = await _read_text(reader, before, path, offset, limit, context)
                after: FileInfo = get_or_throw(await reader.info(context))
                if after.size == before.size and after.mtime_ms == before.mtime_ms:
                    return result
                if attempt == 1:
                    raise RuntimeError(f"{path} changed while it was read")
                attempt += 1
        finally:
            await reader.close(context)

    return define_tool(
        name="read",
        description=READ_DESCRIPTION,
        parameters=READ_SCHEMA,
        execute=execute,
    )


async def _read_text(
    reader: BinaryReader,
    info: FileInfo,
    path: str,
    offset: Optional[float],
    limit: Optional[float],
    context: Context,
) -> ToolExecutionResult:
    async def read_bytes(position: int, length: int) -> bytes:
        return get_or_throw(await reader.read(position, length, context))

    class _Source:
        size = info.size

        async def read(self, position: int, length: int) -> bytes:
            return await read_bytes(position, length)

    mime_type = await detect_supported_image_mime_type_of(_Source())
    if mime_type:
        # Image content is not supported yet.
        return ToolExecutionResult(
            content=[],
            is_error=True,
            diagnostics=[
                ToolDiagnostic(
                    severity="error",
                    code="unsupported_image",
                    message=f"{path} is an image ({mime_type}); reading images is not supported",
                )
            ],
        )

    start_line = max(0, offset - 1) if offset else 0
    start_line_display = start_line + 1
    # Lines are selected like ``all_lines[start:stop]``, which truncates fractional indices.
    slice_start = _slice_index(start_line)
    # One pass finds the line count and the selection; a selection past the last line ends with it, as
    # slicing does, and an empty one is scanned as one line and then ignored.
    scan_start = slice_start if abs(slice_start) < 2 ** 53 else 0
    requested_end = (
        None if limit is None else max(scan_start + 1, _slice_index(start_line + limit))
    )
    scan_end = requested_end if (requested_end is not None and abs(requested_end) < 2 ** 53) else None

    async def scan_of(end_line: Optional[int]):
        options: Dict[str, int] = {"start_line": scan_start}
        if end_line is not None:
            options["end_line"] = end_line
        return get_or_throw(await reader.scan_lines(options, context))

    scan = await scan_of(scan_end)
    total_file_lines = scan.newlines + 1
    if start_line >= total_file_lines:
        raise ValueError(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")

    user_limited_lines: Optional[float] = None
    selected_line_count = total_file_lines - slice_start
    if limit is not None:
        end_line = min(start_line + limit, total_file_lines)
        user_limited_lines = end_line - start_line
        # A negative end counts from the end of the lines, which only the line count tells.
        relative_end = _slice_index(end_line)
        slice_end = max(total_file_lines + relative_end, 0) if relative_end < 0 else relative_end
        selected_line_count = max(0, slice_end - slice_start)
        if selected_line_count > 0 and relative_end < 0:
            scan = await scan_of(slice_end)
    empty = selected_line_count == 0
    # Counted like ``truncate_head``: a trailing newline adds no line, and empty text has none.
    ends_with_newline = not empty and scan.last_line_start == scan.end and scan.last_line_start > scan.start
    totals = {
        "lines": 0 if empty or scan.selected_bytes == 0 else selected_line_count - (1 if ends_with_newline else 0),
        "bytes": 0 if empty else scan.selected_bytes,
    }
    first_bytes: bytes = get_or_throw(await reader.read(0, 3, context))
    head = "" if empty else await _read_head(reader, scan.start, scan.end, starts_with_bom(first_bytes), context)

    truncation = truncate_head_of(head, totals)
    head_text = truncation.content
    diagnostics: List[ToolDiagnostic] = []
    output_text = head_text
    details: Optional[Dict[str, Any]] = None
    if truncation.first_line_exceeds_limit:
        # Show the start of the line, cut at the byte limit on a character boundary.
        integral = float(start_line).is_integer()
        line_text = head.split("\n")[0] if integral else ""
        line_bytes = line_text.encode("utf-8")
        line_size = scan.first_line_bytes if integral else 0
        end = character_end(line_bytes, DEFAULT_MAX_BYTES)
        output_text = line_bytes[:end].decode("utf-8", "replace")
        diagnostics.append(
            ToolDiagnostic(
                severity="warn",
                code="truncated",
                message=(
                    f"Line {start_line_display} is {format_size(line_size)}, exceeds the "
                    f"{format_size(DEFAULT_MAX_BYTES)} limit; showing its first {format_size(end)}. "
                    f"Use bash: sed -n '{start_line_display}p' {path} | tail -c +{end + 1}"
                ),
            )
        )
        details = {
            "truncation": _truncation_details(truncation, output_bytes=end, output_lines=1),
        }
    elif truncation.truncated:
        end_line_display = start_line_display + truncation.output_lines - 1
        next_offset = end_line_display + 1
        limit_text = "" if truncation.truncated_by == "lines" else f" ({format_size(DEFAULT_MAX_BYTES)} limit)"
        diagnostics.append(
            ToolDiagnostic(
                severity="info",
                code="truncated",
                message=(
                    f"Showing lines {start_line_display}-{end_line_display} of {total_file_lines}{limit_text}. "
                    f"Use offset={next_offset} to continue."
                ),
            )
        )
        details = {"truncation": _truncation_details(truncation)}
    elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
        remaining = total_file_lines - (start_line + user_limited_lines)
        next_offset = start_line + user_limited_lines + 1
        diagnostics.append(
            ToolDiagnostic(
                severity="info",
                message=f"{remaining} more lines in file. Use offset={next_offset} to continue.",
            )
        )

    return ToolExecutionResult(
        content=[] if output_text == "" else [{"type": "text", "text": output_text}],
        details=details,
        diagnostics=diagnostics,
    )


def _truncation_details(truncation: TruncationResult, output_bytes: Optional[int] = None, output_lines: Optional[int] = None) -> dict:
    data = {
        "truncated": truncation.truncated,
        "truncatedBy": truncation.truncated_by,
        "totalLines": truncation.total_lines,
        "totalBytes": truncation.total_bytes,
        "outputLines": truncation.output_lines if output_lines is None else output_lines,
        "outputBytes": truncation.output_bytes if output_bytes is None else output_bytes,
        "firstLineExceedsLimit": truncation.first_line_exceeds_limit,
    }
    return data
