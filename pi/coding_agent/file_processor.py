"""Process @file CLI arguments into text content and image attachments.

Port of ``packages/coding-agent/src/cli/file-processor.ts``. Images go through
``process_image``; the one-shot CLI passes ``auto_resize_images=False`` because
resize runs later against the request model's profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..ai.types import ImageContent
from .images import ProcessImageOptions, detect_supported_image_mime_type_from_file, process_image


@dataclass
class ProcessedFiles:
    text: str = ""
    images: List[ImageContent] = field(default_factory=list)


def strip_bom(text: str) -> str:
    return text[1:] if text.startswith("\ufeff") else text


def resolve_read_path(path: str, cwd: str) -> str:
    """Resolve a @file argument: ~ expansion, absolute or cwd-relative."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(cwd, expanded))


async def process_file_arguments(
    file_args: List[str], cwd: Optional[str] = None, auto_resize_images: bool = True
) -> ProcessedFiles:
    """Process @file arguments into inline text and image attachments.

    Raises FileNotFoundError for missing files and UnicodeDecodeError for
    non-UTF-8 text files, matching the TS behavior of exiting with an error.
    """
    cwd = cwd or os.getcwd()
    result = ProcessedFiles()
    for file_arg in file_args:
        absolute_path = resolve_read_path(file_arg, cwd)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"File not found: {absolute_path}")
        if os.path.getsize(absolute_path) == 0:
            continue

        mime_type = detect_supported_image_mime_type_from_file(absolute_path)
        if mime_type:
            with open(absolute_path, "rb") as handle:
                data = handle.read()
            processed = await process_image(
                data, mime_type, ProcessImageOptions(auto_resize_images=auto_resize_images)
            )
            if not processed.ok:
                result.text += f'<file name="{absolute_path}">{processed.message}</file>\n'
                continue
            result.images.append(ImageContent(data=processed.data, mime_type=processed.mime_type))
            if processed.hints:
                hint_lines = "\n".join(processed.hints)
                result.text += f'<file name="{absolute_path}">{hint_lines}</file>\n'
            else:
                result.text += f'<file name="{absolute_path}"></file>\n'
        else:
            with open(absolute_path, "rb") as handle:
                content = handle.read().decode("utf-8")
            result.text += f'<file name="{absolute_path}">\n{strip_bom(content)}\n</file>\n'
    return result
