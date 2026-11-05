"""Durable coding tools: ``read``, ``write``, ``edit``, and ``bash``.

Python port of ``packages/durable/src/tools``. These tools run against the
portable ``ExecutionEnv`` (local host, container, or remote), not the host
process directly; nothing installs them automatically.
"""

from __future__ import annotations

from ..env import ExecutionEnv
from .base import (
    ToolDiagnostic,
    ToolExecutionApi,
    ToolExecutionResult,
    ToolRegistration,
    define_tool,
    require_env,
)
from .bash import BASH_DESCRIPTION, DEFAULT_OUTPUT_LIMITS, create_bash_tool
from .edit import EDIT_DESCRIPTION, create_edit_tool, prepare_edit_arguments
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .image import detect_supported_image_mime_type, detect_supported_image_mime_type_of
from .mutation_queue import with_file_mutation_queue
from .path_utils import resolve_read_tool_path, resolve_tool_path
from .read import READ_DESCRIPTION, create_read_tool
from .write import WRITE_DESCRIPTION, create_write_tool

__all__ = [
    "BASH_DESCRIPTION",
    "DEFAULT_OUTPUT_LIMITS",
    "EDIT_DESCRIPTION",
    "Edit",
    "READ_DESCRIPTION",
    "ToolDiagnostic",
    "ToolExecutionApi",
    "ToolExecutionResult",
    "ToolRegistration",
    "WRITE_DESCRIPTION",
    "apply_edits_to_normalized_content",
    "coding_tools",
    "create_bash_tool",
    "create_edit_tool",
    "create_read_tool",
    "create_write_tool",
    "define_tool",
    "detect_line_ending",
    "detect_supported_image_mime_type",
    "detect_supported_image_mime_type_of",
    "generate_diff_string",
    "generate_unified_patch",
    "normalize_for_fuzzy_match",
    "normalize_to_lf",
    "prepare_edit_arguments",
    "require_env",
    "resolve_read_tool_path",
    "resolve_tool_path",
    "restore_line_endings",
    "strip_bom",
    "with_file_mutation_queue",
]


def coding_tools() -> list:
    """The four coding tools as a list; a host picks them up explicitly (parity with ``CodingTools``)."""
    return [create_read_tool(), create_write_tool(), create_edit_tool(), create_bash_tool()]
