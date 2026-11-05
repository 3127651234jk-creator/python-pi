"""edit tool (port of ``tools/edit.ts``): exact text replacement in one file."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..env import Context, FileError
from .base import ToolExecutionApi, ToolExecutionResult, ToolRegistration, define_tool, require_env
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
        "edits": {
            "type": "array",
            "description": (
                "One or more targeted replacements. Each edit is matched against the original file, not "
                "incrementally. Do not include overlapping or nested edits. If two changes touch the same "
                "block or nearby lines, merge them into one edit instead."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "oldText": {
                        "type": "string",
                        "description": (
                            "Exact text for one targeted replacement. It must be unique in the original file "
                            "and must not overlap with any other edits[].oldText in the same call."
                        ),
                    },
                    "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                },
                "required": ["oldText", "newText"],
            },
        },
    },
    "required": ["path", "edits"],
}

EDIT_DESCRIPTION = (
    "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
    "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
    "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
    "regions just to connect distant changes."
)


def _is_single_edit_input(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("oldText"), str) and isinstance(value.get("newText"), str)


def prepare_edit_arguments(value: Any) -> Any:
    """Repair shapes models commonly send: edits as a JSON string, one edit object, or a top-level pair.

    Works on a copy; the call's arguments stay unchanged.
    """
    if not isinstance(value, dict):
        return value
    args: Dict[str, Any] = dict(value)
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            args["edits"] = parsed
        elif _is_single_edit_input(parsed):
            args["edits"] = [parsed]
    elif _is_single_edit_input(edits):
        args["edits"] = [edits]

    old_text = args.get("oldText")
    new_text = args.get("newText")
    if not (isinstance(old_text, str) and isinstance(new_text, str)):
        return args
    rest = {k: v for k, v in args.items() if k not in ("oldText", "newText")}
    edit_list = list(rest.get("edits")) if isinstance(rest.get("edits"), list) else []
    edit_list.append({"oldText": old_text, "newText": new_text})
    rest["edits"] = edit_list
    return rest


def _validate_edit_input(value: Dict[str, Any]) -> Dict[str, Any]:
    edits = value.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    return {"path": value["path"], "edits": edits}


def _raise_edit_access_error(path: str, error: FileError) -> Exception:
    raise RuntimeError(f"Could not edit file: {path}. Error code: {error.code}.")


def create_edit_tool() -> ToolRegistration:
    async def execute(args: dict, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        validated = _validate_edit_input(args)
        path: str = validated["path"]
        raw_edits: List[dict] = validated["edits"]
        edits = [Edit(edit["oldText"], edit["newText"]) for edit in raw_edits]
        env = require_env(api)
        absolute_path = await resolve_tool_path(env, path, context)

        async def mutate() -> ToolExecutionResult:
            if context.aborted:
                raise RuntimeError("Operation aborted")
            info = await env.file_info(absolute_path, context)
            if not info.ok:
                _raise_edit_access_error(path, info.error)
            if info.value.kind not in ("file", "symlink"):
                raise ValueError(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, context)
            if not read_result.ok:
                _raise_edit_access_error(path, read_result.error)
            if context.aborted:
                raise RuntimeError("Operation aborted")

            bom, content = strip_bom(read_result.value)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            base_content, new_content = applied["base"], applied["new"]
            if context.aborted:
                raise RuntimeError("Operation aborted")

            final_content = bom + restore_line_endings(new_content, original_ending)
            write_result = await env.write_file(absolute_path, final_content, context)
            if not write_result.ok:
                _raise_edit_access_error(path, write_result.error)
            if context.aborted:
                raise RuntimeError("Operation aborted")

            diff_result = generate_diff_string(base_content, new_content)
            details: Dict[str, Any] = {
                "diff": diff_result["diff"],
                "patch": generate_unified_patch(path, base_content, new_content),
            }
            if diff_result["first_changed_line"] is not None:
                details["firstChangedLine"] = diff_result["first_changed_line"]
            return ToolExecutionResult(
                content=[
                    {
                        "type": "text",
                        "text": f"Successfully replaced {len(edits)} block(s) in {path}.",
                    }
                ],
                details=details,
            )

        return await with_file_mutation_queue(env, absolute_path, mutate, context)

    return define_tool(
        name="edit",
        description=EDIT_DESCRIPTION,
        parameters=EDIT_SCHEMA,
        execute=execute,
        prepare_arguments=prepare_edit_arguments,
    )
