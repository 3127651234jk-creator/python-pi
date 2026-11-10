"""edit tool: precise multi-edit file modification with exact text replacement.

Python port of ``packages/coding-agent/src/core/tools/edit.ts``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ...ai.types import TextContent
from ...agent.types import AgentTool, AgentToolResult
from .common import resolve_to_cwd, with_file_mutation_queue
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
)

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
                            "Exact text for one targeted replacement. It must be unique in the original "
                            "file and must not overlap with any other edits[].oldText in the same call."
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

EDIT_PROMPT_SNIPPET = "Make precise file edits with exact text replacement, including multiple disjoint edits in one call"
EDIT_PROMPT_GUIDELINES = [
    "Use edit for precise changes (edits[].oldText must match exactly)",
    "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
    "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
    "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
]


def prepare_edit_arguments(args: Any) -> Any:
    """Compatibility shim: some models send edits as a JSON string or a single
    edit object, or use the legacy top-level oldText/newText."""
    if not isinstance(args, dict):
        return args
    args = dict(args)

    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            import json

            parsed = json.loads(edits)
            if isinstance(parsed, list):
                args["edits"] = parsed
            elif isinstance(parsed, dict) and "oldText" in parsed and "newText" in parsed:
                args["edits"] = [parsed]
        except (ValueError, ImportError):
            pass
    elif isinstance(edits, dict) and "oldText" in edits and "newText" in edits:
        args["edits"] = [edits]

    if isinstance(args.get("oldText"), str) and isinstance(args.get("newText"), str):
        edit_list = list(args.get("edits") or [])
        edit_list.append({"oldText": args.pop("oldText"), "newText": args.pop("newText")})
        args["edits"] = edit_list
    return args


def _validate_edit_input(params: Dict[str, Any]) -> Any:
    edits = params.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    return params


def create_edit_tool(cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: Any,
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        params = _validate_edit_input(params)
        path: str = params["path"]
        edits = [Edit(old_text=e["oldText"], new_text=e["newText"]) for e in params["edits"]]
        absolute_path = resolve_to_cwd(path, cwd)

        def check_aborted() -> None:
            if signal is not None and getattr(signal, "aborted", False):
                from ...util import AbortError

                raise AbortError("Operation aborted")

        async def mutate() -> AgentToolResult:
            check_aborted()
            if not os.path.isfile(absolute_path):
                raise FileNotFoundError(f"Could not edit file: {path}. File does not exist.")
            check_aborted()

            with open(absolute_path, "r", encoding="utf-8", newline="") as handle:
                raw_content = handle.read()
            check_aborted()

            bom = raw_content.startswith("\ufeff")  # UTF-8 BOM
            text = raw_content[1:] if bom else raw_content
            original_ending = detect_line_ending(text)
            normalized_content = normalize_to_lf(text)
            base_content, new_content = apply_edits_to_normalized_content(normalized_content, edits, path)
            check_aborted()

            final_content = ("\ufeff" if bom else "") + restore_line_endings(new_content, original_ending)
            encoded_content = final_content.encode("utf-8")
            with open(absolute_path, "wb") as handle:
                handle.write(encoded_content)
            check_aborted()

            diff, first_changed_line = generate_diff_string(base_content, new_content)
            patch = generate_unified_patch(path, base_content, new_content)
            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details={"diff": diff, "patch": patch, "firstChangedLine": first_changed_line},
            )

        return await with_file_mutation_queue(absolute_path, mutate)

    return AgentTool(
        name="edit",
        label="edit",
        description=EDIT_DESCRIPTION,
        parameters=EDIT_SCHEMA,
        execute=execute,
        prepare_arguments=prepare_edit_arguments,
    )
