"""write tool: create or overwrite a file.

Python port of ``packages/coding-agent/src/core/tools/write.ts``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ...ai.types import TextContent
from ...agent.types import AgentTool, AgentToolResult
from .common import resolve_to_cwd, with_file_mutation_queue

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}

WRITE_DESCRIPTION = (
    "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
    "Automatically creates parent directories."
)

WRITE_PROMPT_SNIPPET = "Create or overwrite files"
WRITE_PROMPT_GUIDELINES = ["Use write only for new files or complete rewrites."]


def create_write_tool(cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: Any,
        signal: Optional[Any] = None,
        on_update: Optional[Any] = None,
    ) -> AgentToolResult:
        path: str = params["path"]
        content: str = params["content"]
        absolute_path = resolve_to_cwd(path, cwd)
        directory = os.path.dirname(absolute_path)

        def check_aborted() -> None:
            if signal is not None and getattr(signal, "aborted", False):
                from ...util import AbortError

                raise AbortError("Operation aborted")

        async def mutate() -> AgentToolResult:
            check_aborted()
            if directory:
                os.makedirs(directory, exist_ok=True)
            check_aborted()
            with open(absolute_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            check_aborted()
            return AgentToolResult(content=[TextContent(text=f"Successfully wrote to {path}")])

        return await with_file_mutation_queue(absolute_path, mutate)

    return AgentTool(
        name="write",
        label="write",
        description=WRITE_DESCRIPTION,
        parameters=WRITE_SCHEMA,
        execute=execute,
    )
