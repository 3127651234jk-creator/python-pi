"""write tool (port of ``tools/write.ts``)."""

from __future__ import annotations

from ..env import Context, get_or_throw
from .base import ToolExecutionApi, ToolExecutionResult, ToolRegistration, define_tool, require_env
from .mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path

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


def create_write_tool() -> ToolRegistration:
    async def execute(args: dict, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        path: str = args["path"]
        content: str = args["content"]
        env = require_env(api)
        absolute_path = await resolve_tool_path(env, path, context)

        async def mutate() -> ToolExecutionResult:
            if context.aborted:
                raise RuntimeError("Operation aborted")
            get_or_throw(await env.write_file(absolute_path, content, context))
            if context.aborted:
                raise RuntimeError("Operation aborted")
            return ToolExecutionResult(content=[{"type": "text", "text": f"Successfully wrote to {path}"}])

        return await with_file_mutation_queue(env, absolute_path, mutate, context)

    return define_tool(name="write", description=WRITE_DESCRIPTION, parameters=WRITE_SCHEMA, execute=execute)
