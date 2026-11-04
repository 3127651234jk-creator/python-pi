"""List directory entries, including dotfiles."""

import asyncio
import os

from ...agent.types import AgentTool
from .common import resolve_to_cwd
from .search import bounded_result, positive_limit

LS_PROMPT_SNIPPET = "List directory entries, including hidden files and directory markers"


def create_ls_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        if signal:
            signal.throw_if_aborted()
        root = resolve_to_cwd(params.get("path", "."), cwd)
        limit = positive_limit(params.get("limit"), 500)
        entries = await asyncio.to_thread(os.listdir, root)
        entries.sort(key=lambda name: (name.casefold(), name))
        lines = []
        for entry in entries[:limit]:
            if signal:
                signal.throw_if_aborted()
            lines.append(entry + ("/" if os.path.isdir(os.path.join(root, entry)) else ""))
        return bounded_result(lines, len(entries) > limit, "(empty directory)")

    return AgentTool(name="ls", description="List directory entries alphabetically, including dotfiles. Directories end in '/'. Defaults: 500 entries and 50KB output.",
                     parameters={"type": "object", "properties": {"path": {"type": "string"},
                         "limit": {"type": "integer"}}}, execute=execute)
