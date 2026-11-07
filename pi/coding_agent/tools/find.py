"""Find files by glob pattern, ripgrep when available."""

import os
import shutil

from ...agent.types import AgentTool
from .common import resolve_to_cwd
from .search import find_python, matches_glob, positive_limit, run_rg

FIND_PROMPT_SNIPPET = "Find files by glob pattern, respecting .gitignore"


def create_find_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        root = resolve_to_cwd(params.get("path", "."), cwd)
        if not os.path.isdir(root):
            raise NotADirectoryError(root)
        limit = positive_limit(params.get("limit"), 1000)
        if shutil.which("rg") is None:
            return await find_python(root, params["pattern"], limit, signal)
        args = ["--no-config", "--files", "--hidden", "--no-require-git", "--sort=path",
                "--glob=!.git/**", "--glob=!node_modules/**", "--", "."]
        return await run_rg(root, args, limit, signal, include=lambda path: matches_glob(path, params["pattern"]))

    return AgentTool(name="find", description="Find files by glob pattern (e.g. '*.py' or 'src/**/*.py'). Paths are relative to the search directory. Respects .gitignore, includes hidden files, skips .git and node_modules. Defaults: 1000 files, 50KB output. Uses ripgrep when installed, otherwise a built-in scan.",
                     parameters={"type": "object", "properties": {"pattern": {"type": "string"},
                         "path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["pattern"]}, execute=execute)
