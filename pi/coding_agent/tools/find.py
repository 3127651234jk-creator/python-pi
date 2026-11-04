"""Find files with ripgrep's ignore-aware glob traversal."""

import os
import fnmatch

from ...agent.types import AgentTool
from .common import resolve_to_cwd
from .search import positive_limit, run_rg

FIND_PROMPT_SNIPPET = "Find files by glob pattern, respecting .gitignore"


def matches_glob(path: str, pattern: str) -> bool:
    path = path.replace("\\", "/").removeprefix("./")
    pattern = pattern.replace("\\", "/").removeprefix("./")
    if "/" not in pattern:
        return fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], pattern)

    def match(parts, patterns):
        if not patterns:
            return not parts
        if patterns[0] == "**":
            return match(parts, patterns[1:]) or bool(parts) and match(parts[1:], patterns)
        return bool(parts) and fnmatch.fnmatchcase(parts[0], patterns[0]) and match(parts[1:], patterns[1:])

    return match(path.split("/"), pattern.split("/"))


def create_find_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        root = resolve_to_cwd(params.get("path", "."), cwd)
        if not os.path.isdir(root):
            raise NotADirectoryError(root)
        limit = positive_limit(params.get("limit"), 1000)
        args = ["--no-config", "--files", "--hidden", "--no-require-git", "--sort=path",
                "--glob=!.git/**", "--glob=!node_modules/**", "--", "."]
        return await run_rg(root, args, limit, signal, include=lambda path: matches_glob(path, params["pattern"]))

    return AgentTool(name="find", description="Find files by glob pattern (e.g. '*.py' or 'src/**/*.py'). Paths are relative to the search directory. Respects .gitignore, includes hidden files, skips .git and node_modules. Defaults: 1000 files, 50KB output. Requires rg on PATH.",
                     parameters={"type": "object", "properties": {"pattern": {"type": "string"},
                         "path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["pattern"]}, execute=execute)
