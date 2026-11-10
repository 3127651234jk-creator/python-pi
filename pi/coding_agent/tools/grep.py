"""Search file contents without shell interpolation, ripgrep when available."""

import os
import re
import shutil

from ...agent.types import AgentTool
from .common import resolve_to_cwd
from .search import grep_python, matches_glob, positive_limit, run_rg

GREP_PROMPT_SNIPPET = "Search file contents by regex or literal text, respecting .gitignore"


def create_grep_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        root = resolve_to_cwd(params.get("path", "."), cwd)
        if not os.path.exists(root):
            raise FileNotFoundError(root)
        limit = positive_limit(params.get("limit"), 100)
        context = params.get("context", 0)
        if type(context) is not int or context < 0:
            raise ValueError("context must be a nonnegative integer")
        if shutil.which("rg") is None:
            return await grep_python(root, params["pattern"], glob=params.get("glob"),
                                     ignore_case=bool(params.get("ignoreCase")),
                                     literal=bool(params.get("literal")),
                                     context=context, limit=limit, signal=signal)
        args = ["--no-config", "--color=never", "--line-number", "--with-filename", "--no-heading",
                "--hidden", "--no-require-git", "--glob=!.git/**", "--glob=!node_modules/**", "--sort=path",
                "--max-columns=500", "--max-columns-preview", "--context", str(context)]
        if params.get("ignoreCase"):
            args.append("--ignore-case")
        if params.get("literal"):
            args.append("--fixed-strings")
        def include(line: str) -> bool:
            # A directory may itself contain a context-like separator (-123-).
            # Only a separator following an actual file ends the path field.
            for match in re.finditer(r"(?::\d+:|-\d+-)", line):
                filename = line[:match.start()]
                if not os.path.isfile(filename):
                    continue
                relative = os.path.relpath(filename, root) if os.path.isdir(root) else os.path.basename(filename)
                return matches_glob(relative, params["glob"])
            return False
        # -e protects patterns beginning with '-' from option interpretation.
        args.extend(["-e", params["pattern"], "--", root])
        return await run_rg(cwd, args, limit, signal, matches=True,
                            include=include if params.get("glob") else None)

    return AgentTool(name="grep", description="Search text files. Returns paths and line numbers; respects .gitignore, includes hidden files, skips .git and node_modules. Defaults: 100 matching lines, 50KB output. Long lines are truncated. Uses ripgrep when installed, otherwise a built-in scan.",
                     parameters={"type": "object", "properties": {
                         "pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
                         "ignoreCase": {"type": "boolean"}, "literal": {"type": "boolean"},
                         "context": {"type": "integer"}, "limit": {"type": "integer"},
                     }, "required": ["pattern"]}, execute=execute)
