"""System prompt construction and project context loading.

Python port of ``packages/coding-agent/src/core/system-prompt.ts``, simplified:
project context comes from AGENTS.md / CLAUDE.md / CONTEXT.md in the working
directory. The CLI attaches the discovered skills as a replayable system section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CONTEXT_FILE_NAMES = ["AGENTS.md", "CLAUDE.md", "CONTEXT.md"]


def _package_root() -> Path:
    """The directory containing the ``pi`` package (documentation lives here)."""
    return Path(__file__).resolve().parents[1]


def get_readme_path() -> str:
    return str(_package_root() / "README.md")


def get_docs_path() -> str:
    return str(_package_root() / "docs")


def get_examples_path() -> str:
    return str(_package_root() / "examples")


PI_DOCUMENTATION_INTRO = (
    "Pi documentation (read only when the user asks about pi itself, its SDK, "
    "extensions, themes, skills, or TUI):"
)


def build_documentation_section() -> str:
    """The <docs> section body; paths are listed even when files are absent."""
    return "\n".join([
        PI_DOCUMENTATION_INTRO,
        f"- Main documentation: {get_readme_path()}",
        f"- Additional docs: {get_docs_path()}",
        f"- Examples: {get_examples_path()} (extensions, custom tools, SDK)",
        "- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing",
        "- Always read pi .md files completely and follow links to related docs",
    ])


def documentation_available() -> bool:
    root = _package_root()
    return (root / "README.md").exists() or (root / "docs").is_dir() or (root / "examples").is_dir()


@dataclass
class SystemPromptOptions:
    custom_prompt: Optional[str] = None
    selected_tools: List[str] = field(default_factory=lambda: ["read", "bash", "edit", "write"])
    tool_snippets: Dict[str, str] = field(default_factory=dict)
    tool_guidelines: Dict[str, List[str]] = field(default_factory=dict)
    prompt_guidelines: List[str] = field(default_factory=list)
    append_system_prompt: str = ""
    cwd: str = os.getcwd()
    context_files: Optional[List[dict]] = None
    # None auto-detects packaged documentation; True forces the <docs> section
    # (eval harness), False removes it.
    include_pi_documentation: Optional[bool] = None


def load_context_files(cwd: str) -> List[dict]:
    """Load project instruction files (AGENTS.md and friends) from ``cwd``."""
    files: List[dict] = []
    for name in CONTEXT_FILE_NAMES:
        path = os.path.join(cwd, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    files.append({"path": name, "content": handle.read()})
            except OSError:
                pass
    return files


def build_rules(options: SystemPromptOptions) -> str:
    rules: List[str] = []
    seen = set()

    def add_rule(rule: str) -> None:
        rule = rule.strip()
        if rule and rule not in seen:
            seen.add(rule)
            rules.append(rule)

    has_bash = "bash" in options.selected_tools
    has_search_tools = any(t in options.selected_tools for t in ("grep", "find", "ls"))
    if has_bash and not has_search_tools:
        add_rule("Use bash for file operations like ls, rg, find")
    for name in options.selected_tools:
        for rule in options.tool_guidelines.get(name, []):
            add_rule(rule)
    for rule in options.prompt_guidelines:
        add_rule(rule)
    add_rule("Be concise in your responses")
    add_rule("Show file paths clearly when working with files")
    return "\n".join(f"- {rule}" for rule in rules)


def build_system_prompt(options: SystemPromptOptions) -> str:
    """Build the ordered system prompt text."""
    if options.context_files is None:
        options.context_files = load_context_files(options.cwd)

    if options.custom_prompt:
        preamble = options.custom_prompt
    else:
        preamble = (
            "You are an expert coding assistant operating inside pi, a coding agent harness. "
            "You help users by reading files, executing commands, editing code, and writing new files."
        )

    sections: List[str] = [preamble]

    visible_tools = [name for name in options.selected_tools if name in options.tool_snippets]
    tools_text = "\n".join(f"- {name}: {options.tool_snippets[name]}" for name in visible_tools) or "(none)"
    sections.append(
        f"<tools>\n{tools_text}\n\n"
        "In addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>"
    )
    sections.append(f"<rules>\n{build_rules(options)}\n</rules>")

    include_docs = options.include_pi_documentation
    if include_docs is None:
        include_docs = documentation_available()
    if include_docs:
        sections.append(f"<docs>\n{build_documentation_section()}\n</docs>")

    if options.append_system_prompt:
        sections.append(f"<addendum>\n{options.append_system_prompt}\n</addendum>")

    if options.context_files:
        rendered = "\n\n".join(
            f'<project_instructions path="{f["path"]}">\n{f["content"]}\n</project_instructions>'
            for f in options.context_files
        )
        sections.append(f"<project_context>\nProject-specific instructions and guidelines:\n\n{rendered}\n</project_context>")

    sections.append(f"<cwd>\n{options.cwd.replace(os.sep, '/')}\n</cwd>")
    return "\n\n".join(sections)
