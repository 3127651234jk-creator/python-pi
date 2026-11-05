"""Discover Markdown skills and expand prompt commands without executing code."""

import os
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .config import Configuration, read_json

RESERVED_COMMANDS = {
    "help", "exit", "quit", "model", "models", "login", "logout", "settings", "reload",
    "clear", "new", "resume", "session", "name", "tree", "fork", "clone", "export", "usage",
    "skills", "prompts", "mcp", "theme", "keys", "compact", "thinking", "package",
    "extensions", "image", "sync-models", "evals", "export-ts", "trust", "bug",
}


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    disable_model_invocation: bool = False


@dataclass
class PromptTemplate:
    name: str
    description: str
    content: str
    path: Path
    argument_hint: str = ""


def read_document(path: Path) -> Tuple[dict, str]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        return {}, text
    end = re.search(r"^---[ \t]*$", text[4:], re.MULTILINE)
    if end is None:
        raise ValueError("Unterminated YAML frontmatter")
    try:
        metadata = yaml.safe_load(text[4:4 + end.start()]) or {}
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML frontmatter") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Frontmatter must be a mapping")
    return metadata, text[4 + end.end():].lstrip("\n")


def parse_arguments(text: str) -> List[str]:
    # Match the TS quoting rules; backslashes remain literal for Windows paths.
    args = []
    current = ""
    quote = None
    started = False
    for char in text:
        if quote:
            if char == quote:
                quote = None
            else:
                current += char
        elif char in ("'", '"'):
            quote = char
            started = True
        elif char.isspace():
            if started:
                args.append(current)
                current = ""
                started = False
        else:
            current += char
            started = True
    if quote:
        raise ValueError("Unterminated quoted argument")
    if started:
        args.append(current)
    return args


def substitute_arguments(content: str, args: List[str]) -> str:
    def substitute(match):
        default_target, default, start, length, simple = match.groups()
        if default_target:
            value = " ".join(args) if default_target in ("@", "ARGUMENTS") else (
                args[int(default_target) - 1] if 0 < int(default_target) <= len(args) else ""
            )
            return value or default
        if start is not None:
            index = max(0, int(start) - 1)
            return " ".join(args[index:index + int(length)] if length is not None else args[index:])
        if simple in ("@", "ARGUMENTS"):
            return " ".join(args)
        index = int(simple) - 1
        return args[index] if 0 <= index < len(args) else ""
    return re.sub(r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)", substitute, content)


class ResourceCatalog:
    def __init__(self, config: Configuration, home: Optional[Path] = None) -> None:
        self.config = config
        self.home = home or Path.home()
        self.skills: Dict[str, Skill] = {}
        self.prompts: Dict[str, PromptTemplate] = {}
        self.diagnostics: List[str] = []
        self.reload()

    def reload(self) -> None:
        self.skills = {}
        self.prompts = {}
        self.diagnostics = []
        trusted = self.config.project_trusted
        # First discovered name wins: explicit project/user paths, then defaults.
        skill_paths = self.configured_paths("skills", trusted)
        prompt_paths = self.configured_paths("prompts", trusted)
        cwd = self.config.cwd.resolve()
        if self.config.settings.get("include_default_resources", True):
            if trusted:
                skill_paths.extend([cwd / ".pi" / "skills"])
                for parent in (cwd, *cwd.parents):
                    skill_paths.append(parent / ".agents" / "skills")
                    if (parent / ".git").exists():
                        break
                prompt_paths.extend([cwd / ".pi" / "prompts"])
            skill_paths.extend([self.config.agent_dir / "skills", self.home / ".agents" / "skills"])
            prompt_paths.extend([self.config.agent_dir / "prompts"])

        seen = set()
        for root in skill_paths:
            for path in self.discover(root, skills=True):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    metadata, body = read_document(path)
                    if not body.strip():
                        raise ValueError("Skill instructions are empty")
                    name = metadata.get("name", path.parent.name)
                    description = metadata.get("description")
                    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
                        raise ValueError("Skill name must use lowercase letters, numbers and single hyphens, up to 64 characters")
                    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
                        raise ValueError("Skill description must contain 1-1024 characters")
                    disabled = metadata.get("disable-model-invocation", False)
                    if not isinstance(disabled, bool):
                        raise ValueError("disable-model-invocation must be boolean")
                    if name in self.skills:
                        raise ValueError(f"Duplicate skill '{name}'; keeping {self.skills[name].path}")
                    self.skills[name] = Skill(name, description.strip(), resolved, disabled)
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    self.diagnostics.append(f"{path}: {exc}")
        seen = set()
        for root in prompt_paths:
            for path in self.discover(root, skills=False):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    metadata, body = read_document(path)
                    name = path.stem
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in RESERVED_COMMANDS:
                        raise ValueError("Invalid or reserved prompt command name")
                    if not body.strip():
                        raise ValueError("Prompt template body is empty")
                    description = metadata.get("description") or next(line.strip()[:80] for line in body.splitlines() if line.strip())
                    hint = metadata.get("argument-hint", "")
                    if not isinstance(description, str) or not isinstance(hint, str):
                        raise ValueError("Template description and argument-hint must be strings")
                    if name in self.prompts:
                        raise ValueError(f"Duplicate template '{name}'; keeping {self.prompts[name].path}")
                    self.prompts[name] = PromptTemplate(name, description, body, resolved, hint)
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    self.diagnostics.append(f"{path}: {exc}")

    def configured_paths(self, kind: str, trusted: bool = True) -> List[Path]:
        paths = []
        for base in ((self.config.cwd / ".pi", self.config.agent_dir) if trusted else (self.config.agent_dir,)):
            for raw in read_json(base / "settings.json").get(kind, []):
                path = Path(raw).expanduser()
                paths.append(path if path.is_absolute() else base / path)
        return paths

    def discover(self, root: Path, skills: bool) -> List[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root] if root.suffix.lower() == ".md" else []
        if not skills:
            try:
                return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".md")
            except OSError as exc:
                self.diagnostics.append(f"{root}: {exc}")
                return []
        paths = []
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda exc: self.diagnostics.append(str(exc))):
            dirs[:] = sorted(name for name in dirs if name not in (".git", "node_modules", "__pycache__"))
            if "SKILL.md" in files:
                paths.append(Path(directory) / "SKILL.md")
        return paths

    def system_prompt(self) -> str:
        skills = [skill for skill in self.skills.values() if not skill.disable_model_invocation]
        if not skills:
            return ""
        rows = ["When a task matches a skill description, use read to load its SKILL.md before following it. Resolve supporting files relative to that skill's directory."]
        for skill in skills:
            rows.append(f'<skill name="{escape(skill.name, quote=True)}" path="{escape(str(skill.path), quote=True)}">{escape(skill.description)}</skill>')
        return "<available_skills>\n" + "\n".join(rows) + "\n</available_skills>"

    def expand(self, text: str) -> str:
        match = re.match(r"^/([^\s]+)(?:\s+([\s\S]*))?$", text)
        if match is None:
            return text
        command, argument = match.group(1), match.group(2) or ""
        if command.startswith("skill:"):
            name = command[6:]
            skill = self.skills.get(name)
            if skill is None:
                raise ValueError(f"Unknown skill: {name}")
            _, body = read_document(skill.path)
            return (f"Skill: {skill.name}\nSkill directory: {skill.path.parent}\n"
                    f"Resolve relative references from that directory.\n\n{body}\n\nUser request:\n{argument}")
        template = self.prompts.get(command)
        if template is None:
            return text
        return substitute_arguments(template.content, parse_arguments(argument))
