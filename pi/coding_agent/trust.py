"""Project trust: gate project-local resources behind an explicit decision.

Port of ``trust-manager.ts`` and ``project-trust.ts``. Decisions persist in
``<agent_dir>/trust.json`` as ``{canonical path: true|false}``; ``get`` walks
up the directory tree and returns the nearest entry. Project resources that
require trust live under ``.pi`` (settings, mcp.json, extensions, skills,
prompts, themes, SYSTEM.md, APPEND_SYSTEM.md) plus ``.agents/skills`` in the
project or any ancestor.

Unlike the TS version there is no cross-process file lock: writes use the
atomic ``write_json`` replacement, which is sufficient for the single-process
Python CLI.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import read_json, write_json

TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES = (
    "settings.json", "mcp.json", "extensions", "skills", "prompts", "themes",
    "SYSTEM.md", "APPEND_SYSTEM.md",
)


def canonical(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def project_trust_parent_path(cwd: str) -> Optional[str]:
    path = Path(canonical(cwd))
    parent = path.parent
    return str(parent) if parent != path else None


def has_trust_requiring_project_resources(cwd: str, home: Optional[Path] = None) -> bool:
    """True when cwd carries project resources gated by project trust."""
    home = Path(home) if home else Path.home()
    user_agents_skills = str((home / ".agents" / "skills").resolve())
    current = Path(canonical(cwd))
    config_dir = current / ".pi"
    if any((config_dir / entry).exists() for entry in TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES):
        return True
    while True:
        agents_skills = current / ".agents" / "skills"
        if agents_skills.exists() and str(agents_skills.resolve()) != user_agents_skills:
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def trust_options(cwd: str) -> List[dict]:
    """Dialog options: label, decision, and trust-store updates when saved."""
    path = canonical(cwd)
    options = [{"label": "Trust", "trusted": True, "updates": [(path, True)]}]
    parent = project_trust_parent_path(cwd)
    if parent is not None:
        options.append({"label": f"Trust parent folder ({parent})", "trusted": True,
                        "updates": [(parent, True), (path, None)]})
    options.append({"label": "Trust (this session only)", "trusted": True, "updates": []})
    options.append({"label": "Do not trust", "trusted": False, "updates": [(path, False)]})
    options.append({"label": "Do not trust (this session only)", "trusted": False, "updates": []})
    return options


class TrustStore:
    def __init__(self, agent_dir) -> None:
        self.path = Path(agent_dir) / "trust.json"

    def _read(self) -> Dict[str, Optional[bool]]:
        try:
            data = read_json(self.path)
        except ValueError as exc:
            logging.getLogger(__name__).warning("%s", exc)
            return {}
        return {key: value for key, value in data.items() if isinstance(value, bool)}

    def get(self, cwd: str) -> Optional[bool]:
        """Nearest decision for cwd or an ancestor; None when undecided."""
        data = self._read()
        current = Path(canonical(cwd))
        while True:
            decision = data.get(str(current))
            if decision is not None:
                return decision
            parent = current.parent
            if parent == current:
                return None
            current = parent

    def set(self, cwd: str, decision: Optional[bool]) -> None:
        self.set_many([(cwd, decision)])

    def set_many(self, updates: List[Tuple[str, Optional[bool]]]) -> None:
        data = self._read()
        for path, decision in updates:
            key = canonical(path)
            if decision is None:
                data.pop(key, None)
            else:
                data[key] = decision
        write_json(self.path, {key: data[key] for key in sorted(data)})


async def resolve_project_trusted(cwd: str, store: TrustStore, choose, settings: Optional[dict] = None,
                                  override: Optional[bool] = None, has_ui: bool = False,
                                  notify=None) -> bool:
    """Resolve trust for cwd and persist a dialog decision.

    ``choose(title, values)`` is the async selector used for the prompt (TUI
    chooser or an ``input()`` fallback); it returns the chosen value or None.
    """
    if override is not None:
        return override
    if not has_trust_requiring_project_resources(cwd):
        return True
    decision = store.get(cwd)
    if decision is not None:
        return decision
    default = (settings or {}).get("default_project_trust", "ask")
    if default == "always":
        return True
    if default == "never":
        return False
    if not has_ui or choose is None:
        if notify is not None:
            notify(f"Project {cwd} has trust-requiring resources; start interactively to decide, "
                   "or pass --approve to trust it once.")
        return False
    options = trust_options(cwd)
    selected = await choose(
        "Trust project folder?\n" + canonical(cwd)
        + "\n\nThis allows pi to load .pi settings and resources, and execute project extensions.",
        [(option["label"], option) for option in options])
    if selected is None:
        return False
    if selected["updates"]:
        store.set_many(selected["updates"])
    return selected["trusted"]
