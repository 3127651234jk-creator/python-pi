"""Single-file HTML session export.

Python port of ``packages/coding-agent/src/core/export-html/index.ts``: the
TypeScript template assets (template.html/css/js plus the vendored marked and
highlight.js bundles) are embedded verbatim under ``assets/export/`` and the
session data rides along as base64 JSON, so the output is one self-contained
file rendered client-side exactly like the TS export.

The TS themes resolve ``okhsl()`` colors to hex at runtime; the two built-in
themes here carry the same colors pre-resolved to hex (via the shared
``packages/tui`` color math) because CSS does not understand okhsl().
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "export"

# The resolved ``colors`` and ``export`` sections of the TS default themes
# (packages/coding-agent/src/modes/interactive/theme/{dark,light}.json).
EXPORT_THEMES: Dict[str, Dict[str, Any]] = {
    "dark": {
        "accent": "#a798d7", "border": "#5fa8cc", "borderAccent": "#a08ed5",
        "success": "#68b78d", "error": "#ea7f81", "warning": "#cd9a22",
        "muted": "#9da5a9", "dim": "#7e888e", "text": "#dee0e1",
        "thinkingText": "#96a0a4", "selectedBg": "#213b49",
        "userMessageBg": "#213b49", "userMessageText": "#dee0e1",
        "customMessageBg": "#3a3055", "customMessageText": "#9da5a9",
        "customMessageLabel": "#a798d7",
        "toolPendingBg": "#34383a", "toolSuccessBg": "#254131", "toolErrorBg": "#5b282a",
        "toolOutput": "#9da5a9",
        "mdHeading": "#cd9a22", "mdLink": "#69add0", "mdCode": "#a798d7",
        "mdCodeBlockBorder": "#9da5a9", "mdQuote": "#9da5a9", "mdQuoteBorder": "#9da5a9",
        "mdHr": "#9da5a9", "mdListBullet": "#a798d7",
        "toolDiffAdded": "#68b78d", "toolDiffRemoved": "#ea7f81", "toolDiffContext": "#9da5a9",
        "syntaxComment": "#9da5a9", "syntaxKeyword": "#69add0", "syntaxFunction": "#cd9a22",
        "syntaxVariable": "#5db3ba", "syntaxString": "#de8d5a", "syntaxNumber": "#68b78d",
        "syntaxType": "#a798d7", "syntaxOperator": "#9da5a9", "syntaxPunctuation": "#9da5a9",
        "export": {"pageBg": "#21252c", "cardBg": "#282c34", "infoBg": "#4e2f1b"},
    },
    "light": {
        "accent": "#7459b4", "border": "#3d8eb3", "borderAccent": "#8a72cb",
        "success": "#337e58", "error": "#c8253d", "warning": "#8f6802",
        "muted": "#677176", "dim": "#879095", "text": "#3b3f41",
        "thinkingText": "#7c868c", "selectedBg": "#dfe7ec",
        "userMessageBg": "#dfe7ec", "userMessageText": "#3b3f41",
        "customMessageBg": "#e6e4ee", "customMessageText": "#677176",
        "customMessageLabel": "#7459b4",
        "toolPendingBg": "#e4e5e6", "toolSuccessBg": "#dee9e1", "toolErrorBg": "#eee2e1",
        "toolOutput": "#677176",
        "mdHeading": "#8f6802", "mdLink": "#2f7899", "mdCode": "#7459b4",
        "mdCodeBlockBorder": "#677176", "mdQuote": "#677176", "mdQuoteBorder": "#677176",
        "mdHr": "#677176", "mdListBullet": "#7459b4",
        "toolDiffAdded": "#337e58", "toolDiffRemoved": "#c8253d", "toolDiffContext": "#677176",
        "syntaxComment": "#677176", "syntaxKeyword": "#2f7899", "syntaxFunction": "#8f6802",
        "syntaxVariable": "#287a81", "syntaxString": "#a45417", "syntaxNumber": "#337e58",
        "syntaxType": "#7459b4", "syntaxOperator": "#677176", "syntaxPunctuation": "#677176",
        "export": {"pageBg": "#efeeee", "cardBg": "#f7f6f6", "infoBg": "#ede3dd"},
    },
}


def _read_asset(name: str) -> str:
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")


def _theme_colors(theme_name: Optional[str]) -> Dict[str, Any]:
    return EXPORT_THEMES["light" if theme_name == "light" else "dark"]


def generate_html(session_data: dict, theme_name: Optional[str] = None) -> str:
    """Fill the vendored TS template with one session's data and theme colors."""
    colors = dict(_theme_colors(theme_name))
    export_colors = colors.pop("export")
    theme_vars = "\n".join(f"--{key}: {value};" for key, value in colors.items())
    theme_vars += "\n--exportPageBg: {pageBg};\n--exportCardBg: {cardBg};\n--exportInfoBg: {infoBg};".format(**export_colors)

    css = (_read_asset("template.css")
           .replace("{{THEME_VARS}}", theme_vars)
           .replace("{{BODY_BG}}", export_colors["pageBg"])
           .replace("{{CONTAINER_BG}}", export_colors["cardBg"])
           .replace("{{INFO_BG}}", export_colors["infoBg"]))
    encoded = base64.b64encode(json.dumps(session_data, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return (_read_asset("template.html")
            .replace("{{CSS}}", css)
            .replace("{{JS}}", _read_asset("template.js"))
            .replace("{{SESSION_DATA}}", encoded)
            .replace("{{MARKED_JS}}", _read_asset(os.path.join("vendor", "marked.min.js")))
            .replace("{{HIGHLIGHT_JS}}", _read_asset(os.path.join("vendor", "highlight.min.js"))))


def generate_session_html(session, theme_name: Optional[str] = None) -> str:
    """Build the export document for a SessionManager's active branch."""
    from .rpc_wire import entry_to_ts
    from ..ai.types import SystemMessage

    entries = [entry_to_ts(entry) for entry in session.get_branch()]
    session_id = os.path.splitext(os.path.basename(session.session_file or ""))[0] or uuid.uuid4().hex
    header = {"type": "session", "version": 3, "id": session_id, "cwd": session.cwd}
    if session.session_file and os.path.exists(session.session_file):
        header["timestamp"] = int(os.path.getmtime(session.session_file) * 1000)
    system_prompt = ""
    for entry in session.get_branch():
        if isinstance(entry.message, SystemMessage) and entry.message.content:
            system_prompt = entry.message.content
            break
    return generate_html({
        "header": header,
        "entries": entries,
        "leafId": session.leaf_id,
        "systemPrompt": system_prompt or None,
    }, theme_name)
