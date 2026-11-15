"""Positional system entries for durable conversations.

Python port of the system-entry planning in ``packages/durable/src/harness/prompt.ts``
plus the ``leadWithSystem`` fix for #10542 (``harness/context.ts``). Generation
prepare renders the desired sections and tools, diffs them against the state
replayed from committed ``pi.system`` entries, and appends an entry only when
something changed.

A ``pi.system`` entry carries ``model = [SystemMessage(content="", sections,
tools_added, tools_removed)]``: the prompt text lives in the sections map,
tool changes ride inline. Replaying the entries in order (sections set in
place, ``None`` deletes, tools removed before added) reconstructs the current
prompt state, so after a compaction or reset a single full baseline entry
re-establishes it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..ai.types import Message, SystemMessage, Tool, ToolReference
from ..ai.utils import ToolStateChanges, _json_equal, _replay_system
from .entries import SYSTEM_ENTRY
from .types import ContextEdit, EntryRecord

INSTRUCTIONS_KEY = "instructions"

# Entry kinds whose model contributes to the model context (generation prepare).
_CONTEXT_ENTRY_KINDS = ("pi.user", "pi.assistant", "pi.system", "pi.tool-result", "pi.compaction")


def lead_with_system(messages: List[Any]) -> List[Any]:
    """Move a system message that only user messages precede to the front.

    A run's input is committed before generation renders the system prompt, so
    the transcript (or the range after a compaction or reset) can start with
    user messages followed by the baseline system message. Providers treat only
    a leading system message as the initial prompt and tool set; without the
    move, a later tool change rewrites the tool list and invalidates the whole
    prompt cache (#10542). Everything else keeps its relative order.
    """

    index = -1
    for i, message in enumerate(messages):
        if getattr(message, "role", "") != "user":
            index = i
            break
    if index <= 0 or getattr(messages[index], "role", "") != "system":
        return messages
    return [messages[index], *messages[:index], *messages[index + 1 :]]


def context_edits(entries: List[EntryRecord]) -> Dict[int, ContextEdit]:
    """Edits carried by the active-range entries, keyed by target (last wins)."""

    edits: Dict[int, ContextEdit] = {}
    for entry in entries:
        for edit in entry.edits or []:
            edits[edit.target] = edit
    return edits


def entry_messages(entry: EntryRecord, edit: Optional[ContextEdit]) -> List[Any]:
    """Model messages an entry contributes, after its edit (omit contributes none)."""
    if edit is not None and edit.action == "omit":
        return []
    if edit is not None and edit.action == "replace" and edit.messages is not None:
        return list(edit.messages)
    return list(entry.model or [])


def _declarations_equal(left: Tool, right: Tool) -> bool:
    """Whether two tools declare the same interface (agent-loop comparison)."""
    return _json_equal(
        [left.description, left.parameters, getattr(left, "hosted", None)],
        [right.description, right.parameters, getattr(right, "hosted", None)],
    )


def plan_tools(offered: List[Tool], desired: List[Tool]) -> ToolStateChanges:
    """Tool changes from ``offered`` to ``desired`` (TS ``planTools``).

    A changed declaration is removed and re-added. Replay keeps retained tools
    in place and appends additions; when that would not yield the desired
    order, every offered tool is removed and every desired tool re-added in
    order.
    """

    wanted = {tool.name: tool for tool in desired}
    kept = [tool for tool in offered
            if tool.name in wanted and _declarations_equal(tool, wanted[tool.name])]
    kept_names = {tool.name for tool in kept}
    added = [tool for tool in desired if tool.name not in kept_names]
    replayed = [*kept, *added]
    if len(replayed) != len(desired) or any(
            tool.name != expected.name for tool, expected in zip(replayed, desired)):
        return ToolStateChanges(tools_added=list(desired),
                                tools_removed=[ToolReference(name=tool.name) for tool in offered])
    return ToolStateChanges(
        tools_added=added,
        tools_removed=[ToolReference(name=tool.name) for tool in offered if tool.name not in kept_names],
    )


def _head_index(entries: List[EntryRecord]) -> Optional[int]:
    """Position of the latest head marker (pi.compaction / pi.reset) in the range."""

    index = None
    for i, entry in enumerate(entries):
        if entry.head is not None:
            index = i
    return index


def _system_message(content: str, sections: Optional[Dict[str, Optional[str]]],
                    tools_added: Optional[List[Tool]],
                    tools_removed: Optional[List[ToolReference]]) -> SystemMessage:
    return SystemMessage(
        content=content,
        timestamp=int(time.time() * 1000),
        sections=sections or None,
        tools_added=tools_added or None,
        tools_removed=tools_removed or None,
    )


def plan_system_entries(
    entries: List[EntryRecord],
    desired_sections: Dict[str, str],
    desired_tools: List[Tool],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Plan the ``pi.system`` entries for one generation prepare.

    Returns ``(drafts, messages)``: ``drafts`` is the possibly-empty list of
    ``append_entry`` drafts to commit, ``messages`` the model context in entry
    order with edits applied, the planned entries included, and a leading
    system message where #10542 would put it after a user prefix.
    """

    edits = context_edits(entries)
    head_idx = _head_index(entries)

    system_entries = [(i, e) for i, e in enumerate(entries) if e.kind == SYSTEM_ENTRY.kind]
    if head_idx is None:
        post_head = system_entries
    else:
        post_head = [(i, e) for i, e in system_entries if i > head_idx]

    # Replayed state from the system entries that survive the existing edits.
    replayed_messages: List[Message] = []
    for _, entry in system_entries:
        replayed_messages.extend(entry_messages(entry, edits.get(entry.id)))
    _, current_sections, current_tools, _ = _replay_system(replayed_messages)

    drafts: List[Dict[str, Any]] = []
    if not post_head:
        # No system entry after the latest head marker: write one full baseline.
        # Entries retained across a compaction carry the old prompt; omit every
        # one of them so the baseline is the only system state in the derived
        # view (as in TS planSystemEntries).
        omit_edits = [ContextEdit(target=entry.id, action="omit")
                      for _, entry in system_entries]
        drafts.append({
            "kind": SYSTEM_ENTRY.kind,
            "model": [_system_message("", dict(desired_sections), list(desired_tools), None)],
            "edits": omit_edits or None,
        })
    else:
        # Steady state: minimal section diff, tool changes on the last entry.
        additions = [k for k in desired_sections if k not in current_sections]
        kept_order = [k for k in current_sections if k in desired_sections]
        order_changed = kept_order + additions != list(desired_sections.keys())
        tool_changes = plan_tools(current_tools, desired_tools)
        section_changes: Dict[str, Optional[str]] = {}
        if not order_changed:
            for name, text in desired_sections.items():
                if name not in current_sections or current_sections[name] != text:
                    section_changes[name] = text
            for name in current_sections:
                if name not in desired_sections:
                    section_changes[name] = None

        changed = order_changed or bool(section_changes) or bool(tool_changes.tools_added) or bool(tool_changes.tools_removed)
        if changed:
            if order_changed:
                # Set-in-place cannot move a key: delete all, then re-add in
                # the desired order (two entries, as in TS planSystemEntries).
                if current_sections:
                    drafts.append({
                        "kind": SYSTEM_ENTRY.kind,
                        "model": [_system_message("", {name: None for name in current_sections}, None, None)],
                        "edits": None,
                    })
                drafts.append({
                    "kind": SYSTEM_ENTRY.kind,
                    "model": [_system_message("", dict(desired_sections),
                                              tool_changes.tools_added, tool_changes.tools_removed)],
                    "edits": None,
                })
            else:
                drafts.append({
                    "kind": SYSTEM_ENTRY.kind,
                    "model": [_system_message("", section_changes or None,
                                              tool_changes.tools_added, tool_changes.tools_removed)],
                    "edits": None,
                })

    # Assemble the derived view: entry-order messages with edits (existing and
    # planned) applied, then the #10542 ordering fix.
    for draft in drafts:
        for edit in draft.get("edits") or []:
            edits[edit.target] = edit
    messages: List[Any] = []
    for entry in entries:
        if entry.kind not in _CONTEXT_ENTRY_KINDS:
            continue
        messages.extend(entry_messages(entry, edits.get(entry.id)))
    for draft in drafts:
        messages.extend(draft["model"])
    return drafts, lead_with_system(messages)
