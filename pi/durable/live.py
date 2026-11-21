"""Built-in live conversation state: the generated partial and running tool slots.

Python port of ``packages/durable/src/harness/live.ts``. Documents store full
base snapshots, so this module only defines the shape and the small helpers
that generation and tool tasks share. A committed partial is presentation
state, not a transcript entry: settle clears it in the same transaction that
appends the final answer, and abort converts it into an aborted assistant
entry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .documents import define_doc

LIVE_KIND = "pi.live"

LiveDoc = define_doc(LIVE_KIND, 1, "conversation", initial=lambda seed: {})
LiveDoc.history, LiveDoc.fork = "latest", "initial"


def generation_state(live: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    generation = live.get("generation")
    return generation if isinstance(generation, dict) else None


def tool_slot(live: Dict[str, Any], task_id: int) -> Optional[Dict[str, Any]]:
    for slot in live.get("tools") or []:
        if isinstance(slot, dict) and slot.get("task_id") == task_id:
            return slot
    return None


def ensure_tool_slot(live: Dict[str, Any], task_id: int, call_id: str, name: str) -> Dict[str, Any]:
    slots = live.setdefault("tools", [])
    slot = tool_slot(live, task_id)
    if slot is None:
        slot = {"task_id": task_id, "call_id": call_id, "name": name, "status": "pending"}
        slots.append(slot)
    return slot


def clear_progress(slot: Dict[str, Any]) -> None:
    for key in ("output", "dropped_bytes", "dropped_lines", "details", "diagnostics"):
        slot.pop(key, None)


def finish_slot(slot: Dict[str, Any], entry: Optional[int]) -> None:
    slot["status"] = "done"
    if entry is not None:
        slot["entry"] = entry
    clear_progress(slot)


def clear_generation(live: Dict[str, Any]) -> None:
    live.pop("generation", None)


def clear_tools(live: Dict[str, Any]) -> None:
    """Drop the current tool round's slots (TS ``delete live.tools``)."""

    live.pop("tools", None)


def end_run(live: Dict[str, Any]) -> None:
    """Clear the presentation owned by an ending run: the partial and tool slots.

    Port of the TS ``endRun`` cleanup (without ``run`` control, which this port
    does not model). Keeping the slots after their run ends would make
    ``pi.live`` accumulate every finished tool round forever.
    """

    clear_generation(live)
    clear_tools(live)


def tool_slots(live: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [slot for slot in live.get("tools") or [] if isinstance(slot, dict)]
