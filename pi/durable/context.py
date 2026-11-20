"""Conversation context reads: bounds capture, derivation, and incremental caching.

Python port of ``packages/durable/src/harness/context.ts``: the active
transcript and the model context derived from it, cut off at a visible entry
(``at``) when asked. ``read_context`` is the one-shot path;
``read_context_from`` reuses a previous range of the same conversation so a
read whose head marker did not move only scans and derives the entries
committed after it (the 1.1.0 context cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..ai.types import TextContent, ToolCall, ToolResultMessage
from .errors import StorageRejected
from .prompt import entry_messages, lead_with_system
from .types import ContextEdit, EntryRecord

SCAN_PAGE_SIZE = 256
# Assistant responses that never enter the model context (TS EXCLUDED_STOP_REASONS).
EXCLUDED_STOP_REASONS = frozenset({"aborted", "error", "deferred"})
MISSING_RESULT_TEXT = "Tool result unavailable: history ends before this call completed."


@dataclass(frozen=True)
class ContextBounds:
    """Head marker and newest visible entry that fix one committed context range."""

    head: Optional[EntryRecord]
    tail: int


@dataclass(frozen=True)
class ContextView:
    """Raw active transcript and derived model context."""

    head: Optional[EntryRecord]
    entries: List[EntryRecord]
    contributions: List[List[Any]]
    messages: List[Any]


@dataclass(frozen=True)
class ContextRange:
    """One context read: the entries it scanned plus its derived view.

    Entries at or below the tail never change, so a later read with the same
    head marker extends both.
    """

    bounds: ContextBounds
    entries: List[EntryRecord]
    view: ContextView
    # Targets of the edits in `entries`.
    edited: FrozenSet[int]
    # `view.messages` from the contributed messages before the last assistant
    # message; later entries cannot change these.
    settled: List[Any]
    # Contributed messages from the last assistant message on, before tool
    # result ordering.
    open: List[Any]


async def capture_context_bounds(storage, conversation_id: int,
                                 at: Optional[int] = None) -> Optional[ContextBounds]:
    """Capture the bounds of the current context, or of the context cut off at
    the visible entry ``at``, with two O(1) reads. Run on the Session line:
    entries at or below the tail are immutable, so ``derive_context`` can then
    scan them off the line."""

    if at is None:
        page = await storage.scan_entries({"conversation_id": conversation_id}, 1, None)
        if not page.items:
            return None
        tail = page.items[0].id
    else:
        if await storage.entry_visible(conversation_id, at) is None:
            raise StorageRejected(f"Entry {at} is not visible from conversation {conversation_id}")
        tail = at
    return ContextBounds(head=await storage.find_latest_head_marker(conversation_id, tail), tail=tail)


async def scan_range(storage, conversation_id: int, range_query: Dict[str, Any]) -> List[EntryRecord]:
    """Visible entries within ``range_query``, oldest first."""

    entries: List[EntryRecord] = []
    cursor: Optional[Dict[str, Any]] = None
    while True:
        page = await storage.scan_entries(
            {"conversation_id": conversation_id, **range_query}, SCAN_PAGE_SIZE, cursor)
        entries.extend(page.items)
        cursor = page.next
        if cursor is None:
            break
    entries.reverse()
    return entries


def range_bounds_query(bounds: ContextBounds) -> Dict[str, Any]:
    """The visible range of ``bounds``: from the head marker's head, or
    transcript start, through the tail."""

    head = bounds.head
    if head is None:
        return {"max_entry_id": bounds.tail}
    return {"min_entry_id": head.head, "max_entry_id": bounds.tail}


def select_active(head: Optional[EntryRecord], entries: List[EntryRecord]) -> List[EntryRecord]:
    """The head marker followed by the range's non-head entries, or the whole
    range without a marker."""

    if head is None:
        return list(entries)
    return [head] + [entry for entry in entries if entry.head is None]


def contribute(entry: EntryRecord, edit: Optional[ContextEdit]) -> List[Any]:
    """One active entry's model messages after its edit and excluded stop
    reasons, before tool result ordering."""

    contributed = entry_messages(entry, edit)
    return [
        message
        for message in contributed
        if not (getattr(message, "role", "") == "assistant"
                and getattr(message, "stop_reason", None) in EXCLUDED_STOP_REASONS)
    ]


def order_tool_results(messages: List[Any]) -> List[Any]:
    """Place each assistant's tool results directly after it in call order.
    Results are taken from the messages before the next assistant; a missing
    result is synthesized and unmatched results are dropped."""

    ordered: List[Any] = []
    for index, message in enumerate(messages):
        if getattr(message, "role", "") == "toolResult":
            continue
        ordered.append(message)
        if getattr(message, "role", "") != "assistant":
            continue
        calls = [block for block in message.content if isinstance(block, ToolCall)]
        if not calls:
            continue
        results: Dict[str, int] = {}
        for next_index in range(index + 1, len(messages)):
            candidate = messages[next_index]
            if getattr(candidate, "role", "") == "assistant":
                break
            if (getattr(candidate, "role", "") == "toolResult"
                    and candidate.tool_call_id not in results):
                results[candidate.tool_call_id] = next_index
        for call in calls:
            result_index = results.get(call.id)
            if result_index is None:
                ordered.append(_missing_result(call, message.timestamp))
            else:
                ordered.append(messages[result_index])
    return ordered


def _missing_result(call: ToolCall, timestamp: int) -> Any:
    return ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[TextContent(text=MISSING_RESULT_TEXT)],
        is_error=True,
        details={"reason": "missing_result"},
        timestamp=timestamp,
    )


def _settle(settled: List[Any], open: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Move the ordered messages before the last assistant message of ``open``
    to ``settled``. ``order_tool_results`` of a sequence equals the
    concatenation over its parts when each later part starts with an assistant
    message."""

    last = None
    for index in range(len(open) - 1, -1, -1):
        if getattr(open[index], "role", "") == "assistant":
            last = index
            break
    if last is None or last <= 0:
        return list(settled), list(open)
    return [*settled, *order_tool_results(open[:last])], list(open[last:])


def derive_range(bounds: ContextBounds, entries: List[EntryRecord]) -> ContextRange:
    """Derive the context view from the entries scanned within ``bounds``.

    H = newest visible head marker; the range runs from ``H.head`` (or
    transcript start) through the tail. Per target, the newest edit in the
    range wins. Context entries are H followed by the range's non-head entries.
    """

    edits: Dict[int, ContextEdit] = {}
    # Edits of every entry in the range count, including older head markers
    # that `select_active` drops.
    for entry in entries:
        for edit in entry.edits or []:
            edits[edit.target] = edit
    active = select_active(bounds.head, entries)
    contributions = [contribute(entry, edits.get(entry.id)) for entry in active]
    settled, open = _settle([], [m for chunk in contributions for m in chunk])
    messages = lead_with_system([*settled, *order_tool_results(open)])
    view = ContextView(head=bounds.head, entries=active, contributions=contributions, messages=messages)
    return ContextRange(bounds=bounds, entries=entries, view=view,
                        edited=frozenset(edits.keys()), settled=settled, open=open)


async def read_context(session, storage, conversation_id: int,
                       at: Optional[int] = None) -> ContextView:
    """Committed context of one conversation: bounds captured on the Session
    line, entries derived off it."""

    bounds = await session.read_on_line(
        lambda: capture_context_bounds(storage, conversation_id, at))
    if bounds is None:
        return ContextView(head=None, entries=[], contributions=[], messages=[])
    return derive_range(bounds, await scan_range(storage, conversation_id, range_bounds_query(bounds))).view


async def read_context_from(session, storage, conversation_id: int, at: Optional[int],
                            previous: Optional[ContextRange]
                            ) -> Tuple[ContextView, Optional[ContextRange]]:
    """``read_context`` that reuses ``previous``, an earlier range of the same
    conversation: with the same head marker, only entries after its tail are
    scanned. Returns the view and the range to pass to the next read."""

    bounds = await session.read_on_line(
        lambda: capture_context_bounds(storage, conversation_id, at))
    if bounds is None:
        return ContextView(head=None, entries=[], contributions=[], messages=[]), None
    if previous is None or (previous.bounds.head is not None) != (bounds.head is not None) or (
            previous.bounds.head is not None and bounds.head is not None
            and previous.bounds.head.id != bounds.head.id):
        context_range = derive_range(bounds, await scan_range(storage, conversation_id, range_bounds_query(bounds)))
    elif bounds.tail == previous.bounds.tail:
        context_range = previous
    elif bounds.tail < previous.bounds.tail:
        context_range = derive_range(
            bounds, [entry for entry in previous.entries if entry.id <= bounds.tail])
    else:
        added = await scan_range(storage, conversation_id,
                                 {"min_entry_id": previous.bounds.tail + 1, "max_entry_id": bounds.tail})
        context_range = _extend_range(previous, bounds, added)
    view = context_range.view
    return (
        ContextView(head=view.head, entries=list(view.entries),
                    contributions=list(view.contributions), messages=list(view.messages)),
        context_range,
    )


def _extend_range(previous: ContextRange, bounds: ContextBounds,
                  added: List[EntryRecord]) -> ContextRange:
    """``previous`` extended by the entries ``added`` after its tail under the
    same head marker: only their contributions and the open messages are
    derived. An added edit can change an earlier entry, so it derives the
    whole range again."""

    entries = [*previous.entries, *added]
    if any(entry.edits or entry.head is not None or entry.id in previous.edited for entry in added):
        return derive_range(bounds, entries)
    contributions = [contribute(entry, None) for entry in added]
    settled, open = _settle(list(previous.settled), [*previous.open, *(m for c in contributions for m in c)])
    view = ContextView(
        head=bounds.head,
        entries=[*previous.view.entries, *added],
        contributions=[*previous.view.contributions, *contributions],
        messages=lead_with_system([*settled, *order_tool_results(open)]),
    )
    return ContextRange(bounds=bounds, entries=entries, view=view,
                        edited=previous.edited, settled=settled, open=open)
