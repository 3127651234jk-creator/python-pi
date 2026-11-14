"""Transactional submission queues and the post-tools/final placement boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from ..ai.json_io import messages_from_json
from ..ai.types import UserMessage
from .documents import define_doc
from .entries import USER_ENTRY
from .transaction import DocDraft, Transaction

InboxDoc = define_doc("pi.inbox", 1, "conversation", lambda seed: {"items": []})
InboxDoc.history, InboxDoc.fork = "latest", "initial"


@dataclass
class Boundary:
    conversation_id: int
    inbox: DocDraft
    head: Optional[int]
    steering_mode: str = "all"
    follow_up_mode: str = "all"


async def prepare_boundary(tx: Transaction, conversation_id: int,
                           steering_mode: str = "all", follow_up_mode: str = "all") -> Boundary:
    if steering_mode not in ("all", "one-at-a-time") or follow_up_mode not in ("all", "one-at-a-time"):
        raise ValueError("queue mode must be all or one-at-a-time")
    marker = await tx.latest_head_marker(conversation_id)
    return Boundary(conversation_id, await tx.doc(InboxDoc, conversation_id),
                    marker.head if marker else None, steering_mode, follow_up_mode)


async def apply_boundary(tx: Transaction, boundary: Boundary, at: str, now: Optional[int] = None) -> dict:
    if at not in ("postTools", "final"):
        raise ValueError("boundary must be postTools or final")
    items = boundary.inbox.value["items"]
    reset = any(item["mode"] == "write" and item["entry"].get("head") == "self" for item in items)
    writes = [i for i, item in enumerate(items) if item["mode"] == "write"]
    users = []
    for mode, policy in (("steer", boundary.steering_mode), ("followUp", boundary.follow_up_mode)):
        if mode == "followUp" and at != "final" and not reset:
            continue
        indexes = [i for i, item in enumerate(items) if item["mode"] == mode]
        users.extend(indexes if policy == "all" else indexes[:1])
    placed = []
    for index in writes:
        item = items[index]
        draft = dict(item["entry"])
        head = draft.get("head")
        if isinstance(head, int) and boundary.head is not None and head < boundary.head:
            await tx.settle_submission(item["id"], {"status": "unanswered", "reason": "stale"})
            continue
        if draft.get("model"):
            draft["model"] = messages_from_json(draft["model"])
        entry = await tx.append_entry(boundary.conversation_id, draft)
        if head is not None:
            boundary.head = entry.id if head == "self" else head
        await tx.place_submission(item["id"], entry.id)
    for index in sorted(users):
        item = items[index]
        entry = await tx.append_entry(boundary.conversation_id, {
            "kind": USER_ENTRY.kind,
            "model": [UserMessage(content=item["content"], timestamp=now if now is not None else int(time.time() * 1000))]})
        await tx.place_submission(item["id"], entry.id)
        placed.append({"id": item["id"], "entry": entry.id})
    for index in sorted(writes + users, reverse=True):
        del items[index]
    if writes or users:
        boundary.inbox.dirty = True
    return {"users": placed, "reset": reset}


async def withdraw_queued_inputs(tx: Transaction, conversation_id: int) -> None:
    inbox = await tx.doc(InboxDoc, conversation_id)
    for item in list(inbox.value["items"]):
        if item["mode"] != "write":
            await tx.settle_submission(item["id"], {"status": "unanswered", "reason": "aborted"})
            inbox.value["items"].remove(item)
            inbox.dirty = True
