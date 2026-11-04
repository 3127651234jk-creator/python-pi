"""Durable data model.

Python port of the runtime parts of ``packages/durable/src/types.ts``: records,
the task state machine, storage writes, publications, and the storage protocol.
Records are dataclasses with JSON conversion for serialization-backed stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..ai.json_io import messages_from_json, to_json as ai_to_json

# Task state machine statuses (spec section 5.5)
TASK_STATUSES = ("pending", "running", "waiting", "completing", "terminal")
# Task outcome statuses
OUTCOME_STATUSES = ("completed", "failed", "aborted", "orphaned", "faulted")
# Submission statuses
SUBMISSION_STATUSES = ("queued", "placed", "done", "unanswered")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ConversationRecord:
    id: int
    parent: Optional[Dict[str, int]] = None  # {"conversation_id", "at"}
    owner: Optional[Dict[str, int]] = None  # {"conversation_id", "task_id"}


@dataclass
class ContextEdit:
    target: int
    action: str  # "omit" | "replace"
    messages: Optional[List[Any]] = None  # pi-ai messages when replacing


@dataclass
class EntryRecord:
    id: int
    conversation_id: int
    kind: str
    model: Optional[List[Any]] = None  # pi-ai messages for model context
    data: Any = None  # JSON payload for the application
    head: Optional[int] = None  # first active entry selected by this entry
    edits: Optional[List[ContextEdit]] = None
    by_task_id: Optional[int] = None


@dataclass
class SubmissionRecord:
    id: int
    conversation_id: int
    type: str  # "input" | "write"
    status: str  # "queued" | "placed" | "done" | "unanswered"
    request_id: Optional[str] = None
    entry: Optional[int] = None
    answer: Optional[int] = None
    reason: Optional[str] = None
    detail: Any = None


@dataclass
class TaskOutcome:
    status: str  # completed | failed | aborted | orphaned | faulted
    result: Any = None
    error: Optional[Dict[str, Any]] = None  # {"message", "detail"?}
    reason: Optional[str] = None


@dataclass
class TaskState:
    status: str  # pending | running | waiting | completing | terminal
    checkpoint: Any = None
    on: Optional[List[int]] = None  # waiting: task ids joined on
    policy: Optional[str] = None  # waiting: "failFast" | "allSettled"
    outcome: Optional[TaskOutcome] = None


@dataclass
class TaskRecord:
    id: int
    conversation_id: int
    kind: str
    version: int
    input: Any
    state: TaskState
    owner: Optional[int] = None
    background: bool = False
    abort_requested: bool = False
    memos: Optional[Dict[str, Any]] = None


@dataclass
class DocumentRecord:
    id: int
    kind: str
    scope: Dict[str, Any]  # {"kind": "session"} | {"kind": "conversation", "conversation_id"} | {"kind": "task", "task_id"}
    created_at: int = 0
    key: Optional[str] = None
    retired_at: Optional[int] = None
    history: Optional[str] = None  # "latest" | "rewindable" (conversation scope)
    fork: Optional[str] = None  # "asOf" | "current" | "initial"


@dataclass
class StoredDocument:
    record: DocumentRecord
    version: int
    value: Dict[str, Any]
    deltas_since_base: int = 0


@dataclass
class Page:
    items: List[Any]
    next: Optional[Dict[str, Any]] = None  # cursor {"after": id}


@dataclass
class CommitPublication:
    seq: int
    changes: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Storage writes (discriminated dicts, port of StorageWrite)
# ---------------------------------------------------------------------------


def write_conversation(value: ConversationRecord) -> Dict[str, Any]:
    return {"type": "conversation", "value": value}


def write_entry(value: EntryRecord) -> Dict[str, Any]:
    return {"type": "entry", "value": value}


def write_task(value: TaskRecord) -> Dict[str, Any]:
    return {"type": "task", "value": value}


def write_submission(value: SubmissionRecord) -> Dict[str, Any]:
    return {"type": "submission", "value": value}


def write_document_create(record: DocumentRecord, version: int, value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "document.create",
        "record": record,
        "content": {"kind": "base", "version": version, "value": value},
    }


def write_document_change(id: int, version: int, value: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "document.change", "id": id, "content": {"kind": "base", "version": version, "value": value}}


def write_document_retire(id: int) -> Dict[str, Any]:
    return {"type": "document.retire", "id": id}


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------


_RECORD_FIELDS = {
    ConversationRecord: ("id", "parent", "owner"),
    EntryRecord: ("id", "conversation_id", "kind", "model", "data", "head", "edits", "by_task_id"),
    SubmissionRecord: ("id", "conversation_id", "type", "status", "request_id", "entry", "answer", "reason", "detail"),
    TaskRecord: ("id", "conversation_id", "kind", "version", "input", "owner", "background", "abort_requested", "state", "memos"),
    DocumentRecord: ("id", "kind", "scope", "created_at", "key", "retired_at", "history", "fork"),
}

_NESTED_TASK_STATE = ("status", "checkpoint", "on", "policy", "outcome")


def record_to_dict(record: Any) -> Dict[str, Any]:
    """Serialize a record (or a plain write payload) to JSON-safe data."""
    if isinstance(record, dict):
        return record
    dataclass_type = type(record)
    if dataclass_type is TaskState:
        out: Dict[str, Any] = {"status": record.status}
        if record.checkpoint is not None:
            out["checkpoint"] = record.checkpoint
        if record.on is not None:
            out["on"] = list(record.on)
        if record.policy is not None:
            out["policy"] = record.policy
        if record.outcome is not None:
            out["outcome"] = record_to_dict(record.outcome)
        return out
    if dataclass_type is TaskOutcome:
        out = {"status": record.status}
        if record.result is not None:
            out["result"] = record.result
        if record.error is not None:
            out["error"] = dict(record.error)
        if record.reason is not None:
            out["reason"] = record.reason
        return out
    if dataclass_type is ContextEdit:
        out = {"target": record.target, "action": record.action}
        if record.messages is not None:
            out["messages"] = record.messages
        return out
    fields = _RECORD_FIELDS.get(dataclass_type)
    if fields is None:
        raise TypeError(f"Cannot serialize record of type {dataclass_type.__name__}")
    out = {}
    for name in fields:
        value = getattr(record, name)
        if value is None:
            continue
        if dataclass_type is EntryRecord and name == "model":
            out[name] = ai_to_json(value)
        elif isinstance(value, (ConversationRecord, EntryRecord, SubmissionRecord, TaskRecord, DocumentRecord, TaskState, TaskOutcome, ContextEdit)):
            out[name] = record_to_dict(value)
        else:
            out[name] = value
    return out


def _context_edit_from_json(data: Dict[str, Any]) -> ContextEdit:
    return ContextEdit(
        target=data["target"],
        action=data["action"],
        messages=messages_from_json(data["messages"]) if data.get("messages") is not None else None,
    )


def record_from_dict(data: Dict[str, Any]) -> Any:
    """Rebuild a record from persisted JSON; role keyed by shape."""
    if "kind" in data and "scope" in data and "conversation_id" not in data and "type" not in data:
        return DocumentRecord(
            id=data["id"],
            kind=data["kind"],
            scope=data["scope"],
            created_at=data.get("created_at", 0),
            key=data.get("key"),
            retired_at=data.get("retired_at"),
            history=data.get("history"),
            fork=data.get("fork"),
        )
    if "state" in data and "kind" in data:
        state_data = data["state"]
        outcome = state_data.get("outcome")
        state = TaskState(
            status=state_data["status"],
            checkpoint=state_data.get("checkpoint"),
            on=state_data.get("on"),
            policy=state_data.get("policy"),
            outcome=TaskOutcome(
                status=outcome["status"],
                result=outcome.get("result"),
                error=outcome.get("error"),
                reason=outcome.get("reason"),
            )
            if outcome
            else None,
        )
        return TaskRecord(
            id=data["id"],
            conversation_id=data["conversation_id"],
            kind=data["kind"],
            version=data["version"],
            input=data.get("input"),
            owner=data.get("owner"),
            background=data.get("background", False),
            abort_requested=data.get("abort_requested", False),
            state=state,
            memos=data.get("memos"),
        )
    if "tool_call_id" in data:
        raise ValueError("pi-ai messages are not durable records")
    if "edits" in data or ("kind" in data and "conversation_id" in data):
        model = data.get("model")
        return EntryRecord(
            id=data["id"],
            conversation_id=data["conversation_id"],
            kind=data["kind"],
            model=messages_from_json(model) if model is not None else None,
            data=data.get("data"),
            head=data.get("head"),
            edits=[_context_edit_from_json(e) for e in (data.get("edits") or [])] or None,
            by_task_id=data.get("by_task_id"),
        )
    if "type" in data and data["type"] in ("input", "write"):
        return SubmissionRecord(
            id=data["id"],
            conversation_id=data["conversation_id"],
            type=data["type"],
            status=data["status"],
            request_id=data.get("request_id"),
            entry=data.get("entry"),
            answer=data.get("answer"),
            reason=data.get("reason"),
            detail=data.get("detail"),
        )
    return ConversationRecord(id=data["id"], parent=data.get("parent"), owner=data.get("owner"))


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

PhaseHandler = Callable[[TaskRecord, Any, Any], Awaitable[None]]


@dataclass
class TaskDefinition:
    """Executable durable state machine definition (port of TaskDefinition)."""

    name: str
    version: int
    initial: Callable[[Any], Any]  # input -> checkpoint dict with "phase"
    phases: Dict[str, PhaseHandler]
    abort: PhaseHandler
    hooks: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError(f"Task definition {self.name} version must be positive")
        if not self.phases:
            raise ValueError(f"Task definition {self.name} must declare phases")


def define_task(
    name: str,
    version: int,
    initial: Callable[[Any], Any],
    phases: Dict[str, PhaseHandler],
    abort: Optional[PhaseHandler] = None,
    hooks: Optional[Dict[str, Any]] = None,
) -> TaskDefinition:
    return TaskDefinition(
        name=name,
        version=version,
        initial=initial,
        phases=phases,
        abort=abort or _default_abort,
        hooks=hooks,
    )


async def _default_abort(task: TaskRecord, runtime: Any, context: Any) -> None:
    await runtime.commit(lambda tx, current: TaskState(status="terminal", outcome=TaskOutcome(status="aborted")))
