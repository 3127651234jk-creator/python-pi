"""Session transactions.

Python port of the core of ``packages/durable/src/session/transaction.ts``:
write staging, read-after-write enforcement, draft settlement, and document
drafts. The TS scope object is reduced to ``TransactionScope`` here carrying
the conversation bound to the transaction and the task attributed to entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .documents import (
    DocDefinition,
    check_record_scope,
    check_record_version,
    initial_value,
    materialize_document,
    resolve_address,
)
from .errors import ReadAfterWrite, StorageRejected
from .types import (
    ContextEdit,
    ConversationRecord,
    DocumentRecord,
    EntryRecord,
    Page,
    SubmissionRecord,
    TaskDefinition,
    TaskRecord,
    TaskState,
    write_conversation,
    write_document_change,
    write_document_create,
    write_document_retire,
    write_entry,
    write_submission,
    write_task,
)


@dataclass
class TransactionScope:
    """Default conversation for created tasks; task attribution for entries."""

    conversation_id: Optional[int] = None
    task_id: Optional[int] = None


class DocDraft:
    """Mutable document value staged inside one transaction."""

    def __init__(self, definition: DocDefinition, record: Optional[DocumentRecord], value: Dict[str, Any]) -> None:
        self.definition = definition
        self.record = record
        self.value = value
        self.dirty = False
        self.retired = False


@dataclass
class _LoadedDocument:
    record: DocumentRecord
    stored_version: int
    value: Dict[str, Any]


class Transaction:
    """One Session commit callback's transaction surface."""

    def __init__(self, host: Any, scope: Optional[TransactionScope] = None) -> None:
        self._host = host
        self._scope = scope or TransactionScope()
        self._writes: List[Dict[str, Any]] = []
        self._wrote_table = False
        self._created_conversations: List[ConversationRecord] = []
        self._created_tasks: List[TaskRecord] = []
        self._doc_drafts: List[DocDraft] = []
        self._pending_submission_updates: Dict[int, SubmissionRecord] = {}
        self._closed = False

    # -- scoped reads (forbidden after the first table write) -----------------

    def _before_table_read(self) -> None:
        if self._wrote_table:
            raise ReadAfterWrite("Tx cannot read tables after the first table write")

    def _mark_table_write(self) -> None:
        self._wrote_table = True

    async def conversation(self, cid: int) -> Optional[ConversationRecord]:
        self._before_table_read()
        for record in self._created_conversations:
            if record.id == cid:
                return record
        return await self._host.storage.conversation(cid)

    async def entry(self, eid: int) -> Optional[EntryRecord]:
        self._before_table_read()
        for write in self._writes:
            if write["type"] == "entry" and write["value"].id == eid:
                return write["value"]
        stored = await self._host.storage.entry_global(eid)
        return None if stored is None else stored[0]

    async def task(self, tid: int) -> Optional[TaskRecord]:
        self._before_table_read()
        for record in self._created_tasks:
            if record.id == tid:
                return record
        for write in self._writes:
            if write["type"] == "task" and write["value"].id == tid:
                return write["value"]
        return await self._host.storage.task(tid)

    async def scan_entries(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self._before_table_read()
        return await self._host.storage.scan_entries(query, limit, cursor)

    async def scan_tasks(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self._before_table_read()
        return await self._host.storage.scan_tasks(query, limit, cursor)

    async def scan_submissions(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self._before_table_read()
        return await self._host.storage.scan_submissions(query, limit, cursor)

    async def scan_conversations(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self._before_table_read()
        return await self._host.storage.scan_conversations(query, limit, cursor)

    async def latest_head_marker(self, cid: int) -> Optional[EntryRecord]:
        self._before_table_read()
        return await self._host.storage.find_latest_head_marker(cid, None)

    async def submission_by_request(self, cid: int, request_id: str) -> Optional[SubmissionRecord]:
        self._before_table_read()
        for write in self._writes:
            if write["type"] == "submission":
                value = write["value"]
                if value.conversation_id == cid and value.request_id == request_id:
                    return value
        return await self._host.storage.submission_by_request(cid, request_id)

    # -- creation -------------------------------------------------------------

    async def create_conversation(self, ownership: dict) -> ConversationRecord:
        self._before_table_read()
        rid = await self._host.storage.mint_id()
        record = ConversationRecord(id=rid)
        if ownership.get("kind") == "task":
            task_id = ownership["task_id"]
            task = await self.task(task_id)
            if task is None:
                raise StorageRejected(f"Unknown owner task: {task_id}")
            record.owner = {"conversation_id": task.conversation_id, "task_id": task_id}
        self._created_conversations.append(record)
        self._writes.append(write_conversation(record))
        self._mark_table_write()
        await self._host.conversation_created(self, record)
        return record

    async def fork_conversation(
        self, parent_conversation_id: int, at: int, ownership: dict
    ) -> ConversationRecord:
        self._before_table_read()
        parent = await self.conversation(parent_conversation_id)
        if parent is None:
            raise StorageRejected(f"Unknown conversation: {parent_conversation_id}")
        stored = await self._host.storage.entry_visible(parent_conversation_id, at)
        if stored is None:
            raise StorageRejected(f"Entry {at} is not visible from conversation {parent_conversation_id}")
        record = await self.create_conversation(ownership)
        record.parent = {"conversation_id": parent_conversation_id, "at": at}
        return record

    async def append_entry(self, cid: int, draft: Dict[str, Any]) -> EntryRecord:
        # Trusts the Session to name a valid conversation: reading the table here
        # would be a read-after-write for the common create-then-append pattern.
        eid = await self._host.storage.mint_id()
        head = draft.get("head")
        entry = EntryRecord(
            id=eid,
            conversation_id=cid,
            kind=draft["kind"],
            model=draft.get("model"),
            data=draft.get("data"),
            head=eid if head == "self" else head,
            edits=draft.get("edits"),
            by_task_id=self._scope.task_id,
        )
        self._writes.append(write_entry(entry))
        self._mark_table_write()
        return entry

    async def create_task(self, definition: TaskDefinition, input: Any, options: Dict[str, Any]) -> TaskRecord:
        ownership = options["ownership"]
        if ownership.get("kind") == "task":
            owner_task = await self.task(ownership["task_id"])
            if owner_task is None:
                raise StorageRejected(f"Unknown owner task: {ownership['task_id']}")
            owner: Optional[int] = owner_task.id
            conversation_id = options.get("conversation_id") or owner_task.conversation_id
        else:
            owner = None
            conversation_id = options.get("conversation_id") or self._scope.conversation_id
            if conversation_id is None:
                raise StorageRejected("Conversation-owned tasks require a conversation")
        # Trusts the Session to name a valid conversation (no table read: the
        # create-conversation-then-create-task pattern must work in one commit).

        tid = await self._host.storage.mint_id()
        checkpoint = definition.initial(input)
        if not isinstance(checkpoint, dict) or "phase" not in checkpoint:
            raise ValueError(f"Task {definition.name} initial checkpoint must carry a phase")
        record = TaskRecord(
            id=tid,
            conversation_id=conversation_id,
            kind=definition.name,
            version=definition.version,
            input=input,
            owner=owner,
            background=bool(options.get("background", False)),
            abort_requested=False,
            state=TaskState(status="pending", checkpoint=checkpoint),
        )
        self._created_tasks.append(record)
        self._writes.append(write_task(record))
        self._mark_table_write()
        return record

    async def create_submission(self, create: Dict[str, Any]) -> SubmissionRecord:
        sid = await self._host.storage.mint_id()
        record = SubmissionRecord(
            id=sid,
            conversation_id=create["conversation_id"],
            type=create["type"],
            status=create.get("status", "queued"),
            request_id=create.get("request_id"),
            entry=create.get("entry"),
            answer=create.get("answer"),
            reason=create.get("reason"),
            detail=create.get("detail"),
        )
        self._writes.append(write_submission(record))
        self._mark_table_write()
        return record

    async def _latest_submission(self, sid: int) -> SubmissionRecord:
        """Resolve the submission against staged writes first, then storage.

        Internal settlement resolution is allowed after table writes (like the
        TS contract: settlement works after table writes).
        """
        for write in self._writes:
            if write["type"] == "submission" and write["value"].id == sid:
                return write["value"]
        stored = await self._host.storage.submission(sid)
        if stored is None:
            raise KeyError(f"Submission {sid} does not exist")
        return stored

    async def settle_submission(self, sid: int, settlement: Dict[str, Any]) -> None:
        """Settle a queued or placed submission.

        Resolved against staged writes first, then storage — internal settlement
        resolution works after table writes (TS contract).
        """
        record = self._staged_or_none(sid)
        if record is None:
            stored = await self._host.storage.submission(sid)
            if stored is None:
                raise KeyError(f"Submission {sid} does not exist")
            record = stored
        if record.status not in ("queued", "placed"):
            return
        if settlement["status"] == "done":
            if record.type != "input" or record.status != "placed":
                raise StorageRejected(f"Submission {sid} cannot be answered")
            record.status = "done"
            record.answer = settlement["answer"]
        else:
            record.status = "unanswered"
            record.reason = settlement["reason"]
            record.detail = settlement.get("detail")
        self._pending_submission_updates[sid] = record

    def _staged_or_none(self, sid: int) -> Optional[SubmissionRecord]:
        pending = self._pending_submission_updates.get(sid)
        if pending is not None:
            return pending
        for write in self._writes:
            if write["type"] == "submission" and write["value"].id == sid:
                return write["value"]
        return None

    async def place_submission(self, sid: int, entry: int) -> None:
        record = self._staged_or_none(sid)
        if record is None:
            stored = await self._host.storage.submission(sid)
            if stored is None:
                raise KeyError(f"Submission {sid} does not exist")
            record = stored
        if record.status != "queued":
            return
        record.status = "placed" if record.type == "input" else "done"
        record.entry = entry
        self._pending_submission_updates[sid] = record

    # -- documents -------------------------------------------------------------

    async def doc(self, definition: DocDefinition, *args: Any) -> DocDraft:
        self._before_table_read()
        address_id, address, next_index = resolve_address(definition, list(args))
        for draft in self._doc_drafts:
            if draft.definition is definition and draft.record is not None:
                pass
        loaded = await self._host.load_document(definition, address)
        if loaded is None:
            seed = args[next_index] if definition.family and next_index < len(args) else None
            if definition.family:
                seed = args[next_index] if next_index < len(args) else None
            draft = DocDraft(definition=definition, record=None, value=initial_value(definition, seed))
        else:
            check_record_scope(definition, loaded.record)
            check_record_version(definition, loaded.record, loaded.stored_version)
            value = materialize_document(definition, loaded)
            draft = DocDraft(definition=definition, record=loaded.record, value=value)
        self._doc_drafts.append(draft)
        return draft

    async def retire_doc(self, definition: DocDefinition, *args: Any) -> None:
        self._before_table_read()
        _address_id, address, _next = resolve_address(definition, list(args))
        record = await self._host.storage.find_document(address, "current")
        if record is not None:
            self._writes.append(write_document_retire(record.id))
            self._mark_table_write()

    # -- settlement -------------------------------------------------------------

    async def settle_success(self) -> List[Dict[str, Any]]:
        """Stage document writes for dirty drafts and return the commit batch."""
        if self._closed:
            raise RuntimeError("Transaction already settled")
        self._closed = True

        for draft in self._doc_drafts:
            if draft.retired:
                continue
            if draft.record is None:
                if not draft.dirty:
                    continue
                rid = await self._host.storage.mint_id()
                record = DocumentRecord(
                    id=rid,
                    kind=draft.definition.kind,
                    scope=self._scope_for_definition(draft.definition),
                    history=None,
                    fork=None,
                )
                self._writes.append(write_document_create(record, draft.definition.version, draft.value))
                self._mark_table_write()
            elif draft.dirty:
                self._writes.append(write_document_change(draft.record.id, draft.definition.version, draft.value))
                self._mark_table_write()

        # Emit updated submission records for place/settle mutations.
        for record in self._pending_submission_updates.values():
            self._writes.append(write_submission(record))

        return list(self._writes)

    def _scope_for_definition(self, definition: DocDefinition) -> Dict[str, Any]:
        kind = definition.scope
        if kind == "session":
            return {"kind": "session"}
        if kind == "conversation":
            if self._scope.conversation_id is None:
                raise StorageRejected("Conversation documents require a bound conversation")
            return {"kind": "conversation", "conversation_id": self._scope.conversation_id}
        if self._scope.task_id is None:
            raise StorageRejected("Task documents require a bound task")
        return {"kind": "task", "task_id": self._scope.task_id}

    async def settle_failure(self) -> None:
        self._closed = True

    def discard(self) -> None:
        self._closed = True
        self._writes.clear()

    def adopt(self, seq: int) -> List[Dict[str, Any]]:
        """Publication fragments for document writes after storage admitted them."""
        changes: List[Dict[str, Any]] = []
        for write in self._writes:
            wtype = write["type"]
            if wtype in ("document.create", "document.change"):
                record = write["record"] if wtype == "document.create" else self._record_of(write["id"])
                changes.append(
                    {
                        "type": "document",
                        "record": record,
                        "version": write["content"]["version"],
                        "value": write["content"]["value"],
                        "ops": [],
                    }
                )
            elif wtype == "document.retire":
                changes.append(
                    {"type": "document", "record": self._record_of(write["id"]), "version": None, "value": None, "ops": []}
                )
        return changes

    def _record_of(self, rid: int) -> DocumentRecord:
        for draft in self._doc_drafts:
            if draft.record is not None and draft.record.id == rid:
                return draft.record
        raise KeyError(f"Document {rid} is not part of this transaction")
