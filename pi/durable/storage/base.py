"""Storage backends.

Python port of ``packages/durable/src/storage``. The TS package implements
``Storage`` three times (memory, jsonl, sqlite) with identical observable
semantics enforced by a conformance suite; this port implements the semantics
once in ``BaseStorage`` (in-memory indexes) and derives:

- ``MemoryStorage`` — reference in-memory backend
- ``JsonlStorage`` — append-only JSONL commit log, replayed on open
- ``SqliteStorage`` — sqlite tables via the standard library

Porting deviation: document content is stored as full base snapshots (no delta
batches); every ``document.change`` writes a new base revision.
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Tuple

from ..errors import StorageRejected
from ..types import (
    DocumentRecord,
    Page,
    StoredDocument,
)


def _scope_key(scope: Dict[str, Any]) -> str:
    kind = scope["kind"]
    if kind == "session":
        return '["session"]'
    if kind == "conversation":
        return f'["conversation",{scope["conversation_id"]}]'
    return f'["task",{scope["task_id"]}]'


def _address_key(address: Dict[str, Any]) -> str:
    key = ["family", address["key"]] if address.get("key") is not None else ["singleton"]
    return f"{address['kind']}|{_scope_key(address['scope'])}|{key}"


def _record_field(record, name: str, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_address_key(record) -> str:
    return _address_key({"kind": _record_field(record, "kind"), "scope": _record_field(record, "scope"), "key": _record_field(record, "key")})


def _is_alive_at(record: DocumentRecord, at) -> bool:
    if at == "current":
        return record.retired_at is None
    return record.created_at <= at and (record.retired_at is None or at < record.retired_at)


def _is_current_only(record: DocumentRecord) -> bool:
    return record.scope.get("kind") != "conversation" or record.history == "latest"


class BaseStorage:
    """In-memory index core shared by every backend (port of MemoryStorage state)."""

    def __init__(self) -> None:
        self.record_types: Dict[int, str] = {}
        self.conversations: Dict[int, Any] = {}
        self.conversation_ids: List[int] = []
        self.conversation_ids_by_owner_conversation: Dict[int, List[int]] = {}
        self.conversation_ids_by_owner_task: Dict[int, List[int]] = {}
        self.entries: Dict[int, Any] = {}
        self.entry_ids: Dict[int, List[int]] = {}
        self.head_entry_ids: Dict[int, List[int]] = {}
        self.entry_commit_seqs: Dict[int, int] = {}
        self.tasks: Dict[int, Any] = {}
        self.task_ids: List[int] = []
        self.task_ids_by_status: Dict[str, List[int]] = {
            s: [] for s in ("pending", "running", "waiting", "completing", "terminal")
        }
        self.submissions: Dict[int, Any] = {}
        self.submission_ids: List[int] = []
        self.submission_ids_by_status: Dict[str, List[int]] = {s: [] for s in ("queued", "placed", "done", "unanswered")}
        self.submission_ids_by_request: Dict[Tuple[int, str], int] = {}
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.document_addresses: Dict[str, Dict[str, Any]] = {}
        self.document_ids_by_scope: Dict[str, List[int]] = {}
        self.next_id = 2
        self.next_seq = 1
        self.closed = False

    # -- id helpers ---------------------------------------------------------

    @staticmethod
    def _lower_bound(ids: List[int], target: int) -> int:
        return bisect.bisect_left(ids, target)

    @staticmethod
    def _upper_bound(ids: List[int], target: int) -> int:
        return bisect.bisect_right(ids, target)

    @classmethod
    def _insert_sorted(cls, ids: List[int], value: int) -> None:
        if not ids or ids[-1] < value:
            ids.append(value)
        else:
            ids.insert(cls._lower_bound(ids, value), value)

    @classmethod
    def _remove_sorted(cls, ids: List[int], value: int) -> None:
        index = cls._lower_bound(ids, value)
        if index < len(ids) and ids[index] == value:
            ids.pop(index)

    @classmethod
    def _insert_map_id(cls, index: Dict[Any, List[int]], key: Any, value: int) -> None:
        ids = index.setdefault(key, [])
        cls._insert_sorted(ids, value)

    @staticmethod
    def _cursor_after(cursor: Optional[Dict[str, Any]]) -> Optional[int]:
        if cursor is None:
            return None
        after = cursor.get("after")
        if after is None:
            return None
        if not isinstance(after, int) or isinstance(after, bool):
            raise StorageRejected("Invalid storage cursor")
        return after

    @classmethod
    def _page(cls, values: List[Any], limit: int) -> Page:
        items = values[:limit]
        if len(values) <= limit:
            return Page(items=list(items))
        return Page(items=list(items), next={"after": items[-1].id})

    def assert_open(self) -> None:
        if self.closed:
            raise StorageRejected("Storage is closed")

    # -- commit -------------------------------------------------------------

    def apply_commit(self, writes: List[Dict[str, Any]], seq: int) -> int:
        """Validate and apply one commit; returns the sequence."""
        writes = [self.resolve_document_copy(w) if w["type"] == "document.copy" else w for w in writes]
        self.check_global_ids(writes)
        document_actions = self.prepare_document_actions(writes)
        self.check_document_actions(document_actions)
        self.apply_writes(writes, document_actions, seq)
        self.next_seq = seq + 1
        return seq

    def check_global_ids(self, writes: List[Dict[str, Any]]) -> None:
        claimed: Dict[int, str] = {}
        for write in writes:
            wtype = write["type"]
            if wtype in ("document.change", "document.retire"):
                continue
            is_document = wtype in ("document.create", "document.copy")
            table = "document" if is_document else wtype
            rid = _record_field(write["record"], "id") if is_document else write["value"].id
            existing = self.record_types.get(rid)
            earlier = claimed.get(rid)
            if table in ("conversation", "entry", "document"):
                if existing is not None:
                    raise StorageRejected(f"ID {rid} already belongs to {existing}")
                if earlier is not None:
                    raise StorageRejected(f"ID {rid} is written more than once")
            else:
                if existing is not None and existing != table:
                    raise StorageRejected(f"ID {rid} already belongs to {existing}")
                if earlier is not None and earlier != table:
                    raise StorageRejected(f"ID {rid} is written as two record types")
            claimed[rid] = table

    def prepare_document_actions(self, writes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        actions: Dict[int, Dict[str, Any]] = {}
        for write in writes:
            wtype = write["type"]
            if wtype not in ("document.create", "document.change", "document.retire"):
                continue
            rid = _record_field(write["record"], "id") if wtype == "document.create" else write["id"]
            action = actions.setdefault(rid, {"retire": False})
            if wtype == "document.create":
                if action.get("create") is not None or action.get("content") is not None:
                    raise StorageRejected(f"Document {rid} has more than one content command")
                action["create"] = write["record"]
                action["content"] = write["content"]
            elif wtype == "document.change":
                if action.get("content") is not None:
                    raise StorageRejected(f"Document {rid} has more than one content command")
                action["content"] = write["content"]
            else:
                if action["retire"]:
                    raise StorageRejected(f"Document {rid} is retired more than once")
                action["retire"] = True
        return actions

    def check_document_actions(self, actions: Dict[int, Dict[str, Any]]) -> None:
        live_counts: Dict[str, int] = {}
        for rid, action in actions.items():
            existing = self.documents.get(rid)
            if action.get("create") is None and existing is None:
                raise StorageRejected(f"Unknown document: {rid}")
            if action.get("create") is not None and existing is not None:
                raise StorageRejected(f"Document {rid} already exists")
            if existing is not None and existing["record"].retired_at is not None:
                raise StorageRejected(f"Document {rid} is retired")
            revisions = existing["revisions"] if existing else []
            previous = revisions[-1] if revisions else None
            content = action.get("content")
            if content is not None and content["kind"] == "delta":
                # Delta content is not produced by this port, but keep the guard.
                if previous is None:
                    raise StorageRejected(f"Document {rid} delta has no base")
                if previous["content"]["version"] != content["version"]:
                    raise StorageRejected(f"Document {rid} version transition requires a base")

            created = action.get("create")
            key = _record_address_key(created) if created is not None else _record_address_key(existing["record"])
            address = self.document_addresses.get(key)
            current_id = address.get("current_id") if address else None
            live = live_counts.get(key)
            if live is None:
                live = 0 if current_id is None else 1
            if action["retire"] and current_id == rid:
                live -= 1
            if created is not None and not action["retire"]:
                live += 1
            live_counts[key] = live

        for live in live_counts.values():
            if live > 1:
                raise StorageRejected("Document address already has a current incarnation")

    def apply_writes(self, writes: List[Dict[str, Any]], document_actions: Dict[int, Dict[str, Any]], seq: int) -> None:
        import copy as _copy

        for write in writes:
            wtype = write["type"]
            if wtype == "conversation":
                value = _copy.deepcopy(write["value"])
                self.record_types[value.id] = "conversation"
                self.conversations[value.id] = value
                self._insert_sorted(self.conversation_ids, value.id)
                if value.owner is not None:
                    self._insert_map_id(self.conversation_ids_by_owner_conversation, value.owner["conversation_id"], value.id)
                    self._insert_map_id(self.conversation_ids_by_owner_task, value.owner["task_id"], value.id)
                self.next_id = max(self.next_id, value.id + 1)
            elif wtype == "entry":
                value = _copy.deepcopy(write["value"])
                self.record_types[value.id] = "entry"
                self.entries[value.id] = value
                self.entry_commit_seqs[value.id] = seq
                self._insert_map_id(self.entry_ids, value.conversation_id, value.id)
                if value.head is not None:
                    self._insert_map_id(self.head_entry_ids, value.conversation_id, value.id)
                self.next_id = max(self.next_id, value.id + 1)
            elif wtype == "task":
                value = _copy.deepcopy(write["value"])
                self.record_types[value.id] = "task"
                previous = self.tasks.get(value.id)
                if previous is None:
                    self._insert_sorted(self.task_ids, value.id)
                    self._insert_sorted(self.task_ids_by_status[value.state.status], value.id)
                elif previous.state.status != value.state.status:
                    self._remove_sorted(self.task_ids_by_status[previous.state.status], value.id)
                    self._insert_sorted(self.task_ids_by_status[value.state.status], value.id)
                self.tasks[value.id] = value
                self.next_id = max(self.next_id, value.id + 1)
            elif wtype == "submission":
                value = _copy.deepcopy(write["value"])
                self.record_types[value.id] = "submission"
                previous = self.submissions.get(value.id)
                if previous is None:
                    self._insert_sorted(self.submission_ids, value.id)
                    self._insert_sorted(self.submission_ids_by_status[value.status], value.id)
                elif previous.status != value.status:
                    self._remove_sorted(self.submission_ids_by_status[previous.status], value.id)
                    self._insert_sorted(self.submission_ids_by_status[value.status], value.id)
                if previous is not None and previous.request_id is not None:
                    request_key = (previous.conversation_id, previous.request_id)
                    if self.submission_ids_by_request.get(request_key) == value.id:
                        del self.submission_ids_by_request[request_key]
                self.submissions[value.id] = value
                if value.request_id is not None:
                    self.submission_ids_by_request[(value.conversation_id, value.request_id)] = value.id
                self.next_id = max(self.next_id, value.id + 1)
            elif wtype == "document.copy":
                raise StorageRejected("Prepared document copy was not resolved")
            # document.create/change/retire are handled by document actions.

        self.apply_document_actions(document_actions, seq)

    def apply_document_actions(self, actions: Dict[int, Dict[str, Any]], seq: int) -> None:
        for rid, action in actions.items():
            stored = self.documents.get(rid)
            created = action.get("create")
            if created is not None:
                record = DocumentRecord(
                    id=_record_field(created, "id"),
                    kind=_record_field(created, "kind"),
                    scope=_record_field(created, "scope"),
                    created_at=seq,
                    key=_record_field(created, "key"),
                    retired_at=seq if action["retire"] else None,
                    history=_record_field(created, "history"),
                    fork=_record_field(created, "fork"),
                )
                content = action["content"]
                stored = {
                    "record": record,
                    "revisions": [
                        {"content": {"kind": "base", "version": content["version"], "value": content["value"]}, "seq": seq}
                    ],
                }
                self.record_types[rid] = "document"
                self.documents[rid] = stored

                key = _record_address_key(record)
                address = self.document_addresses.setdefault(key, {"ids": [], "current_id": None})
                self._insert_sorted(address["ids"], rid)
                scope_ids = self.document_ids_by_scope.setdefault(_scope_key(record.scope), [])
                self._insert_sorted(scope_ids, rid)
                self.next_id = max(self.next_id, rid + 1)
            else:
                content = action.get("content")
                if content is not None:
                    revision = {"content": content, "seq": seq}
                    if content["kind"] == "base" and _is_current_only(stored["record"]):
                        stored["revisions"] = [revision]
                    else:
                        stored["revisions"].append(revision)

            if action["retire"] and created is None:
                stored["record"].retired_at = seq
            if action["retire"] and _is_current_only(stored["record"]):
                stored["revisions"] = []
            if created is not None or action["retire"]:
                address = self.document_addresses[_record_address_key(stored["record"])]
                if action["retire"] and address.get("current_id") == rid:
                    address["current_id"] = None
                if created is not None and not action["retire"]:
                    address["current_id"] = rid

    def resolve_document_copy(self, write: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a ``document.copy`` into a ``document.create`` at apply time."""
        source: Dict[str, Any] = write["source"]
        stored = self.materialize_document(source["id"], source["at"])
        if stored is None:
            raise StorageRejected(f"Fork source document {source['id']} cannot be read")
        record = write["record"]
        src_record = stored.record
        if (
            src_record.scope.get("kind") != "conversation"
            or record["scope"].get("kind") != "conversation"
            or src_record.kind != record["kind"]
            or src_record.key != record.get("key")
            or src_record.history != record.get("history")
            or src_record.fork != record.get("fork")
        ):
            raise StorageRejected(f"Fork source document {source['id']} does not match the copied record")
        return {
            "type": "document.create",
            "record": record,
            "content": {"kind": "base", "version": stored.version, "value": stored.value},
        }

    # -- reads ---------------------------------------------------------------

    def get_conversation(self, cid: int) -> Optional[Any]:
        return self.conversations.get(cid)

    def scan_conversations_sync(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]]) -> Page:
        if query.get("owner_task_id") is not None:
            ids = self.conversation_ids_by_owner_task.get(query["owner_task_id"], [])
        elif query.get("owner_conversation_id") is not None:
            ids = self.conversation_ids_by_owner_conversation.get(query["owner_conversation_id"], [])
        else:
            ids = self.conversation_ids
        after = self._cursor_after(cursor)
        start = 0 if after is None else self._upper_bound(ids, after)
        values: List[Any] = []
        for rid in ids[start:]:
            if len(values) > limit:
                break
            value = self.conversations[rid]
            if query.get("owner_conversation_id") is not None and (
                value.owner is None or value.owner.get("conversation_id") != query["owner_conversation_id"]
            ):
                continue
            values.append(value)
        return self._page(values, limit)

    def visible_entries(self, cid: int, min_entry_id: int = 0, max_entry_id: int = 1 << 62):
        if cid not in self.conversations:
            raise StorageRejected(f"Unknown conversation: {cid}")
        current_id = cid
        upper_entry_id = max_entry_id
        while True:
            ids = self.entry_ids.get(current_id, [])
            for rid in reversed(ids[: self._upper_bound(ids, upper_entry_id)]):
                if rid < min_entry_id:
                    break
                yield self.entries[rid]
            conversation = self.conversations[current_id]
            if conversation.parent is None:
                break
            upper_entry_id = min(upper_entry_id, conversation.parent["at"])
            if upper_entry_id < min_entry_id:
                break
            current_id = conversation.parent["conversation_id"]

    def get_entry_global(self, eid: int) -> Optional[Tuple[Any, int]]:
        entry = self.entries.get(eid)
        if entry is None:
            return None
        return entry, self.entry_commit_seqs[eid]

    def get_entry_visible(self, cid: int, eid: int) -> Optional[Tuple[Any, int]]:
        for entry in self.visible_entries(cid, eid, eid):
            return entry, self.entry_commit_seqs[entry.id]
        return None

    def find_latest_head_marker_sync(self, cid: int, at_or_before: Optional[int]) -> Optional[Any]:
        if cid not in self.conversations:
            raise StorageRejected(f"Unknown conversation: {cid}")
        current_id = cid
        upper_entry_id = at_or_before if at_or_before is not None else (1 << 62)
        while True:
            ids = self.head_entry_ids.get(current_id, [])
            index = self._upper_bound(ids, upper_entry_id) - 1
            if index >= 0:
                return self.entries[ids[index]]
            conversation = self.conversations[current_id]
            if conversation.parent is None:
                return None
            upper_entry_id = min(upper_entry_id, conversation.parent["at"])
            current_id = conversation.parent["conversation_id"]

    def scan_entries_sync(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]]) -> Page:
        after = self._cursor_after(cursor)
        max_entry_id = query.get("max_entry_id", 1 << 62)
        if after is not None:
            max_entry_id = min(max_entry_id, after - 1)
        visible: List[Any] = []
        for entry in self.visible_entries(query["conversation_id"], query.get("min_entry_id", 0), max_entry_id):
            visible.append(entry)
            if len(visible) > limit:
                break
        return self._page(visible, limit)

    def get_task(self, tid: int) -> Optional[Any]:
        return self.tasks.get(tid)

    def scan_tasks_sync(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]]) -> Page:
        ids = self.task_ids if query.get("status") is None else self.task_ids_by_status[query["status"]]
        after = self._cursor_after(cursor)
        start = 0 if after is None else self._upper_bound(ids, after)
        values: List[Any] = []
        for rid in ids[start:]:
            if len(values) > limit:
                break
            value = self.tasks[rid]
            if query.get("conversation_id") is not None and value.conversation_id != query["conversation_id"]:
                continue
            if query.get("kind") is not None and value.kind != query["kind"]:
                continue
            if query.get("abort_requested") is not None and value.abort_requested != query["abort_requested"]:
                continue
            if query.get("background") is not None and value.background != query["background"]:
                continue
            values.append(value)
        return self._page(values, limit)

    def get_submission(self, sid: int) -> Optional[Any]:
        return self.submissions.get(sid)

    def scan_submissions_sync(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]]) -> Page:
        ids = self.submission_ids if query.get("status") is None else self.submission_ids_by_status[query["status"]]
        after = self._cursor_after(cursor)
        start = 0 if after is None else self._upper_bound(ids, after)
        values: List[Any] = []
        for rid in ids[start:]:
            if len(values) > limit:
                break
            value = self.submissions[rid]
            if query.get("conversation_id") is not None and value.conversation_id != query["conversation_id"]:
                continue
            values.append(value)
        return self._page(values, limit)

    def get_submission_by_request(self, cid: int, request_id: str) -> Optional[Any]:
        rid = self.submission_ids_by_request.get((cid, request_id))
        return None if rid is None else self.submissions[rid]

    def find_document_sync(self, address: Dict[str, Any], at) -> Optional[DocumentRecord]:
        key = _address_key(address)
        index = self.document_addresses.get(key)
        if at == "current":
            current_id = index.get("current_id") if index else None
            return None if current_id is None else self.documents[current_id]["record"]
        for rid in index["ids"] if index else []:
            record = self.documents[rid]["record"]
            if _is_alive_at(record, at):
                return record
        return None

    def materialize_document(self, rid: int, at) -> Optional[StoredDocument]:
        stored = self.documents.get(rid)
        if stored is None:
            return None
        if at != "current" and _is_current_only(stored["record"]):
            raise StorageRejected(f"Document {rid} does not retain historical content")
        if not _is_alive_at(stored["record"], at):
            return None
        revisions = stored["revisions"] if at == "current" else [r for r in stored["revisions"] if r["seq"] <= at]
        base_index = len(revisions) - 1
        while base_index >= 0 and revisions[base_index]["content"]["kind"] != "base":
            base_index -= 1
        if base_index < 0:
            raise StorageRejected(f"Document {rid} is missing a required base")
        base = revisions[base_index]["content"]
        return StoredDocument(
            record=stored["record"],
            version=base["version"],
            value=base["value"],
            deltas_since_base=len(revisions) - base_index - 1,
        )

    def get_document(self, rid: int, at) -> Optional[StoredDocument]:
        return self.materialize_document(rid, at)

    def scan_documents_sync(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]]) -> Page:
        ids = self.document_ids_by_scope.get(_scope_key(query["scope"]), [])
        after = self._cursor_after(cursor)
        start = 0 if after is None else self._upper_bound(ids, after)
        values: List[Any] = []
        for rid in ids[start:]:
            if len(values) > limit:
                break
            record = self.documents[rid]["record"]
            if query.get("kind") is not None and record.kind != query["kind"]:
                continue
            if _is_alive_at(record, query["at"]):
                values.append(record)
        return self._page(values, limit)
