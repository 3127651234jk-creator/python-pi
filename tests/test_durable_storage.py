"""Storage conformance tests (port of testing/storage-conformance.ts coverage).

The same assertions run against every backend; each backend instance is fresh
so tests are order-independent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.errors import StorageRejected
from pi.ai.types import UserMessage
from pi.durable.storage import JsonlStorage, MemoryStorage, SqliteStorage
from pi.durable.types import (
    ConversationRecord,
    ContextEdit,
    DocumentRecord,
    EntryRecord,
    SubmissionRecord,
    TaskRecord,
    TaskState,
    record_from_dict,
    record_to_dict,
    write_conversation,
    write_document_change,
    write_document_create,
    write_document_retire,
    write_entry,
    write_submission,
    write_task,
)

BACKENDS = ["memory", "jsonl", "sqlite"]


def make_storage(backend: str, tmp_path):
    if backend == "memory":
        return MemoryStorage()
    if backend == "jsonl":
        return JsonlStorage(str(tmp_path / f"{backend}.jsonl"))
    return SqliteStorage(str(tmp_path / f"{backend}.db"))


def conversation(id: int, **kwargs) -> ConversationRecord:
    return ConversationRecord(id=id, **kwargs)


def entry(id: int, conversation_id: int, **kwargs) -> EntryRecord:
    return EntryRecord(id=id, conversation_id=conversation_id, kind="pi.user", **kwargs)


def task_record(id: int, conversation_id: int, status: str = "pending") -> TaskRecord:
    return TaskRecord(
        id=id,
        conversation_id=conversation_id,
        kind="pi.test",
        version=1,
        input={"x": 1},
        state=TaskState(status=status, checkpoint={"phase": "start"}),
    )


def submission(id: int, conversation_id: int, **kwargs) -> SubmissionRecord:
    return SubmissionRecord(
        id=id, conversation_id=conversation_id, type="input", status="queued", **kwargs
    )


@pytest.fixture(params=BACKENDS)
def storage(request, tmp_path):
    return make_storage(request.param, tmp_path)


@pytest.fixture(params=BACKENDS)
def reopened_storage(request, tmp_path):
    """A backend factory that reopens the same backing file."""
    if request.param == "memory":
        pytest.skip("memory storage has no persistence to reopen")
    return make_storage(request.param, tmp_path)


def test_mint_ids_increase(storage):
    async def scenario():
        first = await storage.mint_id()
        second = await storage.mint_id()
        assert second == first + 1

    asyncio.run(scenario())


def test_commit_and_read_back(storage):
    async def scenario():
        cid = await storage.mint_id()
        eid = await storage.mint_id()
        seq = await storage.commit(
            [write_conversation(conversation(cid)), write_entry(entry(eid, cid, model=[]))]
        )
        assert seq == 1
        stored = await storage.conversation(cid)
        assert stored is not None and stored.id == cid
        found = await storage.entry_global(eid)
        assert found is not None and found[0].kind == "pi.user"
        assert found[1] == seq

    asyncio.run(scenario())


def test_entry_visibility_walks_fork_ancestry(storage):
    async def scenario():
        c1, c2, e1, e2 = await storage.mint_id(), await storage.mint_id(), await storage.mint_id(), await storage.mint_id()
        await storage.commit([write_conversation(conversation(c1)), write_entry(entry(e1, c1))])
        child = ConversationRecord(id=c2, parent={"conversation_id": c1, "at": e1})
        await storage.commit([write_conversation(child), write_entry(entry(e2, c2))])

        # Newest-first visible scan walks into the parent.
        page = await storage.scan_entries({"conversation_id": c2}, 10)
        assert [e.id for e in page.items] == [e2, e1]

        # A cutoff at the fork point excludes parent entries above it.
        page = await storage.scan_entries({"conversation_id": c2, "max_entry_id": e1}, 10)
        assert [e.id for e in page.items] == [e1]

        visible = await storage.entry_visible(c2, e1)
        assert visible is not None
        invisible = await storage.entry_visible(c1, e2)
        assert invisible is None

    asyncio.run(scenario())


def test_head_marker_scan(storage):
    async def scenario():
        c1, e1, e2 = await storage.mint_id(), await storage.mint_id(), await storage.mint_id()
        marker = entry(e2, c1, head=e1)
        await storage.commit(
            [write_conversation(conversation(c1)), write_entry(entry(e1, c1)), write_entry(marker)]
        )
        found = await storage.find_latest_head_marker(c1, None)
        assert found is not None and found.id == e2 and found.head == e1
        found = await storage.find_latest_head_marker(c1, e1)
        assert found is None

    asyncio.run(scenario())


def test_task_status_index_and_filters(storage):
    async def scenario():
        c1 = await storage.mint_id()
        t1, t2 = await storage.mint_id(), await storage.mint_id()
        await storage.commit(
            [write_conversation(conversation(c1)), write_task(task_record(t1, c1, "pending"))]
        )
        running = task_record(t2, c1, "running")
        await storage.commit([write_task(running)])

        page = await storage.scan_tasks({"status": "running"}, 10)
        assert [t.id for t in page.items] == [t2]

        # Status transitions move the index entry.
        running.state = TaskState(status="terminal", outcome=None)
        await storage.commit([write_task(running)])
        page = await storage.scan_tasks({"status": "running"}, 10)
        assert page.items == []
        page = await storage.scan_tasks({"status": "terminal"}, 10)
        assert [t.id for t in page.items] == [t2]

    asyncio.run(scenario())


def test_submission_request_index(storage):
    async def scenario():
        c1, s1 = await storage.mint_id(), await storage.mint_id()
        await storage.commit([write_conversation(conversation(c1))])
        sub = submission(s1, c1, request_id="req-1")
        await storage.commit([write_submission(sub)])
        found = await storage.submission_by_request(c1, "req-1")
        assert found is not None and found.id == s1
        assert await storage.submission_by_request(c1, "req-2") is None

        # Settling keeps the request index pointing at the same record.
        sub.status = "done"
        sub.answer = 123
        await storage.commit([write_submission(sub)])
        found = await storage.submission_by_request(c1, "req-1")
        assert found is not None and found.status == "done"

    asyncio.run(scenario())


def test_duplicate_global_id_rejected(storage):
    async def scenario():
        c1 = await storage.mint_id()
        await storage.commit([write_conversation(conversation(c1))])
        with pytest.raises(StorageRejected):
            await storage.commit([write_conversation(conversation(c1))])

    asyncio.run(scenario())


def test_commit_seq_increases(storage):
    async def scenario():
        c1 = await storage.mint_id()
        seq1 = await storage.commit([write_conversation(conversation(c1))])
        seq2 = await storage.commit([write_entry(entry(await storage.mint_id(), c1))])
        assert seq2 > seq1

    asyncio.run(scenario())


def test_documents_create_change_retire(storage):
    async def scenario():
        c1 = await storage.mint_id()
        d1 = await storage.mint_id()
        await storage.commit([write_conversation(conversation(c1))])
        record = {
            "id": d1,
            "kind": "pi.test.doc",
            "scope": {"kind": "conversation", "conversation_id": c1},
            "history": "rewindable",
            "fork": "current",
        }
        await storage.commit([write_document_create(record, 1, {"count": 0})])

        found = await storage.find_document({"kind": "pi.test.doc", "scope": record["scope"], "key": None}, "current")
        assert found is not None and found.id == d1

        await storage.commit([write_document_change(d1, 1, {"count": 5})])
        stored = await storage.document(d1, "current")
        assert stored is not None and stored.value == {"count": 5}
        # Historical read at the creation commit sees the original value.
        # Commit 1 carried the conversation; the document was created at commit 2.
        stored = await storage.document(d1, 2)
        assert stored is not None and stored.value == {"count": 0}

        await storage.commit([write_document_retire(d1)])
        assert await storage.find_document({"kind": "pi.test.doc", "scope": record["scope"], "key": None}, "current") is None

    asyncio.run(scenario())


def test_document_scope_is_detached_from_input(storage):
    async def scenario():
        cid = await storage.mint_id()
        rid = await storage.mint_id()
        await storage.commit([write_conversation(conversation(cid))])
        scope = {"kind": "conversation", "conversation_id": cid}
        record = {"id": rid, "kind": "pi.test.doc", "scope": scope, "history": "rewindable", "fork": "current"}
        await storage.commit([write_document_create(record, 1, {})])
        scope["conversation_id"] = 99

        stored = await storage.document(rid)
        assert stored is not None
        assert stored.record.scope == {"kind": "conversation", "conversation_id": cid}
        address = {"kind": "pi.test.doc", "scope": {"kind": "conversation", "conversation_id": cid}}
        found = await storage.find_document(address)
        assert found is not None and found.id == rid
        page = await storage.scan_documents({"scope": address["scope"], "at": "current"}, 10)
        assert [item.id for item in page.items] == [rid]
        await storage.commit([write_document_retire(rid)])
        assert await storage.find_document(address) is None

    asyncio.run(scenario())


def test_record_json_roundtrip(storage):
    record = task_record(5, 2)
    rebuilt = record_from_dict(record_to_dict(record))
    assert rebuilt == record
    entry_record = entry(7, 2, data={"k": [1, 2]})
    rebuilt = record_from_dict(record_to_dict(entry_record))
    assert rebuilt == entry_record
    conversation_record = conversation(3, parent={"conversation_id": 1, "at": 2})
    rebuilt = record_from_dict(record_to_dict(conversation_record))
    assert rebuilt == conversation_record


def test_jsonl_reopens_with_state(tmp_path):
    async def scenario():
        path = str(tmp_path / "log.jsonl")
        first = JsonlStorage(path)
        c1 = await first.mint_id()
        e1 = await first.mint_id()
        await first.commit([write_conversation(conversation(c1)), write_entry(entry(e1, c1))])
        await first.close()

        second = JsonlStorage(path)
        stored = await second.conversation(c1)
        assert stored is not None
        page = await second.scan_entries({"conversation_id": c1}, 10)
        assert [e.id for e in page.items] == [e1]
        assert second.next_seq == 2
        await second.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("tail", [b'{"seq":2,"writes":[', b'{"text":"\xe4\xb8', b'{"seq":2,"writes":[]}'])
def test_jsonl_recovers_unfinished_tail_and_can_append(tmp_path, tail):
    async def scenario():
        path = tmp_path / "log.jsonl"
        first = JsonlStorage(str(path))
        await first.commit([write_conversation(conversation(2))])
        await first.close()
        complete = path.read_bytes()
        path.write_bytes(complete + tail)

        recovered = JsonlStorage(str(path))
        assert path.read_bytes() == complete
        assert await recovered.conversation(2) is not None
        assert recovered.next_seq == 2
        assert await recovered.commit([write_entry(entry(3, 2))]) == 2
        await recovered.close()

        reopened = JsonlStorage(str(path))
        assert await reopened.entry_global(3) is not None
        assert reopened.next_seq == 3
        await reopened.close()

    asyncio.run(scenario())


def test_jsonl_rejects_corrupt_complete_line_without_truncating(tmp_path):
    path = tmp_path / "log.jsonl"
    original = b'{broken}\n{"seq":2,"writes":[]}\n'
    path.write_bytes(original)
    with pytest.raises(json.JSONDecodeError):
        JsonlStorage(str(path))
    assert path.read_bytes() == original


def test_sqlite_reopens_with_state(tmp_path):
    async def scenario():
        path = str(tmp_path / "store.db")
        first = SqliteStorage(path)
        c1 = await first.mint_id()
        t1 = await first.mint_id()
        await first.commit([write_conversation(conversation(c1)), write_task(task_record(t1, c1))])
        await first.close()

        second = SqliteStorage(path)
        stored = await second.task(t1)
        assert stored is not None and stored.kind == "pi.test"
        assert second.next_seq == 2
        await second.close()

    asyncio.run(scenario())


def test_sqlite_document_record_survives_reopen(tmp_path):
    async def scenario():
        path = str(tmp_path / "document.db")
        storage = SqliteStorage(path)
        record = DocumentRecord(id=2, kind="example", scope={"kind": "session"})
        await storage.commit([write_document_create(record, 1, {"count": 7})])
        await storage.close()
        reopened = SqliteStorage(path)
        stored = await reopened.document(2)
        assert stored is not None and stored.value == {"count": 7}
        assert stored.record.kind == "example"
        await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [object(), "\ud800"])
def test_sqlite_encoding_failure_leaves_connection_usable(tmp_path, invalid):
    async def scenario():
        storage = SqliteStorage(str(tmp_path / "failure.db"))
        with pytest.raises((TypeError, UnicodeEncodeError)):
            await storage.commit([write_entry(entry(2, 3, data=invalid))])
        assert not storage.conn.in_transaction
        assert storage.next_seq == 1
        assert await storage.entry_global(2) is None
        assert storage.conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0] == 0
        assert await storage.commit([write_conversation(conversation(3))]) == 1
        await storage.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["jsonl", "sqlite"])
def test_context_edits_survive_reopen(tmp_path, backend):
    async def scenario():
        storage = make_storage(backend, tmp_path)
        edits = [
            ContextEdit(target=3, action="omit"),
            ContextEdit(target=3, action="replace", messages=[UserMessage(content="replacement", timestamp=123)]),
        ]
        record = entry(4, 2, edits=edits)
        # Exercise the actual JSON boundary, not just dict conversion.
        assert record_from_dict(json.loads(json.dumps(record_to_dict(record)))) == record
        await storage.commit([write_conversation(conversation(2)), write_entry(entry(3, 2)), write_entry(record)])
        await storage.close()
        reopened = make_storage(backend, tmp_path)
        found = await reopened.entry_global(4)
        assert found is not None and found[0] == record
        await reopened.close()

    asyncio.run(scenario())


def test_closed_storage_rejects(storage):
    async def scenario():
        await storage.close()
        with pytest.raises(StorageRejected):
            await storage.mint_id()

    asyncio.run(scenario())
