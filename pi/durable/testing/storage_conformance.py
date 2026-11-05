"""Storage conformance suite (port of ``testing/storage-conformance.ts``).

Runner-independent cases over the Python ``Storage`` protocol (snake_case API).
`with_storage` must call and await its callback exactly once per case with a
fresh backend. Adapted from the TS cases:

- lone-surrogate string identities become astral-character identities (the
  Python backends store UTF-8 text; lone surrogates are not valid UTF-8);
- prototype-pollution keys are plain dict keys in Python.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..errors import StorageRejected
from ..ids import ROOT_CONVERSATION_ID, id_from_number
from ..types import (
    ConversationRecord,
    EntryRecord,
    SubmissionRecord,
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
from .types import ConformanceAssertions, StorageConformanceCase, StorageConformanceOptions

MAX_SAFE_ID = 2**53 - 1


def _replace(record, **changes):
    import dataclasses

    return dataclasses.replace(record, **changes)


async def _create_root(storage) -> int:
    await storage.commit([write_conversation(ConversationRecord(id=ROOT_CONVERSATION_ID))])
    return ROOT_CONVERSATION_ID


def _pending_task(task_id: int, conversation_id: int, phase: str = "ready") -> TaskRecord:
    return TaskRecord(
        id=task_id,
        conversation_id=conversation_id,
        kind="test.task",
        version=1,
        input={"value": task_id},
        state=TaskState(status="pending", checkpoint={"phase": phase}),
    )


def _entry(entry_id: int, conversation_id: int, kind: str = "pi.user", **extra) -> EntryRecord:
    return EntryRecord(id=entry_id, conversation_id=conversation_id, kind=kind, **extra)


def _submission(submission_id: int, conversation_id: int, **extra) -> SubmissionRecord:
    fields: Dict[str, Any] = {"type": "input", "status": "queued"}
    fields.update(extra)
    return SubmissionRecord(id=submission_id, conversation_id=conversation_id, **fields)


def create_storage_conformance(options: StorageConformanceOptions) -> List[StorageConformanceCase]:
    assert_obj: ConformanceAssertions = options.assertions

    def case(name: str, test) -> StorageConformanceCase:
        return StorageConformanceCase(name=name, run=lambda: options.with_storage(test))

    cases: List[StorageConformanceCase] = [
        case("reserves ID 1 for the immutable root conversation", _case_root_conversation),
        case("commits mixed table writes atomically and rolls all of them back on failure", _case_atomic_rollback),
        case("detaches retained writes and every returned record", _case_detached),
        case("detaches prototype-like JSON keys without changing object prototypes", _case_proto_keys),
        case("indexes entries committed out of ID order", _case_out_of_order),
        case("continues an entry cursor below its last item after a newer commit", _case_entry_cursor),
        case("paginates conversations by opaque cursor in ascending ID order", _case_conversation_cursor),
        case("filters and pages conversations by durable owner edges", _case_owner_edges),
        case("scans deep fork history newest-first through every ancestor cap", _case_fork_history),
        case("replaces complete task records and pages filtered task scans", _case_task_replace),
        case("stores owners and scans waiting and completing tasks by status", _case_task_status),
        case(
            "indexes request IDs per conversation and replaces complete submission records",
            _case_request_ids,
        ),
        case("stores passive write submissions without input-only lifecycle states", _case_passive_submissions),
        case(
            "reconstructs rewindable documents and preserves half-open incarnations",
            _case_rewindable_documents,
        ),
        case("streams long document tails across root replacement deltas", _case_long_tails),
        case(
            "copies stored document bases independently and rejects ambiguous sources",
            _case_document_copies,
        ),
        case(
            "uses bases for version transitions and rejects historical reads of current-only documents",
            _case_version_transitions,
        ),
        case("indexes logical addresses and exact-scope scans independently", _case_logical_addresses),
        case(
            "keeps document lifecycle failures atomic and gives create-plus-retire an empty lifetime",
            _case_document_lifecycle,
        ),
        case(
            "rolls back record tables and secondary indexes when a document command fails",
            _case_document_rollback,
        ),
        case("keeps indexed string identities lossless", _case_string_identities),
        case(
            "keeps one global record ID namespace and rejects exhausted ID minting",
            _case_id_namespace,
        ),
        case("rejects every operation after close", _case_after_close),
    ]
    return cases


async def _case_root_conversation(storage) -> None:
    assert_obj = _A
    assert_obj.strict_equal(await storage.mint_id(), 2)
    root_id = await _create_root(storage)
    assert_obj.strict_equal(root_id, ROOT_CONVERSATION_ID)
    found = await storage.conversation(ROOT_CONVERSATION_ID)
    assert_obj.deep_equal(found, ConversationRecord(id=ROOT_CONVERSATION_ID))
    await assert_obj.rejects(
        storage.commit([write_conversation(ConversationRecord(id=ROOT_CONVERSATION_ID))]),
        f"ID {ROOT_CONVERSATION_ID} already belongs to conversation",
    )


async def _case_atomic_rollback(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    entry_id = await storage.mint_id()
    task_id = await storage.mint_id()
    submission_id = await storage.mint_id()
    task = _pending_task(task_id, root_id)
    stored_input = _submission(submission_id, root_id, request_id="request-1", status="placed", entry=entry_id)
    initial_seq = await storage.commit(
        [
            write_entry(_entry(entry_id, root_id, "pi.user", data={"text": "hello"})),
            write_task(task),
            write_submission(stored_input),
        ]
    )

    found = await storage.entry_global(entry_id)
    assert_obj.deep_equal(found, (_entry(entry_id, root_id, "pi.user", data={"text": "hello"}), initial_seq))
    assert_obj.deep_equal(await storage.task(task_id), task)
    assert_obj.deep_equal(await storage.submission(submission_id), stored_input)

    transient_entry_id = await storage.mint_id()
    running_task = TaskRecord(
        id=task.id,
        conversation_id=task.conversation_id,
        kind=task.kind,
        version=task.version,
        input=task.input,
        state=TaskState(status="running", checkpoint={"phase": "effect"}),
    )
    done_input = SubmissionRecord(
        id=stored_input.id,
        conversation_id=stored_input.conversation_id,
        type=stored_input.type,
        status="done",
        request_id=stored_input.request_id,
        entry=stored_input.entry,
        answer=transient_entry_id,
    )
    await assert_obj.rejects(
        storage.commit(
            [
                write_task(running_task),
                write_submission(done_input),
                write_entry(_entry(transient_entry_id, root_id, "pi.assistant")),
                write_conversation(ConversationRecord(id=root_id)),
            ]
        ),
        f"ID {root_id} already belongs to conversation",
    )

    assert_obj.deep_equal(await storage.task(task_id), task)
    assert_obj.deep_equal(await storage.submission(submission_id), stored_input)
    assert_obj.strict_equal(await storage.entry_global(transient_entry_id), None)
    after_rollback_seq = await storage.commit(
        [write_entry(_entry(await storage.mint_id(), root_id, "after-rollback"))]
    )
    assert_obj.greater_than(after_rollback_seq, initial_seq)


async def _case_detached(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    entry_id = await storage.mint_id()
    task_id = await storage.mint_id()
    submission_id = await storage.mint_id()
    entry_data = {"nested": [1, 2]}
    checkpoint = {"phase": "ready", "nested": {"count": 1}}
    detail = {"codes": ["initial"]}
    stored_task = TaskRecord(
        id=task_id,
        conversation_id=root_id,
        kind="test.task",
        version=1,
        input={"value": task_id},
        state=TaskState(status="pending", checkpoint=checkpoint),
    )
    stored_input = _submission(
        submission_id, root_id, status="unanswered", reason="failed", detail=detail
    )
    await storage.commit(
        [
            write_entry(_entry(entry_id, root_id, "note", data=entry_data)),
            write_task(stored_task),
            write_submission(stored_input),
        ]
    )

    entry_data["nested"].append(3)
    checkpoint["nested"]["count"] = 2
    detail["codes"].append("mutated")
    assert_obj.deep_equal((await storage.entry_global(entry_id))[0].data, {"nested": [1, 2]})
    assert_obj.deep_equal(
        (await storage.task(task_id)).state,
        TaskState(status="pending", checkpoint={"phase": "ready", "nested": {"count": 1}}),
    )
    assert_obj.deep_equal((await storage.submission(submission_id)).detail, {"codes": ["initial"]})

    read_entry = (await storage.entry_global(entry_id))[0]
    read_entry.data["nested"].append(9)
    read_task = await storage.task(task_id)
    read_task.state.checkpoint["nested"]["count"] = 9
    read_input = await storage.submission(submission_id)
    read_input.detail["codes"].append("read mutation")

    assert_obj.deep_equal((await storage.entry_global(entry_id))[0].data, {"nested": [1, 2]})
    assert_obj.deep_equal(
        (await storage.task(task_id)).state,
        TaskState(status="pending", checkpoint={"phase": "ready", "nested": {"count": 1}}),
    )
    assert_obj.deep_equal((await storage.submission(submission_id)).detail, {"codes": ["initial"]})


async def _case_proto_keys(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    entry_id = await storage.mint_id()
    import json

    data = json.loads('{"__proto__":{"polluted":false},"constructor":{"label":"stored"},"toString":"value"}')
    await storage.commit([write_entry(_entry(entry_id, root_id, "note", data=data))])

    data["__proto__"]["polluted"] = True
    data["constructor"]["label"] = "mutated"
    first_read = (await storage.entry_global(entry_id))[0].data
    assert_obj.deep_equal(first_read["__proto__"], {"polluted": False})
    assert_obj.deep_equal(first_read["constructor"], {"label": "stored"})
    assert_obj.strict_equal(first_read["toString"], "value")

    first_read["__proto__"]["polluted"] = True
    second_read = (await storage.entry_global(entry_id))[0].data
    assert_obj.deep_equal(second_read["__proto__"], {"polluted": False})
    assert_obj.deep_equal(second_read["constructor"], {"label": "stored"})
    assert_obj.strict_equal(second_read["toString"], "value")


async def _case_out_of_order(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    await storage.commit(
        [
            write_entry(_entry(id_from_number(30), root_id)),
            write_entry(_entry(id_from_number(10), root_id)),
            write_entry(_entry(id_from_number(20), root_id, "marker", head=10)),
        ]
    )
    page = await storage.scan_entries({"conversation_id": root_id}, 10)
    assert_obj.deep_equal([item.id for item in page.items], [30, 20, 10])
    marker = await storage.find_latest_head_marker(root_id)
    assert_obj.strict_equal(marker.id, 20)


async def _case_entry_cursor(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    oldest_id = await storage.mint_id()
    middle_id = await storage.mint_id()
    newest_id = await storage.mint_id()
    await storage.commit(
        [
            write_entry(_entry(oldest_id, root_id)),
            write_entry(_entry(middle_id, root_id)),
            write_entry(_entry(newest_id, root_id)),
        ]
    )

    first = await storage.scan_entries({"conversation_id": root_id}, 2)
    assert_obj.deep_equal([item.id for item in first.items], [newest_id, middle_id])
    appended_id = await storage.mint_id()
    await storage.commit([write_entry(_entry(appended_id, root_id))])
    second = await storage.scan_entries({"conversation_id": root_id}, 2, first.next)
    assert_obj.deep_equal([item.id for item in second.items], [oldest_id])
    assert_obj.strict_equal(second.next, None)


async def _case_conversation_cursor(storage) -> None:
    import json as _json

    assert_obj = _A
    root_id = await _create_root(storage)
    second_id = await storage.mint_id()
    third_id = await storage.mint_id()
    await storage.commit(
        [
            write_conversation(ConversationRecord(id=third_id)),
            write_conversation(ConversationRecord(id=second_id)),
        ]
    )

    first = await storage.scan_conversations({}, 2)
    assert_obj.deep_equal([item.id for item in first.items], [root_id, second_id])
    assert_obj.ok(first.next is not None, "cursor present")
    round_tripped = _json.loads(_json.dumps(first.next))
    second = await storage.scan_conversations({}, 2, round_tripped)
    assert_obj.deep_equal([item.id for item in second.items], [third_id])
    assert_obj.strict_equal(second.next, None)


async def _case_owner_edges(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    other_owner_id = await storage.mint_id()
    first_task_id = await storage.mint_id()
    second_task_id = await storage.mint_id()
    first_id = await storage.mint_id()
    second_id = await storage.mint_id()
    third_id = await storage.mint_id()
    await storage.commit(
        [
            write_conversation(ConversationRecord(id=other_owner_id)),
            write_conversation(
                ConversationRecord(id=first_id, owner={"conversation_id": root_id, "task_id": first_task_id})
            ),
            write_conversation(
                ConversationRecord(id=second_id, owner={"conversation_id": root_id, "task_id": second_task_id})
            ),
            write_conversation(
                ConversationRecord(id=third_id, owner={"conversation_id": other_owner_id, "task_id": first_task_id})
            ),
        ]
    )

    first = await storage.scan_conversations({"owner_conversation_id": root_id}, 1)
    assert_obj.deep_equal([item.id for item in first.items], [first_id])
    assert_obj.ok(first.next is not None, "cursor present")
    second = await storage.scan_conversations({"owner_conversation_id": root_id}, 1, first.next)
    assert_obj.deep_equal([item.id for item in second.items], [second_id])
    assert_obj.strict_equal(second.next, None)
    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_conversations({"owner_task_id": first_task_id}, 10)).items],
        [first_id, third_id],
    )
    assert_obj.deep_equal(
        [
            item.id
            for item in (
                await storage.scan_conversations(
                    {"owner_conversation_id": root_id, "owner_task_id": first_task_id}, 10
                )
            ).items
        ],
        [first_id],
    )


async def _case_fork_history(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    root_first = await storage.mint_id()
    root_fork_point = await storage.mint_id()
    root_excluded_same_commit = await storage.mint_id()
    root_entries_seq = await storage.commit(
        [
            write_entry(_entry(root_first, root_id)),
            write_entry(_entry(root_fork_point, root_id, "marker", head=root_first)),
            write_entry(_entry(root_excluded_same_commit, root_id)),
        ]
    )
    child_id = await storage.mint_id()
    await storage.commit(
        [write_conversation(ConversationRecord(id=child_id, parent={"conversation_id": root_id, "at": root_fork_point}))]
    )
    child_fork_point = await storage.mint_id()
    child_excluded = await storage.mint_id()
    await storage.commit(
        [write_entry(_entry(child_fork_point, child_id, "note")), write_entry(_entry(child_excluded, child_id))]
    )
    root_excluded_later = await storage.mint_id()
    await storage.commit([write_entry(_entry(root_excluded_later, root_id))])
    grandchild_id = await storage.mint_id()
    await storage.commit(
        [
            write_conversation(
                ConversationRecord(id=grandchild_id, parent={"conversation_id": child_id, "at": child_fork_point})
            )
        ]
    )
    grandchild_head = await storage.mint_id()
    grandchild_tail = await storage.mint_id()
    grandchild_entries_seq = await storage.commit(
        [
            write_entry(_entry(grandchild_head, grandchild_id, "marker", head=grandchild_head)),
            write_entry(_entry(grandchild_tail, grandchild_id)),
        ]
    )
    child_excluded_later = await storage.mint_id()
    await storage.commit([write_entry(_entry(child_excluded_later, child_id))])

    first = await storage.scan_entries({"conversation_id": grandchild_id}, 2)
    assert_obj.deep_equal([item.id for item in first.items], [grandchild_tail, grandchild_head])
    second = await storage.scan_entries({"conversation_id": grandchild_id}, 2, first.next)
    assert_obj.deep_equal([item.id for item in second.items], [child_fork_point, root_fork_point])
    third = await storage.scan_entries({"conversation_id": grandchild_id}, 2, second.next)
    assert_obj.deep_equal([item.id for item in third.items], [root_first])
    assert_obj.strict_equal(third.next, None)

    current_marker = await storage.find_latest_head_marker(grandchild_id)
    assert_obj.strict_equal(current_marker.id, grandchild_head)
    assert_obj.strict_equal(current_marker.head, grandchild_head)
    historical_marker = await storage.find_latest_head_marker(grandchild_id, child_fork_point)
    assert_obj.strict_equal(historical_marker.id, root_fork_point)
    assert_obj.strict_equal(historical_marker.head, root_first)
    assert_obj.strict_equal(await storage.find_latest_head_marker(grandchild_id, root_first), None)

    active_first = await storage.scan_entries(
        {"conversation_id": grandchild_id, "min_entry_id": current_marker.head}, 1
    )
    assert_obj.deep_equal([item.id for item in active_first.items], [grandchild_tail])
    assert_obj.ok(active_first.next is not None, "cursor present")
    active_second = await storage.scan_entries(
        {"conversation_id": grandchild_id, "min_entry_id": current_marker.head}, 1, active_first.next
    )
    assert_obj.deep_equal([item.id for item in active_second.items], [grandchild_head])
    assert_obj.strict_equal(active_second.next, None)

    page = await storage.scan_entries(
        {
            "conversation_id": grandchild_id,
            "min_entry_id": historical_marker.head,
            "max_entry_id": child_fork_point,
        },
        10,
    )
    assert_obj.deep_equal([item.id for item in page.items], [child_fork_point, root_fork_point, root_first])

    assert_obj.deep_equal(
        await storage.entry_global(root_first), (_entry(root_first, root_id), root_entries_seq)
    )
    assert_obj.strict_equal((await storage.entry_global(root_fork_point))[1], root_entries_seq)
    assert_obj.strict_equal((await storage.entry_global(grandchild_head))[1], grandchild_entries_seq)
    assert_obj.strict_equal((await storage.entry_global(grandchild_tail))[1], grandchild_entries_seq)
    assert_obj.strict_equal(await storage.entry_global(999_999), None)

    assert_obj.deep_equal(
        await storage.entry_visible(grandchild_id, root_first), (_entry(root_first, root_id), root_entries_seq)
    )
    assert_obj.strict_equal((await storage.entry_visible(grandchild_id, child_fork_point))[0].conversation_id, child_id)
    assert_obj.strict_equal((await storage.entry_visible(grandchild_id, grandchild_tail))[1], grandchild_entries_seq)
    assert_obj.strict_equal(await storage.entry_visible(grandchild_id, root_excluded_same_commit), None)
    assert_obj.strict_equal(await storage.entry_visible(grandchild_id, root_excluded_later), None)
    assert_obj.strict_equal(await storage.entry_visible(grandchild_id, child_excluded), None)
    assert_obj.strict_equal(await storage.entry_visible(grandchild_id, child_excluded_later), None)
    assert_obj.strict_equal(await storage.entry_visible(root_id, grandchild_head), None)
    assert_obj.strict_equal(await storage.entry_visible(grandchild_id, 999_999), None)
    await assert_obj.rejects(storage.entry_visible(999_999, root_first), "Unknown conversation")
    await assert_obj.rejects(
        storage.scan_entries({"conversation_id": 999_999}, 10), "Unknown conversation"
    )


async def _case_task_replace(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    first_id = await storage.mint_id()
    second_id = await storage.mint_id()
    third_id = await storage.mint_id()
    first = TaskRecord(
        id=first_id,
        conversation_id=root_id,
        kind="test.task",
        version=1,
        input={"value": first_id},
        state=TaskState(status="pending", checkpoint={"phase": "ready"}),
        memos={"winner": "first"},
    )
    second = TaskRecord(
        id=second_id,
        conversation_id=root_id,
        kind="test.task",
        version=1,
        input={"value": second_id},
        state=TaskState(status="pending", checkpoint={"phase": "ready"}),
        background=True,
    )
    third = TaskRecord(
        id=third_id,
        conversation_id=root_id,
        kind="test.task",
        version=1,
        input={"value": third_id},
        state=TaskState(status="pending", checkpoint={"phase": "ready"}),
        abort_requested=True,
    )
    await storage.commit([write_task(first), write_task(second), write_task(third)])

    running = TaskRecord(
        id=first.id,
        conversation_id=first.conversation_id,
        kind=first.kind,
        version=first.version,
        input=first.input,
        state=TaskState(status="running", checkpoint={"phase": "effect", "attempt": 1}),
        background=first.background,
        abort_requested=True,
        memos=first.memos,
    )
    await storage.commit([write_task(running)])
    assert_obj.deep_equal(await storage.task(first_id), running)
    terminal = TaskRecord(
        id=first_id,
        conversation_id=root_id,
        kind=first.kind,
        version=first.version,
        input=first.input,
        state=TaskState(status="terminal", outcome={"status": "completed", "result": {"entryId": 99}}),
        abort_requested=True,
    )
    await storage.commit([write_task(terminal)])
    assert_obj.deep_equal(await storage.task(first_id), terminal)

    pending_page = await storage.scan_tasks({"status": "pending"}, 1)
    assert_obj.deep_equal([item.id for item in pending_page.items], [second_id])
    assert_obj.ok(pending_page.next is not None, "cursor present")
    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_tasks({"status": "pending"}, 1, pending_page.next)).items],
        [third_id],
    )
    assert_obj.deep_equal(
        (await storage.scan_tasks({"status": "terminal", "abort_requested": True}, 10)).items, [terminal]
    )
    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_tasks({"background": True}, 10)).items], [second_id]
    )


async def _case_task_status(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    owner_id = await storage.mint_id()
    waiting_id = await storage.mint_id()
    completing_id = await storage.mint_id()
    owner = _pending_task(owner_id, root_id)
    waiting = TaskRecord(
        id=waiting_id,
        conversation_id=root_id,
        kind="test.task",
        version=1,
        input={"value": waiting_id},
        owner=owner_id,
        state=TaskState(status="waiting", checkpoint={"phase": "next"}, on=[owner_id], policy="allSettled"),
        memos={"kept": True},
    )
    completing = TaskRecord(
        id=completing_id,
        conversation_id=root_id,
        kind="test.task",
        version=1,
        input={"value": completing_id},
        owner=owner_id,
        state=TaskState(status="completing", outcome={"status": "failed", "error": {"message": "held"}}),
    )
    await storage.commit([write_task(owner), write_task(waiting), write_task(completing)])
    assert_obj.deep_equal(await storage.task(waiting_id), waiting)
    assert_obj.deep_equal(await storage.task(completing_id), completing)

    async def scan(status: str):
        return (await storage.scan_tasks({"status": status}, 10)).items

    assert_obj.deep_equal(await scan("waiting"), [waiting])
    assert_obj.deep_equal(await scan("completing"), [completing])
    assert_obj.deep_equal([item.id for item in await scan("pending")], [owner_id])
    terminal = TaskRecord(
        id=completing.id,
        conversation_id=completing.conversation_id,
        kind=completing.kind,
        version=completing.version,
        input=completing.input,
        owner=completing.owner,
        background=completing.background,
        abort_requested=completing.abort_requested,
        state=TaskState(status="terminal", outcome=completing.state.outcome),
    )
    await storage.commit([write_task(terminal)])
    assert_obj.deep_equal(await scan("completing"), [])
    assert_obj.deep_equal(await scan("terminal"), [terminal])


async def _case_request_ids(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    second_conversation_id = await storage.mint_id()
    await storage.commit([write_conversation(ConversationRecord(id=second_conversation_id))])
    first_id = await storage.mint_id()
    second_id = await storage.mint_id()
    other_conversation_id = await storage.mint_id()
    first = _submission(first_id, root_id, request_id="same")
    second = _submission(second_id, root_id, request_id="other")
    other_conversation = _submission(other_conversation_id, second_conversation_id, request_id="same")
    await storage.commit(
        [write_submission(first), write_submission(second), write_submission(other_conversation)]
    )
    assert_obj.deep_equal(await storage.submission_by_request(root_id, "same"), first)
    assert_obj.deep_equal(await storage.submission_by_request(second_conversation_id, "same"), other_conversation)

    placed_second = SubmissionRecord(
        id=second.id,
        conversation_id=second.conversation_id,
        type=second.type,
        status="placed",
        request_id=second.request_id,
        entry=await storage.mint_id(),
    )
    await storage.commit([write_submission(placed_second)])
    assert_obj.deep_equal(await storage.submission(second_id), placed_second)
    assert_obj.deep_equal(await storage.submission_by_request(root_id, "other"), placed_second)

    async def ids(query: Dict[str, Any]) -> List[int]:
        found: List[int] = []
        cursor = None
        while True:
            page = await storage.scan_submissions(query, 1, cursor)
            found.extend(item.id for item in page.items)
            cursor = page.next
            if cursor is None:
                return found

    assert_obj.deep_equal(await ids({}), [first_id, second_id, other_conversation_id])
    assert_obj.deep_equal(await ids({"conversation_id": root_id}), [first_id, second_id])
    # A status change moves the record between status scans.
    assert_obj.deep_equal(await ids({"status": "queued"}), [first_id, other_conversation_id])
    assert_obj.deep_equal(await ids({"status": "placed"}), [second_id])
    assert_obj.deep_equal(
        await ids({"conversation_id": second_conversation_id, "status": "queued"}), [other_conversation_id]
    )
    assert_obj.deep_equal(await ids({"conversation_id": second_conversation_id, "status": "placed"}), [])
    assert_obj.deep_equal(
        (await storage.scan_submissions({"status": "placed"}, 10)).items, [placed_second]
    )


async def _case_passive_submissions(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    done_id = await storage.mint_id()
    failed_id = await storage.mint_id()
    queued_done = SubmissionRecord(
        id=done_id, conversation_id=root_id, type="write", status="queued", request_id="passive-done"
    )
    queued_failed = SubmissionRecord(
        id=failed_id, conversation_id=root_id, type="write", status="queued", request_id="passive-failed"
    )
    await storage.commit([write_submission(queued_done), write_submission(queued_failed)])

    done = SubmissionRecord(
        id=done_id, conversation_id=root_id, type="write", status="done",
        request_id="passive-done", entry=await storage.mint_id(),
    )
    unanswered = SubmissionRecord(
        id=failed_id, conversation_id=root_id, type="write", status="unanswered",
        request_id="passive-failed", reason="closed", detail={"retryable": False},
    )
    await storage.commit([write_submission(done), write_submission(unanswered)])
    assert_obj.deep_equal(await storage.submission(done_id), done)
    assert_obj.deep_equal(await storage.submission_by_request(root_id, "passive-done"), done)
    assert_obj.deep_equal(await storage.submission(failed_id), unanswered)
    assert_obj.deep_equal(await storage.submission_by_request(root_id, "passive-failed"), unanswered)


def _doc_create(record_id: int, kind: str, scope: Dict[str, Any], **extra) -> Dict[str, Any]:
    return {"id": record_id, "kind": kind, "scope": scope, **extra}


async def _case_rewindable_documents(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    first_id = await storage.mint_id()
    scope = {"kind": "conversation", "conversation_id": root_id}
    first_record = _doc_create(
        first_id, "conversation.notes", scope, history="rewindable", fork="asOf"
    )
    initial = {"items": ["a"], "nested": {"count": 1}}
    created_at = await storage.commit(
        [write_document_create(first_record, 1, initial)]
    )
    appended = ["b"]
    ops = [["p", ["items"], 1, 0, appended], ["s", ["nested", "count"], 2]]
    changed_at = await storage.commit(
        [{"type": "document.change", "id": first_id, "content": {"kind": "delta", "version": 1, "ops": ops}}]
    )

    initial["items"].append("caller mutation")
    appended.append("caller mutation")
    stored = await storage.document(first_id, created_at)
    assert_obj.strict_equal(stored.version, 1)
    assert_obj.deep_equal(stored.value, {"items": ["a"], "nested": {"count": 1}})
    assert_obj.strict_equal(stored.deltas_since_base, 0)
    changed = await storage.document(first_id, changed_at)
    assert_obj.deep_equal(changed.value, {"items": ["a", "b"], "nested": {"count": 2}})
    assert_obj.strict_equal(changed.deltas_since_base, 1)
    changed.value["items"].append("read mutation")
    assert_obj.deep_equal(
        (await storage.document(first_id, "current")).value, {"items": ["a", "b"], "nested": {"count": 2}}
    )

    checkpoint_at = await storage.commit(
        [
            {
                "type": "document.change",
                "id": first_id,
                "content": {"kind": "base", "version": 2, "value": {"items": ["checkpoint"], "nested": {"count": 3}}},
            }
        ]
    )
    replaced_at = await storage.commit(
        [
            {
                "type": "document.change",
                "id": first_id,
                "content": {
                    "kind": "delta",
                    "version": 2,
                    "ops": [["r", {"items": ["replacement"], "nested": {"count": 4}}]],
                },
            }
        ]
    )
    stored = await storage.document(first_id, changed_at)
    assert_obj.strict_equal(stored.version, 1)
    assert_obj.deep_equal(stored.value, {"items": ["a", "b"], "nested": {"count": 2}})
    stored = await storage.document(first_id, checkpoint_at)
    assert_obj.strict_equal(stored.version, 2)
    assert_obj.deep_equal(stored.value, {"items": ["checkpoint"], "nested": {"count": 3}})
    assert_obj.strict_equal(stored.deltas_since_base, 0)
    stored = await storage.document(first_id, replaced_at)
    assert_obj.deep_equal(stored.value, {"items": ["replacement"], "nested": {"count": 4}})
    assert_obj.strict_equal(stored.deltas_since_base, 1)
    assert_obj.strict_equal((await storage.document(first_id, "current")).deltas_since_base, 1)

    second_id = await storage.mint_id()
    retired_at = await storage.commit(
        [
            {
                "type": "document.create",
                "record": dict(first_record, id=second_id),
                "content": {"kind": "base", "version": 1, "value": {"items": ["new"]}},
            },
            {"type": "document.retire", "id": first_id},
            {
                "type": "document.change",
                "id": first_id,
                "content": {"kind": "delta", "version": 2, "ops": [["s", ["retiring"], True]]},
            },
        ]
    )
    address = {"kind": first_record["kind"], "scope": first_record["scope"]}
    assert_obj.strict_equal((await storage.find_document(address, changed_at)).id, first_id)
    assert_obj.deep_equal(
        await storage.find_document(address, retired_at),
        _doc_record(second_id, "conversation.notes", scope, created_at=retired_at, history="rewindable", fork="asOf"),
    )
    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_documents({"scope": scope, "at": changed_at}, 10)).items],
        [first_id],
    )
    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_documents({"scope": scope, "at": retired_at}, 10)).items],
        [second_id],
    )
    assert_obj.strict_equal(await storage.document(first_id, retired_at), None)
    assert_obj.deep_equal((await storage.document(second_id, "current")).value, {"items": ["new"]})


def _doc_record(record_id: int, kind: str, scope: Dict[str, Any], **extra):
    return _record_from(_doc_create(record_id, kind, scope, **extra))


def _record_from(record: Dict[str, Any]):
    from ..types import DocumentRecord

    return DocumentRecord(
        id=record["id"],
        kind=record["kind"],
        scope=record["scope"],
        created_at=record.get("created_at", 0),
        key=record.get("key"),
        retired_at=record.get("retired_at"),
        history=record.get("history"),
        fork=record.get("fork"),
    )


async def _case_long_tails(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    doc_id = await storage.mint_id()
    scope = {"kind": "conversation", "conversation_id": root_id}
    record = _doc_create(doc_id, "conversation.long-tail", scope, history="rewindable", fork="asOf")
    initial = {
        "revision": 0,
        "rows": [{"value": value, "stable": f"row-{value}"} for value in range(512)],
    }
    created_at = await storage.commit([write_document_create(record, 1, initial)])
    before_replacement = _clone(initial)
    before_replacement_at = created_at
    for revision in range(1, 25):
        index = (revision * 17) % len(before_replacement["rows"])
        before_replacement["rows"][index]["value"] = -revision
        before_replacement["revision"] = revision
        before_replacement_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": doc_id,
                    "content": {
                        "kind": "delta",
                        "version": 1,
                        "ops": [
                            ["s", ["rows", index, "value"], -revision],
                            ["s", ["revision"], revision],
                        ],
                    },
                }
            ]
        )

    replacement = {
        "revision": 100,
        "rows": [{"value": 10_000 + value, "stable": f"new-{value}"} for value in range(512)],
    }
    replacement_snapshot = _clone(replacement)
    replacement_at = await storage.commit(
        [
            {
                "type": "document.change",
                "id": doc_id,
                "content": {"kind": "delta", "version": 1, "ops": [["r", replacement]]},
            }
        ]
    )
    replacement["rows"][0]["value"] = -999

    current = _clone(replacement_snapshot)
    for revision in range(101, 125):
        index = (revision * 19) % len(current["rows"])
        current["rows"][index]["value"] = -revision
        current["revision"] = revision
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": doc_id,
                    "content": {
                        "kind": "delta",
                        "version": 1,
                        "ops": [
                            ["s", ["rows", index, "value"], -revision],
                            ["s", ["revision"], revision],
                        ],
                    },
                }
            ]
        )

    assert_obj.deep_equal((await storage.document(doc_id, created_at)).value, initial)
    assert_obj.deep_equal((await storage.document(doc_id, before_replacement_at)).value, before_replacement)
    assert_obj.deep_equal((await storage.document(doc_id, replacement_at)).value, replacement_snapshot)
    read = await storage.document(doc_id, "current")
    assert_obj.deep_equal(read.value, current)
    read.value["rows"][0]["value"] = -1_000
    assert_obj.deep_equal((await storage.document(doc_id, "current")).value, current)


def _clone(value):
    import copy

    return copy.deepcopy(value)


async def _case_document_copies(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    child_id = await storage.mint_id()
    second_child_id = await storage.mint_id()
    await storage.commit(
        [
            write_conversation(ConversationRecord(id=child_id)),
            write_conversation(ConversationRecord(id=second_child_id)),
        ]
    )
    source_id = await storage.mint_id()
    source_record = _doc_create(
        source_id, "copy.source", {"kind": "conversation", "conversation_id": root_id},
        history="rewindable", fork="asOf",
    )
    created_at = await storage.commit(
        [
            {
                "type": "document.create",
                "record": source_record,
                "content": {"kind": "base", "version": 2, "value": {"count": 1, "rows": [{"value": "base"}]}},
            }
        ]
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": source_id,
                "content": {
                    "kind": "delta",
                    "version": 2,
                    "ops": [["s", ["count"], 2], ["p", ["rows"], 1, 0, [{"value": "current"}]]],
                },
            }
        ]
    )
    historical_copy_id = await storage.mint_id()
    current_copy_id = await storage.mint_id()
    retired_copy_id = await storage.mint_id()

    def child_record(copy_id: int, conversation_id: int) -> Dict[str, Any]:
        return _doc_create(
            copy_id, "copy.source", {"kind": "conversation", "conversation_id": conversation_id},
            history="rewindable", fork="asOf",
        )

    await storage.commit(
        [
            {
                "type": "document.copy",
                "record": child_record(historical_copy_id, child_id),
                "source": {"id": source_id, "at": created_at},
            },
            {
                "type": "document.copy",
                "record": child_record(current_copy_id, second_child_id),
                "source": {"id": source_id, "at": "current"},
            },
            {
                "type": "document.copy",
                "record": child_record(retired_copy_id, root_id),
                "source": {"id": source_id, "at": "current"},
            },
            {"type": "document.retire", "id": retired_copy_id},
        ]
    )
    stored = await storage.document(historical_copy_id, "current")
    assert_obj.strict_equal(stored.version, 2)
    assert_obj.deep_equal(stored.value, {"count": 1, "rows": [{"value": "base"}]})
    stored = await storage.document(current_copy_id, "current")
    assert_obj.strict_equal(stored.version, 2)
    assert_obj.deep_equal(stored.value, {"count": 2, "rows": [{"value": "base"}, {"value": "current"}]})
    assert_obj.strict_equal(await storage.document(retired_copy_id, "current"), None)

    await storage.commit(
        [
            {
                "type": "document.change",
                "id": source_id,
                "content": {"kind": "base", "version": 2, "value": {"count": 99, "rows": []}},
            },
            {"type": "document.retire", "id": source_id},
        ]
    )
    assert_obj.deep_equal(
        (await storage.document(current_copy_id, "current")).value,
        {"count": 2, "rows": [{"value": "base"}, {"value": "current"}]},
    )

    latest_source_id = await storage.mint_id()
    latest_copy_id = await storage.mint_id()
    latest_source = _doc_create(
        latest_source_id, "copy.latest", {"kind": "conversation", "conversation_id": root_id},
        history="latest", fork="current",
    )
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": latest_source,
                "content": {"kind": "base", "version": 4, "value": {"retained": "copy"}},
            }
        ]
    )
    await storage.commit(
        [
            {
                "type": "document.copy",
                "record": dict(latest_source, id=latest_copy_id, scope={"kind": "conversation", "conversation_id": child_id}),
                "source": {"id": latest_source_id, "at": "current"},
            }
        ]
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": latest_source_id,
                "content": {"kind": "base", "version": 4, "value": {"retained": "source-only"}},
            },
            {"type": "document.retire", "id": latest_source_id},
        ]
    )
    stored = await storage.document(latest_copy_id, "current")
    assert_obj.strict_equal(stored.version, 4)
    assert_obj.deep_equal(stored.value, {"retained": "copy"})

    conflict_id = await storage.mint_id()
    conflict_error = None
    try:
        await storage.commit(
            [
                {
                    "type": "document.copy",
                    "record": child_record(conflict_id, child_id),
                    "source": {"id": current_copy_id, "at": "current"},
                },
                {"type": "document.retire", "id": current_copy_id},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        conflict_error = exc
    assert_obj.ok(conflict_error is not None and type(conflict_error).__name__ == "StorageRejected")
    assert_obj.strict_equal(await storage.document(conflict_id, "current"), None)
    assert_obj.deep_equal(
        (await storage.document(current_copy_id, "current")).value,
        {"count": 2, "rows": [{"value": "base"}, {"value": "current"}]},
    )

    mismatch_id = await storage.mint_id()
    mismatch_error = None
    try:
        await storage.commit(
            [
                {
                    "type": "document.copy",
                    "record": dict(child_record(mismatch_id, child_id), kind="copy.mismatch"),
                    "source": {"id": current_copy_id, "at": "current"},
                }
            ]
        )
    except Exception as exc:  # noqa: BLE001
        mismatch_error = exc
    assert_obj.ok(mismatch_error is not None and type(mismatch_error).__name__ == "StorageRejected")
    assert_obj.strict_equal(await storage.document(mismatch_id, "current"), None)


async def _case_version_transitions(storage) -> None:
    assert_obj = _A
    await _create_root(storage)
    doc_id = await storage.mint_id()
    record = _doc_create(doc_id, "session.settings", {"kind": "session"})
    await storage.commit(
        [{"type": "document.create", "record": record, "content": {"kind": "base", "version": 1, "value": {"count": 1}}}]
    )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": doc_id,
                "content": {"kind": "delta", "version": 1, "ops": [["s", ["count"], 2]]},
            }
        ]
    )
    migrated_at = await storage.commit(
        [
            {
                "type": "document.change",
                "id": doc_id,
                "content": {"kind": "base", "version": 2, "value": {"count": 3}},
            }
        ]
    )
    stored = await storage.document(doc_id, "current")
    assert_obj.strict_equal(stored.version, 2)
    assert_obj.deep_equal(stored.value, {"count": 3})
    await assert_obj.rejects(storage.document(doc_id, migrated_at), "does not retain historical content")

    await assert_obj.rejects(
        storage.commit(
            [
                {
                    "type": "document.change",
                    "id": doc_id,
                    "content": {"kind": "delta", "version": 1, "ops": [["s", ["count"], 4]]},
                }
            ]
        ),
        "version transition requires a base",
    )
    assert_obj.deep_equal((await storage.document(doc_id, "current")).value, {"count": 3})
    await storage.commit([{"type": "document.retire", "id": doc_id}])
    assert_obj.strict_equal(await storage.document(doc_id, "current"), None)


async def _case_logical_addresses(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    first_id = await storage.mint_id()
    second_id = await storage.mint_id()
    conversation_doc_id = await storage.mint_id()
    task_id = await storage.mint_id()
    task_singleton_id = await storage.mint_id()
    task_family_id = await storage.mint_id()
    task_other_kind_id = await storage.mint_id()
    created_at = await storage.commit(
        [
            write_task(_pending_task(task_id, root_id)),
            {
                "type": "document.create",
                "record": _doc_create(first_id, "cache", {"kind": "session"}, key="__proto__"),
                "content": {"kind": "base", "version": 1, "value": {"owner": "first"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(second_id, "cache", {"kind": "session"}, key="constructor"),
                "content": {"kind": "base", "version": 1, "value": {"owner": "second"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(
                    conversation_doc_id, "cache", {"kind": "conversation", "conversation_id": root_id},
                    history="latest", fork="current", key="__proto__",
                ),
                "content": {"kind": "base", "version": 1, "value": {"owner": "conversation"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(task_singleton_id, "task.cache", {"kind": "task", "task_id": task_id}),
                "content": {"kind": "base", "version": 1, "value": {"owner": "singleton"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(task_family_id, "task.cache", {"kind": "task", "task_id": task_id}, key="member"),
                "content": {"kind": "base", "version": 1, "value": {"owner": "family"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(task_other_kind_id, "task.other", {"kind": "task", "task_id": task_id}),
                "content": {"kind": "base", "version": 1, "value": {"owner": "other"}},
            },
        ]
    )

    assert_obj.strict_equal(
        (
            await storage.find_document({"kind": "cache", "scope": {"kind": "session"}, "key": "__proto__"})
        ).id,
        first_id,
    )
    assert_obj.strict_equal(
        len((await storage.scan_documents({"scope": {"kind": "session"}, "at": "current"}, 1)).items), 1
    )
    first_page = await storage.scan_documents({"scope": {"kind": "session"}, "at": "current"}, 1)
    second_page = await storage.scan_documents({"scope": {"kind": "session"}, "at": "current"}, 1, first_page.next)
    assert_obj.deep_equal(
        [item.id for item in [*first_page.items, *second_page.items]], [first_id, second_id]
    )
    assert_obj.deep_equal(
        [
            item.id
            for item in (
                await storage.scan_documents(
                    {"scope": {"kind": "conversation", "conversation_id": root_id}, "at": "current"}, 10
                )
            ).items
        ],
        [conversation_doc_id],
    )
    assert_obj.strict_equal(
        (await storage.find_document({"kind": "task.cache", "scope": {"kind": "task", "task_id": task_id}})).id,
        task_singleton_id,
    )
    assert_obj.strict_equal(
        (
            await storage.find_document(
                {"kind": "task.cache", "scope": {"kind": "task", "task_id": task_id}, "key": "member"}
            )
        ).id,
        task_family_id,
    )
    assert_obj.deep_equal(
        [
            item.id
            for item in (
                await storage.scan_documents(
                    {"scope": {"kind": "task", "task_id": task_id}, "at": "current", "kind": "task.cache"}, 10
                )
            ).items
        ],
        [task_singleton_id, task_family_id],
    )
    await assert_obj.rejects(
        storage.document(task_singleton_id, created_at), "does not retain historical content"
    )


async def _case_document_lifecycle(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    first_id = await storage.mint_id()
    second_id = await storage.mint_id()
    record = _doc_create(first_id, "singleton", {"kind": "session"})
    await storage.commit(
        [{"type": "document.create", "record": record, "content": {"kind": "base", "version": 1, "value": {"value": 1}}}]
    )
    await assert_obj.rejects(
        storage.commit(
            [
                {
                    "type": "document.create",
                    "record": dict(record, id=second_id),
                    "content": {"kind": "base", "version": 1, "value": {"value": 2}},
                },
                {
                    "type": "document.change",
                    "id": first_id,
                    "content": {"kind": "delta", "version": 1, "ops": []},
                },
            ]
        ),
        "already has a current incarnation",
    )
    assert_obj.deep_equal((await storage.document(first_id, "current")).value, {"value": 1})
    assert_obj.strict_equal(await storage.document(second_id, "current"), None)

    empty_id = await storage.mint_id()
    empty_at = await storage.commit(
        [
            {
                "type": "document.create",
                "record": _doc_create(
                    empty_id,
                    "singleton",
                    {"kind": "conversation", "conversation_id": root_id},
                    key="empty",
                    history="rewindable",
                    fork="initial",
                ),
                "content": {"kind": "base", "version": 1, "value": {}},
            },
            {"type": "document.retire", "id": empty_id},
        ]
    )
    assert_obj.strict_equal(await storage.document(empty_id, "current"), None)
    assert_obj.strict_equal(await storage.document(empty_id, empty_at), None)
    assert_obj.strict_equal(
        await storage.find_document(
            {
                "kind": "singleton",
                "scope": {"kind": "conversation", "conversation_id": root_id},
                "key": "empty",
            },
            empty_at,
        ),
        None,
    )


async def _case_document_rollback(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    task_id = await storage.mint_id()
    submission_id = await storage.mint_id()
    document_id = await storage.mint_id()
    task = _pending_task(task_id, root_id)
    submission = _submission(submission_id, root_id, request_id="atomic")
    record = _doc_create(document_id, "atomic", {"kind": "session"})
    baseline_seq = await storage.commit(
        [
            write_task(task),
            write_submission(submission),
            {
                "type": "document.create",
                "record": record,
                "content": {"kind": "base", "version": 1, "value": {"count": 1}},
            },
        ]
    )

    entry_id = await storage.mint_id()
    conflicting_document_id = await storage.mint_id()
    await assert_obj.rejects(
        storage.commit(
            [
                write_task(
                    TaskRecord(
                        id=task.id,
                        conversation_id=task.conversation_id,
                        kind=task.kind,
                        version=task.version,
                        input=task.input,
                        state=TaskState(status="running", checkpoint={"phase": "effect"}),
                    )
                ),
                write_submission(
                    SubmissionRecord(
                        id=submission.id,
                        conversation_id=submission.conversation_id,
                        type=submission.type,
                        status="unanswered",
                        request_id=submission.request_id,
                        reason="failed",
                    )
                ),
                write_entry(_entry(entry_id, root_id, "transient")),
                {
                    "type": "document.create",
                    "record": dict(record, id=conflicting_document_id),
                    "content": {"kind": "base", "version": 1, "value": {"count": 2}},
                },
            ]
        ),
        "already has a current incarnation",
    )

    assert_obj.deep_equal(await storage.task(task_id), task)
    assert_obj.deep_equal((await storage.scan_tasks({"status": "pending"}, 10)).items, [task])
    assert_obj.deep_equal(await storage.submission_by_request(root_id, "atomic"), submission)
    assert_obj.strict_equal(await storage.entry_global(entry_id), None)
    assert_obj.strict_equal(await storage.document(conflicting_document_id, "current"), None)
    assert_obj.strict_equal(
        (await storage.find_document({"kind": "atomic", "scope": {"kind": "session"}})).id, document_id
    )
    after_rollback_seq = await storage.commit(
        [
            {
                "type": "document.change",
                "id": document_id,
                "content": {"kind": "delta", "version": 1, "ops": [["s", ["count"], 3]]},
            }
        ]
    )
    assert_obj.greater_than(after_rollback_seq, baseline_seq)


async def _case_string_identities(storage) -> None:
    # Python backends store UTF-8 text: astral characters stand in for the TS lone-surrogate identities.
    assert_obj = _A
    root_id = await _create_root(storage)
    first = "𝄞"
    second = "😀"
    first_task_id = await storage.mint_id()
    second_task_id = await storage.mint_id()
    first_submission_id = await storage.mint_id()
    second_submission_id = await storage.mint_id()
    first_kind_document_id = await storage.mint_id()
    second_kind_document_id = await storage.mint_id()
    first_key_document_id = await storage.mint_id()
    second_key_document_id = await storage.mint_id()
    await storage.commit(
        [
            write_task(_replace(_pending_task(first_task_id, root_id), kind=first)),
            write_task(_replace(_pending_task(second_task_id, root_id), kind=second)),
            write_submission(_submission(first_submission_id, root_id, request_id=first)),
            write_submission(_submission(second_submission_id, root_id, request_id=second)),
            {
                "type": "document.create",
                "record": _doc_create(first_kind_document_id, first, {"kind": "session"}),
                "content": {"kind": "base", "version": 1, "value": {"identity": "first kind"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(second_kind_document_id, second, {"kind": "session"}),
                "content": {"kind": "base", "version": 1, "value": {"identity": "second kind"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(first_key_document_id, "family", {"kind": "session"}, key=first),
                "content": {"kind": "base", "version": 1, "value": {"identity": "first key"}},
            },
            {
                "type": "document.create",
                "record": _doc_create(second_key_document_id, "family", {"kind": "session"}, key=second),
                "content": {"kind": "base", "version": 1, "value": {"identity": "second key"}},
            },
        ]
    )

    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_tasks({"kind": first}, 10)).items], [first_task_id]
    )
    assert_obj.deep_equal(
        [item.id for item in (await storage.scan_tasks({"kind": second}, 10)).items], [second_task_id]
    )
    assert_obj.strict_equal((await storage.task(first_task_id)).kind, first)
    assert_obj.strict_equal((await storage.task(second_task_id)).kind, second)
    assert_obj.strict_equal((await storage.submission_by_request(root_id, first)).request_id, first)
    assert_obj.strict_equal((await storage.submission_by_request(root_id, first)).id, first_submission_id)
    assert_obj.strict_equal((await storage.submission_by_request(root_id, second)).id, second_submission_id)
    assert_obj.strict_equal(
        (await storage.find_document({"kind": first, "scope": {"kind": "session"}})).id,
        first_kind_document_id,
    )
    assert_obj.strict_equal(
        (await storage.find_document({"kind": second, "scope": {"kind": "session"}})).id,
        second_kind_document_id,
    )
    assert_obj.strict_equal(
        (await storage.find_document({"kind": "family", "key": first, "scope": {"kind": "session"}})).id,
        first_key_document_id,
    )
    assert_obj.strict_equal(
        (await storage.find_document({"kind": "family", "key": second, "scope": {"kind": "session"}})).id,
        second_key_document_id,
    )
    assert_obj.deep_equal(
        [
            item.id
            for item in (
                await storage.scan_documents({"scope": {"kind": "session"}, "at": "current", "kind": first}, 10)
            ).items
        ],
        [first_kind_document_id],
    )


async def _case_id_namespace(storage) -> None:
    assert_obj = _A
    root_id = await _create_root(storage)
    explicit_entry_id = id_from_number(100)
    await storage.commit([write_entry(_entry(explicit_entry_id, root_id))])
    assert_obj.strict_equal(await storage.mint_id(), 101)
    await assert_obj.rejects(
        storage.commit([write_task(_pending_task(explicit_entry_id, root_id))]),
        f"ID {explicit_entry_id} already belongs to entry",
    )

    await storage.commit([write_entry(_entry(id_from_number(MAX_SAFE_ID), root_id, "last-id"))])
    await assert_obj.rejects(storage.mint_id(), "ID space is exhausted")
    await assert_obj.rejects(storage.mint_id(), "ID space is exhausted")


async def _case_after_close(storage) -> None:
    assert_obj = _A
    await _create_root(storage)
    await storage.close()
    await assert_obj.rejects(storage.conversation(ROOT_CONVERSATION_ID), "closed")
    await assert_obj.rejects(storage.commit([]), "closed")
    await assert_obj.rejects(storage.mint_id(), "closed")


_A = ConformanceAssertions()
