"""Session kernel tests: transactions, read-after-write, forks, publication."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.documents import define_doc
from pi.durable.errors import ReadAfterWrite
from pi.durable.entries import USER_ENTRY
from pi.durable.session import create_session
from pi.durable.types import define_task
from pi.durable.storage import JsonlStorage, MemoryStorage


def test_commit_publishes_changes():
    async def scenario():
        session = create_session(MemoryStorage())
        publications = []
        session.subscribe_commits(publications.append)

        cid = (await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))).id
        record = await session.commit(lambda tx: tx.append_entry(cid, {"kind": USER_ENTRY.kind, "model": []}))

        assert len(publications) == 2
        assert publications[0].changes[0]["type"] == "conversation"
        assert publications[1].changes[0]["type"] == "entry"
        assert publications[1].changes[0]["value"].id == record.id

    asyncio.run(scenario())


def test_commit_listener_failure_does_not_fail_the_commit():
    async def scenario():
        loop = asyncio.get_running_loop()
        reported = []
        loop.set_exception_handler(lambda loop, context: reported.append(context))

        session = create_session(MemoryStorage())

        def broken(publication):
            raise RuntimeError("listener failed")

        session.subscribe_commits(broken)
        cid = (await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))).id

        assert cid
        assert reported and "listener" in reported[0]["message"]

    asyncio.run(scenario())


def test_read_after_write_rejected():
    async def scenario():
        session = create_session(MemoryStorage())

        async def change(tx):
            await tx.create_conversation({"kind": "ownerless"})
            with pytest.raises(ReadAfterWrite):
                await tx.create_conversation({"kind": "ownerless"})

        await session.commit(change)

    asyncio.run(scenario())


def test_failed_callback_writes_nothing():
    async def scenario():
        session = create_session(MemoryStorage())
        publications = []
        session.subscribe_commits(publications.append)

        async def change(tx):
            await tx.create_conversation({"kind": "ownerless"})
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await session.commit(change)

        # The storage batch never reached commit, so nothing is published.
        assert publications == []

    asyncio.run(scenario())


def test_fork_visibility():
    async def scenario():
        session = create_session(MemoryStorage())
        parent = (await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))).id
        e1 = (await session.commit(lambda tx: tx.append_entry(parent, {"kind": "pi.user", "model": []}))).id
        e2 = (await session.commit(lambda tx: tx.append_entry(parent, {"kind": "pi.user", "model": []}))).id

        child = (
            await session.commit(lambda tx: tx.fork_conversation(parent, e1, {"kind": "ownerless"}))
        ).id
        inherited = await session.commit(lambda tx: tx.append_entry(child, {"kind": "pi.reset", "head": "self"}))

        async def read_entries(tx):
            return await tx.scan_entries({"conversation_id": child}, 10)

        page = await session.commit(read_entries)
        ids = [e.id for e in page.items]
        assert ids[0] == inherited.id
        assert e1 in ids
        assert e2 not in ids

    asyncio.run(scenario())


def test_tasks_and_submissions_roundtrip():
    async def scenario():
        session = create_session(MemoryStorage())
        definition = define_task(
            name="pi.test",
            version=1,
            initial=lambda input: {"phase": "start"},
            phases={"start": None},
        )

        async def change(tx):
            cid = (await tx.create_conversation({"kind": "ownerless"})).id
            task = await tx.create_task(definition, {"n": 1}, {"ownership": {"kind": "conversation"}, "conversation_id": cid})
            submission = await tx.create_submission(
                {"conversation_id": cid, "type": "input", "status": "queued", "request_id": "r1"}
            )
            entry = await tx.append_entry(cid, {"kind": "pi.user", "model": []})
            await tx.place_submission(submission.id, entry.id)
            return cid, task, submission, entry

        cid, task, submission, entry = await session.commit(change)
        assert task.state.status == "pending"
        assert submission.status == "placed"

        async def read_back(tx):
            loaded_task = await tx.task(task.id)
            loaded_sub = await tx.submission_by_request(cid, "r1")
            return loaded_task, loaded_sub

        loaded_task, loaded_sub = await session.commit(read_back)
        assert loaded_task is not None and loaded_task.kind == "pi.test"
        assert loaded_sub is not None and loaded_sub.status == "placed"

    asyncio.run(scenario())


def test_documents_snapshot_across_sessions():
    async def scenario():
        storage = MemoryStorage()
        session = create_session(storage)
        counter = define_doc(kind="pi.counter", version=2, scope="session", initial=lambda seed: {"count": 0})

        async def change(tx):
            draft = await tx.doc(counter)
            draft.value["count"] = 5
            draft.dirty = True
            return draft.value["count"]

        assert await session.commit(change) == 5
        assert await session.snapshot(counter) == {"count": 5}

        # A second session over the same storage observes committed state.
        session2 = create_session(storage)
        assert await session2.snapshot(counter) == {"count": 5}

    asyncio.run(scenario())


def test_session_serializes_commits():
    async def scenario():
        session = create_session(MemoryStorage())
        order = []

        def make_change(name):
            async def change(tx):
                order.append(f"{name}:start")
                await asyncio.sleep(0.01)
                order.append(f"{name}:end")
                await tx.create_conversation({"kind": "ownerless"})

            return change

        await asyncio.gather(session.commit(make_change("a")), session.commit(make_change("b")))
        # Each commit body runs atomically on the line: start/end never interleave.
        assert order == ["a:start", "a:end", "b:start", "b:end"] or order == [
            "b:start",
            "b:end",
            "a:start",
            "a:end",
        ]

    asyncio.run(scenario())


def test_reopen_continues_sequence():
    async def scenario():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp-durable-session.jsonl")
        if os.path.exists(path):
            os.remove(path)
        try:
            first = create_session(JsonlStorage(path))
            cid = (await first.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))).id
            await first.close()

            second = create_session(JsonlStorage(path))
            entry = await second.commit(lambda tx: tx.append_entry(cid, {"kind": "pi.user", "model": []}))
            assert entry.conversation_id == cid
            await second.close()
        finally:
            if os.path.exists(path):
                os.remove(path)

    asyncio.run(scenario())
