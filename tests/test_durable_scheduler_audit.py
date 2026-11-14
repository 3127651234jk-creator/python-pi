"""Regression tests for durable pagination and dependency settlement."""

import asyncio

import pytest

from pi.ai.models import resolve_model
from pi.durable.harness import Harness, TaskRuntime
from pi.durable.session import create_session
from pi.durable.storage import MemoryStorage
from pi.durable.types import ConversationRecord, EntryRecord, TaskOutcome, TaskRecord, TaskState, write_conversation, write_entry, write_task


def task(tid, state):
    return TaskRecord(id=tid, conversation_id=1, kind="custom", version=1, input={}, state=state)


@pytest.mark.parametrize("status", ["terminal", "completing"])
def test_runtime_rejects_missing_outcome_atomically(status):
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([write_task(task(2, TaskState(status="running", checkpoint={"phase": "run"})))])
        runtime = TaskRuntime(harness, 2, asyncio.Event())

        async def change(tx, current):
            tx._writes.append(write_task(task(3, TaskState(status="pending"))))
            tx._mark_table_write()
            return TaskState(status=status)

        with pytest.raises(ValueError, match="requires an outcome"):
            await runtime.commit(change)
        assert (await storage.task(2)).state.status == "running"
        assert await storage.task(3) is None

    asyncio.run(scenario())


def test_fail_fast_handles_persisted_terminal_without_outcome():
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([
            write_task(task(2, TaskState(status="terminal"))),
            write_task(task(3, TaskState(status="running", checkpoint={"phase": "run"}))),
            write_task(task(4, TaskState(status="waiting", on=[2, 3], policy="failFast"))),
            write_task(task(5, TaskState(status="waiting", on=[2], policy="allSettled",
                                         checkpoint={"phase": "run"}))),
        ])
        assert await harness._resume_waiting_tasks()
        joined = await storage.task(4)
        assert joined.state.status == "terminal"
        assert joined.state.outcome.status == "aborted"
        assert joined.state.outcome.reason == "join failed: task 2"
        assert (await storage.task(5)).state.status == "pending"
        assert not await harness._resume_waiting_tasks()

    asyncio.run(scenario())


def test_transcript_and_task_listing_read_all_pages():
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([write_conversation(ConversationRecord(id=1)),
                              *[write_entry(EntryRecord(id=i, conversation_id=1, kind="message"))
                                for i in range(2, 1003)]])
        entries = await harness.transcript(1)
        assert [entry.id for entry in entries] == list(range(2, 1003))
        await storage.commit([write_task(task(i, TaskState(status="terminal", outcome=TaskOutcome(status="completed"))))
                              for i in range(1003, 2004)])
        assert len(await harness.list_tasks(1)) == 1001

    asyncio.run(scenario())


def test_crash_recovery_reads_more_than_one_page():
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([write_task(task(i, TaskState(status="running", checkpoint={"phase": "run"})))
                              for i in range(2, 103)])
        await harness._recover_running_tasks()
        page = await storage.scan_tasks({"status": "running"}, 200)
        assert not page.items

    asyncio.run(scenario())


def test_multiple_ready_waiters_resume_in_one_transaction():
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([
            write_task(task(2, TaskState(status="terminal", outcome=TaskOutcome(status="completed")))),
            *[write_task(task(i, TaskState(status="waiting", on=[2], policy="allSettled",
                                          checkpoint={"phase": "run"}))) for i in (3, 4)],
        ])
        assert await harness._resume_waiting_tasks()
        assert (await storage.task(3)).state.status == "pending"
        assert (await storage.task(4)).state.status == "pending"

    asyncio.run(scenario())


def test_ready_waiter_beyond_blocked_first_page_resumes():
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([
            write_task(task(2, TaskState(status="running", checkpoint={"phase": "run"}))),
            *[write_task(task(i, TaskState(status="waiting", on=[2], policy="allSettled",
                                          checkpoint={"phase": "run"}))) for i in range(3, 13)],
            write_task(task(13, TaskState(status="waiting", on=[], policy="allSettled",
                                         checkpoint={"phase": "run"}))),
        ])
        assert await harness._resume_waiting_tasks()
        assert (await storage.task(13)).state.status == "pending"

    asyncio.run(scenario())


def test_fail_fast_does_not_wait_for_other_running_dependencies():
    async def scenario():
        storage = MemoryStorage()
        harness = Harness(create_session(storage), resolve_model("faux"))
        await storage.commit([
            write_task(task(2, TaskState(status="running", checkpoint={"phase": "run"}))),
            write_task(task(3, TaskState(status="terminal", outcome=TaskOutcome(status="failed")))),
            write_task(task(4, TaskState(status="waiting", on=[2, 3], policy="failFast",
                                        checkpoint={"phase": "run"}))),
        ])
        assert await harness._resume_waiting_tasks()
        assert (await storage.task(4)).state.outcome.status == "aborted"

    asyncio.run(scenario())
