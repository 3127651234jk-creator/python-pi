"""Regression tests for durable pagination and dependency settlement."""

import asyncio

from pi.ai.models import resolve_model
from pi.durable.harness import Harness
from pi.durable.session import create_session
from pi.durable.storage import MemoryStorage
from pi.durable.types import ConversationRecord, EntryRecord, TaskOutcome, TaskRecord, TaskState, write_conversation, write_entry, write_task


def task(tid, state):
    return TaskRecord(id=tid, conversation_id=1, kind="custom", version=1, input={}, state=state)


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
