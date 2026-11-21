import asyncio

import pytest

from pi.ai.models import Model
from pi.durable.entries import TOOL_RESULT_ENTRY
from pi.durable.harness import Harness, TaskRuntime
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage
from pi.durable.types import TaskState
from pi.durable.worker import SessionEndpoint, SessionServices


@pytest.mark.parametrize("policy", ["allSettled", "failFast"])
@pytest.mark.parametrize("reopen", [False, True])
def test_waiting_abort_settles_submission_without_join(tmp_path, policy, reopen):
    async def scenario():
        path = str(tmp_path / "session.jsonl")
        session = create_session(JsonlStorage(path))
        model = Model("faux", "Faux", "faux", "faux", "")
        harness = Harness(session, model)
        cid = await harness.create_conversation()
        task_id = await harness.submit_input(cid, "hello")
        task = await harness._get_task(task_id)
        submission_id = task.input["submission_id"]
        await harness._reserve(task_id, harness.definitions[task.kind], task)
        runtime = TaskRuntime(harness, task_id, asyncio.Event())
        await runtime.commit(lambda tx, current: TaskState(
            status="waiting", checkpoint={"phase": "prepare"}, on=[999999], policy=policy))
        # Unsatisfied joins stay waiting until an abort is requested.
        assert await harness.run_pending_tasks() == 0
        assert (await harness._get_task(task_id)).state.status == "waiting"
        await harness.abort_task(task_id)
        if reopen:
            await session.close()
            session = create_session(JsonlStorage(path))
            harness = Harness(session, model)
        await asyncio.wait_for(harness.run_pending_tasks(), 2)
        record = await harness._get_task(task_id)
        assert record.state.status == "terminal"
        assert record.state.outcome.status == "aborted"
        submission = await session.storage.submission(submission_id)
        assert submission.status == "unanswered"
        assert await harness.run_pending_tasks() == 0
        await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("explicit_task_id", [False, True])
def test_worker_abort_settles_waiting_generation_and_its_tools(tmp_path, explicit_task_id):
    """The worker abort must find a waiting generation and cascade to its tools.

    A waiting generation is blocked on its tool round, so marking only it would
    leave the child pi.tool tasks running; they would later commit tool results
    for a turn that was already aborted.
    """

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        generation_id = await harness.submit_input(cid, "hello")
        generation = await harness._get_task(generation_id)
        await harness._reserve(generation_id, harness.definitions[generation.kind], generation)
        child_id = await harness.submit_tool_task(cid, "echo", {"text": "x"})
        runtime = TaskRuntime(harness, generation_id, asyncio.Event())
        await runtime.commit(lambda tx, current: TaskState(
            status="waiting", checkpoint={"phase": "after_tools", "tools": [child_id]},
            on=[child_id], policy="allSettled"))
        assert (await harness._get_task(generation_id)).state.status == "waiting"

        endpoint = SessionEndpoint(SessionServices(harness, cid, asyncio.Event()))
        args = [{"taskId": generation_id}] if explicit_task_id else []
        reply = await endpoint.invoke_service({"member": "abort", "args": args}, None, None)
        assert reply == {"aborted": True, "taskId": generation_id}
        # Every non-terminal task of the conversation is marked, not only the
        # generation that the reply reports.
        assert (await harness._get_task(generation_id)).abort_requested
        assert (await harness._get_task(child_id)).abort_requested

        await asyncio.wait_for(harness.run_pending_tasks(), 2)
        generation = await harness._get_task(generation_id)
        child = await harness._get_task(child_id)
        assert (generation.state.status, generation.state.outcome.status) == ("terminal", "aborted")
        assert (child.state.status, child.state.outcome.status) == ("terminal", "aborted")
        # The tool never ran, so no tool result was committed.
        assert TOOL_RESULT_ENTRY.kind not in [entry.kind for entry in await harness.transcript(cid)]

        # A terminal task is no longer abortable.
        reply = await endpoint.invoke_service(
            {"member": "abort", "args": [{"taskId": generation_id}]}, None, None)
        assert reply == {"aborted": False}
        await session.close()

    asyncio.run(scenario())
