import asyncio

import pytest

from pi.ai.models import Model
from pi.durable.harness import Harness, TaskRuntime
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage
from pi.durable.types import TaskState


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
