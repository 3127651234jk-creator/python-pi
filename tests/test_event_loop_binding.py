"""Regression tests for Python 3.9 asyncio loop binding.

On 3.9 asyncio primitives capture the current loop at construction. Objects
built by synchronous APIs (``create_session``, ``RpcClient()``, ``AbortSignal``,
``EventStream``, ``Harness``) must therefore not create primitives eagerly, or
awaiting them inside ``asyncio.run()`` fails with "got Future attached to a
different loop" (or construction itself raises after a prior run).
"""

import asyncio

from pi.ai.models import resolve_model
from pi.ai.providers import vertex
from pi.coding_agent.rpc_client import RpcClient
from pi.durable.harness import Harness
from pi.durable.session import create_session
from pi.durable.storage import MemoryStorage
from pi.util import AbortSignal, EventStream


def test_abort_signal_built_before_loop_waits_in_new_loop():
    signal = AbortSignal()

    async def scenario():
        waiter = asyncio.create_task(signal.wait())
        await asyncio.sleep(0)
        signal.abort()
        await asyncio.wait_for(waiter, 1)
        assert signal.aborted

    asyncio.run(scenario())


def test_event_stream_built_before_loop_replays_buffered_events():
    stream = EventStream(lambda event: event == "terminal", lambda event: "result")
    stream.push("first")
    stream.push("terminal")

    async def scenario():
        assert [event async for event in stream] == ["first", "terminal"]
        assert await stream.result() == "result"

    asyncio.run(scenario())


def test_rpc_client_built_before_loop_contends_its_lock_inside_new_loop():
    client = RpcClient()

    async def scenario():
        client.events.put_nowait({"type": "ready"})
        assert await client.events.get() == {"type": "ready"}
        await client.lock.acquire()
        waiter = asyncio.create_task(client.lock.acquire())
        await asyncio.sleep(0)
        client.lock.release()
        await asyncio.wait_for(waiter, 1)
        client.lock.release()

    asyncio.run(scenario())


def test_session_created_before_loop_serializes_commits():
    session = create_session(MemoryStorage())

    async def change(tx):
        await asyncio.sleep(0)
        return await tx.create_conversation({"kind": "ownerless"})

    async def scenario():
        first, second = await asyncio.gather(session.commit(change), session.commit(change))
        assert first.id != second.id
        await session.close()

    asyncio.run(scenario())


def test_harness_stop_event_waits_inside_consumer_loop():
    harness = Harness(create_session(MemoryStorage()), resolve_model("faux"))

    async def scenario():
        waiter = asyncio.create_task(harness._stop.wait())
        await asyncio.sleep(0)
        harness._stop.set()
        await asyncio.wait_for(waiter, 1)

    asyncio.run(scenario())


def test_vertex_token_lock_is_per_event_loop():
    locks = []

    async def scenario(hold):
        lock = vertex._token_lock()
        locks.append(lock)
        await lock.acquire()
        if not hold:
            lock.release()

    asyncio.run(scenario(True))
    asyncio.run(scenario(False))
    assert locks[0] is not locks[1]
