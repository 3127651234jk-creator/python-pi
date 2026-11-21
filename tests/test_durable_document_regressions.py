"""Document transaction and snapshot isolation regressions."""

import asyncio

import pytest

from pi.durable.documents import define_doc, define_doc_family, resolve_address
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage, MemoryStorage, SqliteStorage


@pytest.fixture(params=["memory", "jsonl", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        storage = MemoryStorage()
    elif request.param == "jsonl":
        storage = JsonlStorage(str(tmp_path / "store.jsonl"))
    else:
        storage = SqliteStorage(str(tmp_path / "store.sqlite"))
    return storage


async def save(session, definition, *args):
    async def change(tx):
        draft = await tx.doc(definition, *args)
        draft.dirty = True
    await session.commit(change)


def test_snapshot_and_failed_draft_do_not_mutate_cached_state(storage):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.isolation", 1, "session", lambda seed: {"nested": {"n": 1}})
        try:
            await save(session, definition)
            snapshot = await session.snapshot(definition)
            snapshot["nested"]["n"] = 10
            assert await session.snapshot(definition) == {"nested": {"n": 1}}

            async def fail(tx):
                draft = await tx.doc(definition)
                draft.value["nested"]["n"] = 20
                draft.dirty = True
                raise RuntimeError("rollback")

            with pytest.raises(RuntimeError, match="rollback"):
                await session.commit(fail)
            assert await session.snapshot(definition) == {"nested": {"n": 1}}
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("existing", [False, True])
def test_repeated_doc_returns_same_draft(storage, existing):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.reuse", 1, "session", lambda seed: {"n": 0})
        try:
            if existing:
                await save(session, definition)

            async def change(tx):
                first = await tx.doc(definition)
                first.value["n"] += 1
                first.dirty = True
                second = await tx.doc(definition)
                assert second is first
                second.value["n"] += 1

            await session.commit(change)
            assert await session.snapshot(definition) == {"n": 2}
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["session", "conversation"])
def test_family_address_is_preserved(storage, scope):
    async def scenario():
        session = create_session(storage)
        definition = define_doc_family("audit.family", 1, scope, lambda seed: {"seed": seed})
        try:
            owners = [[]]
            if scope == "conversation":
                owners = []
                for _ in range(2):
                    owner = await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))
                    owners.append([owner.id])
            for index, owner in enumerate(owners):
                for key in ("a", "b"):
                    await save(session, definition, *owner, key, f"{index}:{key}")
            for index, owner in enumerate(owners):
                for key in ("a", "b"):
                    assert await session.snapshot(definition, *owner, key) == {"seed": f"{index}:{key}"}
        finally:
            await session.close()
    asyncio.run(scenario())


def test_retire_without_loading_draft_can_publish(storage):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.retire", 1, "session", lambda seed: {"n": 1})
        publications = []
        session.subscribe_commits(publications.append)
        try:
            await save(session, definition)
            await session.commit(lambda tx: tx.retire_doc(definition))
            assert await session.snapshot(definition) is None
            assert publications[-1].changes[0]["version"] is None
            # Retirement must leave the session usable for subsequent commits.
            await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("existing", [False, True])
def test_retiring_dirty_draft_does_not_recreate_document(storage, existing):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.retire-dirty", 1, "session", lambda seed: {"n": 1})
        try:
            if existing:
                await save(session, definition)

            async def change(tx):
                draft = await tx.doc(definition)
                draft.value["n"] = 2
                draft.dirty = True
                await tx.retire_doc(definition)

            await session.commit(change)
            assert await session.snapshot(definition) is None
        finally:
            await session.close()
    asyncio.run(scenario())


def test_snapshot_rejects_newer_version_and_migration_is_detached(storage):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.version", 2, "session", lambda seed: {"nested": {"n": 1}})
        try:
            await save(session, definition)
            old = define_doc("audit.version", 1, "session", lambda seed: {})
            with pytest.raises(ValueError, match="newer than definition"):
                await session.snapshot(old)

            def migrate(value, version):
                value["nested"]["n"] = version + 1
                return value

            new = define_doc("audit.version", 3, "session", lambda seed: {}, migrate=migrate)
            assert await session.snapshot(new) == {"nested": {"n": 3}}
            assert await session.snapshot(definition) == {"nested": {"n": 1}}
        finally:
            await session.close()
    asyncio.run(scenario())


def test_document_retention_evicts_unused_documents(storage):
    async def scenario():
        definition = define_doc("audit.retention", 1, "session", lambda seed: {"n": 0})
        clock = {"now": 1_000_000}
        session = create_session(storage, now=lambda: clock["now"])
        try:
            await save(session, definition)
            await session.snapshot(definition)
            address_id, _address, _next = resolve_address(definition, [])
            cached = session._document_cache.get(address_id)
            assert cached is not None

            # Within the retention window the cached incarnation is reused.
            clock["now"] += 1
            await session.snapshot(definition)
            assert session._document_cache.get(address_id) is cached

            # Past the window the document is evicted and reloads fresh.
            clock["now"] += 600_000
            await session.snapshot(definition)
            reloaded = session._document_cache.get(address_id)
            assert reloaded is not None and reloaded is not cached
        finally:
            await session.close()

    asyncio.run(scenario())


def test_document_retention_zero_evicts_after_each_use(storage):
    async def scenario():
        definition = define_doc("audit.retention0", 1, "session", lambda seed: {"n": 0})
        session = create_session(storage, document_retention_ms=0)
        try:
            await save(session, definition)
            await session.snapshot(definition)
            address_id, _address, _next = resolve_address(definition, [])
            first = session._document_cache.get(address_id)
            assert first is not None
            # Retention 0 drops every cached document on the next access.
            assert await session.snapshot(definition) == {"n": 0}
            assert session._document_cache.get(address_id) is not first
        finally:
            await session.close()

    asyncio.run(scenario())
