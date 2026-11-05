"""Storage conformance suite runs (port of testing coverage to the Python backends).

The runner-independent cases in ``pi.durable.testing`` execute against each
backend; every case runs as its own pytest test with a fresh storage instance.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.storage import JsonlStorage, MemoryStorage, SqliteStorage
from pi.durable.testing.types import ConformanceAssertions, StorageConformanceOptions

BACKENDS = ["memory", "jsonl", "sqlite"]


def make_storage(backend: str, tmp_path):
    if backend == "memory":
        return MemoryStorage()
    if backend == "jsonl":
        return JsonlStorage(str(tmp_path / f"{backend}.jsonl"))
    return SqliteStorage(str(tmp_path / f"{backend}.db"))


def _case_names():
    options = StorageConformanceOptions(assertions=ConformanceAssertions(), with_storage=None)
    from pi.durable.testing import create_storage_conformance

    return [case.name for case in create_storage_conformance(options)]


CASE_NAMES = _case_names()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_storage_conformance(backend, case_name, tmp_path):
    async def main():
        storage = make_storage(backend, tmp_path)
        try:

            async def with_storage(use):
                await use(storage)

            from pi.durable.testing import create_storage_conformance

            options = StorageConformanceOptions(assertions=ConformanceAssertions(), with_storage=with_storage)
            cases = create_storage_conformance(options)
            target = next(candidate for candidate in cases if candidate.name == case_name)
            await target.run()
        finally:
            await storage.close()

    asyncio.run(main())
