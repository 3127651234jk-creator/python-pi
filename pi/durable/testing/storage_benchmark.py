"""Storage benchmarks (port of ``testing/storage-benchmark.ts``).

Seed deterministic representative data through only the public ``Storage``
contract, then measure read and write operations against it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..ids import ROOT_CONVERSATION_ID
from ..types import (
    ConversationRecord,
    EntryRecord,
    SubmissionRecord,
    TaskRecord,
    TaskState,
    write_conversation,
    write_document_create,
    write_document_change,
    write_entry,
    write_submission,
    write_task,
)

REPLAY_TAILS = [0, 16, 128, 1_024]
HISTORY_SEGMENT_LENGTH = 128
FORK_DEPTH = 8
ENTRIES_PER_FORK = 32
BATCH_SIZE = 100


@dataclass
class StorageBenchmarkScale:
    name: str
    entry_count: int
    task_count: int
    document_count: int


STORAGE_MEMORY_SCALES = [
    StorageBenchmarkScale(name="1k", entry_count=1_000, task_count=200, document_count=200),
    StorageBenchmarkScale(name="10k", entry_count=10_000, task_count=2_000, document_count=2_000),
]

TIMING_SCALE = StorageBenchmarkScale(name="timing", entry_count=1_000, task_count=300, document_count=300)


def storage_benchmark_primary_record_count(scale: StorageBenchmarkScale) -> int:
    return (
        1
        + scale.entry_count
        + scale.task_count
        + scale.document_count
        + len(REPLAY_TAILS)
        + 1
        + FORK_DEPTH * (1 + ENTRIES_PER_FORK)
    )


@dataclass
class StorageBenchmarkDataset:
    first_entry_id: int
    filtered_task_count: int
    exact_document_id: int
    exact_document_key: str
    replay_document_ids: Dict[int, int]
    historical_document_id: int
    ancient_at: int
    recent_at: int
    deepest_conversation_id: int
    ancestor_head_entry_id: int


def _task(task_id: int, index: int) -> TaskRecord:
    statuses = ["pending", "running", "terminal"]
    status = statuses[index % len(statuses)]
    common = dict(
        id=task_id,
        conversation_id=ROOT_CONVERSATION_ID,
        kind="benchmark.filtered" if index % 4 == 0 else "benchmark.other",
        version=1,
        input={"index": index},
        background=index % 5 == 0,
        abort_requested=index % 7 == 0,
    )
    if status == "terminal":
        return TaskRecord(
            **common,
            state=TaskState(status="terminal", outcome={"status": "completed", "result": {"index": index}}),
        )
    return TaskRecord(**common, state=TaskState(status=status, checkpoint={"index": index, "payload": "x" * 64}))


async def seed_storage_benchmark(storage, scale: Optional[StorageBenchmarkScale] = None) -> StorageBenchmarkDataset:
    """Seed deterministic representative data through only the public ``Storage`` contract."""
    scale = scale or TIMING_SCALE
    await storage.commit([write_conversation(ConversationRecord(id=ROOT_CONVERSATION_ID))])

    first_entry_id: Optional[int] = None
    for start in range(0, scale.entry_count, BATCH_SIZE):
        writes = []
        for index in range(start, min(start + BATCH_SIZE, scale.entry_count)):
            entry_id = await storage.mint_id()
            if index == 0:
                first_entry_id = entry_id
            writes.append(
                write_entry(
                    EntryRecord(
                        id=entry_id,
                        conversation_id=ROOT_CONVERSATION_ID,
                        kind="benchmark.entry",
                        head=entry_id if index == 0 else None,
                        data={"index": index, "text": f"entry-{index}-{'x' * 96}"},
                    )
                )
            )
        await storage.commit(writes)

    for start in range(0, scale.task_count, BATCH_SIZE):
        writes = [
            write_task(_task(await storage.mint_id(), index))
            for index in range(start, min(start + BATCH_SIZE, scale.task_count))
        ]
        await storage.commit(writes)

    exact_document_id: Optional[int] = None
    for start in range(0, scale.document_count, BATCH_SIZE):
        writes = []
        for index in range(start, min(start + BATCH_SIZE, scale.document_count)):
            document_id = await storage.mint_id()
            exact_document_id = document_id
            writes.append(
                write_document_create(
                    {
                        "id": document_id,
                        "kind": "benchmark.family",
                        "key": f"key-{index}",
                        "scope": {"kind": "session"},
                    },
                    1,
                    {"index": index, "text": "x" * 128},
                )
            )
        await storage.commit(writes)

    replay_entries = [(tail, await storage.mint_id()) for tail in REPLAY_TAILS]
    await storage.commit(
        [
            write_document_create(
                {
                    "id": document_id,
                    "kind": "benchmark.replay",
                    "key": str(tail),
                    "scope": {"kind": "conversation", "conversation_id": ROOT_CONVERSATION_ID},
                    "history": "rewindable",
                    "fork": "asOf",
                },
                1,
                {"count": 0, "text": "x" * 64},
            )
            for tail, document_id in replay_entries
        ]
    )
    for count in range(1, max(REPLAY_TAILS) + 1):
        await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": document_id,
                    "content": {"kind": "delta", "version": 1, "ops": [["s", ["count"], count]]},
                }
                for tail, document_id in replay_entries
                if count <= tail
            ]
        )

    historical_document_id = await storage.mint_id()
    historical_record = {
        "id": historical_document_id,
        "kind": "benchmark.history",
        "scope": {"kind": "conversation", "conversation_id": ROOT_CONVERSATION_ID},
        "history": "rewindable",
        "fork": "asOf",
    }
    await storage.commit(
        [
            {
                "type": "document.create",
                "record": historical_record,
                "content": {"kind": "base", "version": 1, "value": {"count": 0}},
            }
        ]
    )
    ancient_at: Optional[int] = None
    for count in range(1, HISTORY_SEGMENT_LENGTH + 1):
        ancient_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": historical_document_id,
                    "content": {"kind": "delta", "version": 1, "ops": [["s", ["count"], count]]},
                }
            ]
        )
    await storage.commit(
        [
            {
                "type": "document.change",
                "id": historical_document_id,
                "content": {"kind": "base", "version": 1, "value": {"count": HISTORY_SEGMENT_LENGTH}},
            }
        ]
    )
    if ancient_at is None:
        raise RuntimeError("Benchmark history seed produced no commits")
    recent_at = ancient_at
    for count in range(HISTORY_SEGMENT_LENGTH + 1, HISTORY_SEGMENT_LENGTH * 2 + 1):
        recent_at = await storage.commit(
            [
                {
                    "type": "document.change",
                    "id": historical_document_id,
                    "content": {"kind": "delta", "version": 1, "ops": [["s", ["count"], count]]},
                }
            ]
        )

    if first_entry_id is None:
        raise RuntimeError("Benchmark scale must create entries")
    parent_conversation_id = ROOT_CONVERSATION_ID
    parent_at = first_entry_id
    deepest_conversation_id = ROOT_CONVERSATION_ID
    for depth in range(FORK_DEPTH):
        conversation_id = await storage.mint_id()
        await storage.commit(
            [
                write_conversation(
                    ConversationRecord(
                        id=conversation_id, parent={"conversation_id": parent_conversation_id, "at": parent_at}
                    )
                )
            ]
        )
        ids = [await storage.mint_id() for _ in range(ENTRIES_PER_FORK)]
        await storage.commit(
            [
                write_entry(
                    EntryRecord(
                        id=entry_id,
                        conversation_id=conversation_id,
                        kind="benchmark.fork",
                        data={"depth": depth, "index": index},
                    )
                )
                for index, entry_id in enumerate(ids)
            ]
        )
        parent_conversation_id = conversation_id
        parent_at = ids[-1]
        deepest_conversation_id = conversation_id

    if exact_document_id is None:
        raise RuntimeError("Benchmark scale must create documents")
    return StorageBenchmarkDataset(
        first_entry_id=first_entry_id,
        filtered_task_count=min(50, -(-scale.task_count // 60)),
        exact_document_id=exact_document_id,
        exact_document_key=f"key-{scale.document_count - 1}",
        replay_document_ids={tail: document_id for tail, document_id in replay_entries},
        historical_document_id=historical_document_id,
        ancient_at=ancient_at,
        recent_at=recent_at,
        deepest_conversation_id=deepest_conversation_id,
        ancestor_head_entry_id=first_entry_id,
    )


@dataclass
class StorageReadBenchmark:
    """One read operation. ``run`` returns the value to compare with ``expected(dataset)``."""

    name: str
    run: Callable[[Any, StorageBenchmarkDataset], Awaitable[Any]]
    expected: Callable[[StorageBenchmarkDataset], Any]


def _read_benchmarks() -> List[StorageReadBenchmark]:
    async def exact_entry(storage, dataset) -> int:
        return (await storage.entry_global(dataset.first_entry_id))[0].id

    async def entry_page(storage, dataset) -> int:
        return len((await storage.scan_entries({"conversation_id": ROOT_CONVERSATION_ID}, 100)).items)

    async def filtered_tasks(storage, dataset) -> int:
        page = await storage.scan_tasks(
            {"kind": "benchmark.filtered", "status": "pending", "background": True}, 50
        )
        return len(page.items)

    async def exact_document(storage, dataset) -> int:
        found = await storage.find_document(
            {"kind": "benchmark.family", "key": dataset.exact_document_key, "scope": {"kind": "session"}},
            "current",
        )
        return -1 if found is None else found.id

    def replay_of(tail: int):
        async def run(storage, dataset) -> int:
            return (await storage.document(dataset.replay_document_ids[tail], "current")).value["count"]

        return run

    async def ancient(storage, dataset) -> int:
        return (await storage.document(dataset.historical_document_id, dataset.ancient_at)).value["count"]

    async def recent(storage, dataset) -> int:
        return (await storage.document(dataset.historical_document_id, dataset.recent_at)).value["count"]

    async def fork_scan(storage, dataset) -> int:
        page = await storage.scan_entries({"conversation_id": dataset.deepest_conversation_id}, 100)
        return len(page.items)

    async def head_lookup(storage, dataset) -> int:
        marker = await storage.find_latest_head_marker(dataset.deepest_conversation_id)
        return -1 if marker is None else marker.id

    benchmarks = [
        StorageReadBenchmark("exact entry lookup", exact_entry, lambda d: d.first_entry_id),
        StorageReadBenchmark("entry page scan (100)", entry_page, lambda d: 100),
        StorageReadBenchmark("filtered task scan (50)", filtered_tasks, lambda d: d.filtered_task_count),
        StorageReadBenchmark("exact document address among many", exact_document, lambda d: d.exact_document_id),
    ]
    benchmarks.extend(
        StorageReadBenchmark(f"document replay tail ({tail})", replay_of(tail), lambda d, tail=tail: tail)
        for tail in REPLAY_TAILS
    )
    benchmarks.extend(
        [
            StorageReadBenchmark("ancient historical read before newer base", ancient, lambda d: HISTORY_SEGMENT_LENGTH),
            StorageReadBenchmark("recent historical read after newer base", recent, lambda d: HISTORY_SEGMENT_LENGTH * 2),
            StorageReadBenchmark("fork-depth history scan (100)", fork_scan, lambda d: 100),
            StorageReadBenchmark("fork-depth head lookup", head_lookup, lambda d: d.ancestor_head_entry_id),
        ]
    )
    return benchmarks


STORAGE_READ_BENCHMARKS = _read_benchmarks()


@dataclass
class StorageWriteBenchmark:
    name: str
    expected: int
    run: Callable[[Any], Awaitable[int]]


async def seed_storage_write_benchmark(storage) -> None:
    """Seed the common state expected by every write benchmark sample."""
    await storage.commit([write_conversation(ConversationRecord(id=ROOT_CONVERSATION_ID))])
    writes = []
    for index in range(100):
        writes.append(
            write_entry(
                EntryRecord(
                    id=await storage.mint_id(),
                    conversation_id=ROOT_CONVERSATION_ID,
                    kind="benchmark.baseline",
                    data={"index": index},
                )
            )
        )
    await storage.commit(writes)


async def _write_one_entry(storage) -> int:
    entry_id = await storage.mint_id()
    await storage.commit(
        [
            write_entry(
                EntryRecord(
                    id=entry_id,
                    conversation_id=ROOT_CONVERSATION_ID,
                    kind="benchmark.write",
                    data={"text": "x" * 128},
                )
            )
        ]
    )
    return 1


async def _write_hundred_entries(storage) -> int:
    writes = []
    for index in range(100):
        writes.append(
            write_entry(
                EntryRecord(
                    id=await storage.mint_id(),
                    conversation_id=ROOT_CONVERSATION_ID,
                    kind="benchmark.write",
                    data={"index": index, "text": "x" * 128},
                )
            )
        )
    await storage.commit(writes)
    return len(writes)


async def _write_mixed(storage) -> int:
    entry_id = await storage.mint_id()
    task_id = await storage.mint_id()
    submission_id = await storage.mint_id()
    document_id = await storage.mint_id()
    writes = [
        write_entry(EntryRecord(id=entry_id, conversation_id=ROOT_CONVERSATION_ID, kind="benchmark.mixed")),
        write_task(_task(task_id, task_id)),
        write_submission(
            SubmissionRecord(
                id=submission_id,
                conversation_id=ROOT_CONVERSATION_ID,
                request_id=f"benchmark-{submission_id}",
                type="write",
                status="done",
                entry=entry_id,
            )
        ),
        write_document_create(
            {
                "id": document_id,
                "kind": "benchmark.mixed",
                "key": str(document_id),
                "scope": {"kind": "session"},
            },
            1,
            {"entryId": entry_id, "taskId": task_id},
        ),
    ]
    await storage.commit(writes)
    return len(writes)


STORAGE_WRITE_BENCHMARKS = [
    StorageWriteBenchmark("commit one entry", 1, _write_one_entry),
    StorageWriteBenchmark("commit 100 entries", 100, _write_hundred_entries),
    StorageWriteBenchmark("commit mixed entry/task/submission/document", 4, _write_mixed),
]


async def measure_read_benchmarks(storage, scale: Optional[StorageBenchmarkScale] = None, repeat: int = 3) -> Dict[str, float]:
    """Seed, verify each benchmark's expected result once, and report average milliseconds per run."""
    dataset = await seed_storage_benchmark(storage, scale)
    results: Dict[str, float] = {}
    for benchmark in STORAGE_READ_BENCHMARKS:
        value = await benchmark.run(storage, dataset)
        expected = benchmark.expected(dataset)
        if value != expected:
            raise AssertionError(f"{benchmark.name}: expected {expected!r}, got {value!r}")
        total = 0.0
        for _ in range(repeat):
            started = time.perf_counter()
            await benchmark.run(storage, dataset)
            total += time.perf_counter() - started
        results[benchmark.name] = total / repeat * 1000
    return results


async def measure_write_benchmarks(storage, repeat: int = 20) -> Dict[str, float]:
    """Seed the write baseline, verify counts, and report average milliseconds per commit."""
    await seed_storage_write_benchmark(storage)
    results: Dict[str, float] = {}
    for benchmark in STORAGE_WRITE_BENCHMARKS:
        total = 0.0
        for _ in range(repeat):
            started = time.perf_counter()
            count = await benchmark.run(storage)
            total += time.perf_counter() - started
            if count != benchmark.expected:
                raise AssertionError(f"{benchmark.name}: expected {benchmark.expected}, got {count}")
        results[benchmark.name] = total / repeat * 1000
    return results
