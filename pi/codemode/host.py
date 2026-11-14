"""Codemode host: runs one user script in an isolated subprocess.

TS counterpart: ``packages/codemode/src/runtime/host.ts`` plus ``source.ts``
and ``identifier.ts``. The tool adapter (``pi.codemode.tool``) resolves the
script options and hands over source, tool metadata, the initial store, a
timeout, the signal, and two callbacks; this module owns everything
process-shaped: spawning ``python -I`` per script run, the UTF-8 JSON-line
protocol, concurrent pending calls with out-of-order replies, the deadline,
abort, cancellation of unfinished calls on normal completion, and process
reaping. Every worker payload is validated before use: a malformed one fails
the run as a sandbox error instead of leaking bad data into the host or
leaving it unsettled. Call records and their publish timing stay here so a
single owner produces the snapshots; the executor callback receives the live
record so the runner-assigned call id is backfilled where the call runs, not
swapped at the end.

This module does not import agent tool or session types; the outcome dataclass
is an internal boundary, not a public SDK type.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..util import AbortSignal
from .worker_source import WORKER_SOURCE

ARGS_PREVIEW_CHARS = 200
ERROR_PREVIEW_CHARS = 500

# Outcome statuses: a normal return ("ok") or one of the four failure classes
# the tool adapter renders into distinct result texts.
STATUS_OK = "ok"
STATUS_SCRIPT_ERROR = "script_error"
STATUS_TIMEOUT = "timeout"
STATUS_ABORTED = "aborted"
STATUS_CRASHED = "crashed"

# Executes one script call: (name, args, signal, record) -> script-visible
# value; raises to fail the call. Defined here so the tool adapter and the
# host agree on the record contract without sharing agent types.
CallExecutor = Callable[[str, dict, Optional[AbortSignal], "CallRecord"], Awaitable[Any]]
# Receives the calls snapshot at the original publish points.
UpdateListener = Callable[[dict], Union[None, Awaitable[None]]]


@dataclass
class CallRecord:
    """One script tool call as published in snapshots.

    The host creates it and finalizes status/duration; the executor mutates it
    in place (at least ``id``, backfilled with the id the runner assigned).
    """

    id: str
    name: str
    args: str
    status: str = "running"
    duration_ms: Optional[int] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class CodemodeRuntime:
    """Worker program and scratch directory, prepared once per tool."""

    workdir: Path
    worker_path: Path


def prepare_runtime(prefix: str = "pi-codemode-") -> CodemodeRuntime:
    """Create the scratch dir and write the worker program; called by the tool
    factory, so per-script runs reuse the same worker file and workdir."""

    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    worker_path = workdir / "codemode_worker.py"
    worker_path.write_text(WORKER_SOURCE, encoding="utf-8")
    return CodemodeRuntime(workdir=workdir, worker_path=worker_path)


@dataclass
class CodemodeOutcome:
    """Internal result of one script run; the tool adapter renders it."""

    status: str
    value: Any = None
    output_items: List[dict] = field(default_factory=list)
    calls: List[dict] = field(default_factory=list)
    writes: Dict[str, Any] = field(default_factory=lambda: {"set": {}, "delete": []})
    wall_time: float = 0.0
    # STATUS_SCRIPT_ERROR: the validated error payload ({name, message,
    # stack?}); name and message are always strings, stack may be absent.
    error: Optional[dict] = None
    # STATUS_CRASHED: the worker's exit code, or "unknown" when it has none.
    exit_code: Union[int, str, None] = None


def to_codemode_identifier(name: str) -> str:
    """The script identifier for a tool name: invalid characters become ``_``."""

    identifier = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not identifier or not re.match(r"[A-Za-z_]", identifier[0]):
        identifier = "_" + identifier
    return identifier


def to_codemode_file_stem(call_id: str) -> str:
    """Filesystem-safe file stem for scratch names derived from a call id.

    Call ids may carry path separators (nested runners use ``parent/1``-style
    ids), which would otherwise escape the workdir and crash the run; safe ids
    pass through unchanged.
    """

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", call_id)
    return stem or "call"


def parse_codemode_source(code: str) -> tuple:
    """Split the optional ``# @options:`` first line from the script source."""

    stripped = code.lstrip()
    if not stripped.startswith("#@options:") and not stripped.startswith("# @options:"):
        return {}, code
    first = stripped.splitlines()[0]
    payload = first.split(":", 1)[1].strip()
    try:
        options = json.loads(payload)
    except ValueError as exc:
        raise ValueError(f"codemode @options line is not valid JSON: {exc}") from None
    if not isinstance(options, dict) or set(options) - {"timeout_ms", "max_output_tokens"}:
        raise ValueError('codemode @options supports "timeout_ms" and "max_output_tokens"')
    for key, value in options.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"codemode @options {key} must be a positive integer")
    rest = code[code.index(first) + len(first):].lstrip("\r\n")
    return options, rest


class BridgeError(Exception):
    """A worker payload failed validation.

    The worker writes the bridge protocol by hand inside the script's process,
    so a worker defect or a script that reached past its sandbox can make it
    send malformed data. Payloads that do not decode or validate raise this
    error, and the run finishes as a sandbox error instead of feeding bad data
    into the host.
    """


def parse_bridge_json(payload: Union[str, bytes], what: str) -> Any:
    """The one validating decode entry point for data received from the worker."""

    try:
        return json.loads(payload)
    except ValueError:
        raise BridgeError(f"{what} is not valid JSON") from None


def parse_script_error(error: Any) -> dict:
    """The ``script_error`` payload: ``{name?, message, stack?}``, all strings."""

    if not isinstance(error, dict):
        raise BridgeError("script error is not an object")
    name = error.get("name")
    message = error.get("message")
    stack = error.get("stack")
    if not isinstance(message, str) or (name is not None and not isinstance(name, str)) \
            or (stack is not None and not isinstance(stack, str)):
        raise BridgeError("script error is malformed")
    parsed = {"message": message}
    if name is not None:
        parsed["name"] = name
    if stack is not None:
        parsed["stack"] = stack
    return parsed


def parse_store_writes(worker_writes: Any) -> Dict[str, Any]:
    """The ``done`` write set: ``{"set": {key: value}, "delete": [key, ...]}``.

    Unlike the TS prelude, the Python worker sends values as decoded JSON, not
    JSON text, so only the container shapes are checked here.
    """

    if not isinstance(worker_writes, dict):
        raise BridgeError("store writes are not an object")
    stored = worker_writes.get("set")
    if not isinstance(stored, dict):
        raise BridgeError('store writes "set" is not an object')
    deleted = worker_writes.get("delete")
    if not isinstance(deleted, list) or not all(isinstance(key, str) for key in deleted):
        raise BridgeError('store writes "delete" is not a list of keys')
    return {"set": stored, "delete": deleted}


def _sandbox_error(message: str) -> dict:
    return {"name": "SandboxError", "message": message}


async def execute_codemode(
    *,
    code: str,
    root_id: str,
    tools_meta: List[dict],
    store: Dict[str, Any],
    timeout_ms: Optional[int],
    signal: Optional[AbortSignal],
    on_call: CallExecutor,
    on_update: Optional[UpdateListener] = None,
    runtime: CodemodeRuntime,
) -> CodemodeOutcome:
    """Run ``code`` in a fresh ``python -I`` subprocess and collect the outcome.

    ``on_call`` executes one script call through the caller's pipeline and may
    backfill ``record.id``; ``on_update`` receives ``{"calls": [...]}`` at the
    same points the monolithic implementation published its snapshot.
    """

    code_path = runtime.workdir / f"script_{to_codemode_file_stem(root_id)}.py"
    code_path.write_text(code, encoding="utf-8")
    init = {"tools": tools_meta, "store": store}
    started_at = asyncio.get_event_loop().time()

    calls: List[CallRecord] = []
    writes = {"set": {}, "delete": []}
    output_items: List[dict] = []
    pending: Dict[int, asyncio.Future] = {}
    state = {"value": None, "error": None, "finished": False, "timed_out": False, "crashed": False}

    def snapshot() -> dict:
        return {"calls": [record.as_dict() for record in calls]}

    def publish() -> None:
        if on_update is not None:
            outcome = on_update(snapshot())
            if asyncio.iscoroutine(outcome):
                asyncio.ensure_future(outcome)

    process = await asyncio.create_subprocess_exec(
        sys.executable, "-I", str(runtime.worker_path), str(code_path),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        limit=64 * 1024 * 1024)

    async def feed():
        stream = process.stdin
        stream.write((json.dumps(init, ensure_ascii=False) + "\n").encode())
        await stream.drain()

    async def pump():
        while True:
            line = await process.stdout.readline()
            if not line:
                state["crashed"] = not state["finished"]
                return
            # An exception here used to bubble out of the pump task (or the
            # payload was skipped), leaving the run hung or unsettled with no
            # reason. Every payload is validated and a broken bridge fails the
            # run as a sandbox error instead.
            try:
                message = parse_bridge_json(line, "worker message")
                if not isinstance(message, dict):
                    raise BridgeError("unknown message from the worker")
                kind = message.get("type")
                if kind == "call":
                    await _start_call(message)
                elif kind == "output":
                    item = message.get("item")
                    if not isinstance(item, dict):
                        raise BridgeError("output message has no item object")
                    output_items.append(item)
                elif kind == "script_error":
                    state["error"] = parse_script_error(message.get("error"))
                    return
                elif kind == "done":
                    # Decode everything before recording it: a malformed
                    # payload must leave nothing behind in the outcome.
                    worker_writes = parse_store_writes(message.get("writes"))
                    state["value"] = message.get("value")
                    state["finished"] = True
                    writes["set"].update(worker_writes["set"])
                    for key in worker_writes["delete"]:
                        writes["set"].pop(key, None)
                        if key not in writes["delete"]:
                            writes["delete"].append(key)
                    return
                else:
                    raise BridgeError("unknown message from the worker")
            except BridgeError as error:
                state["error"] = _sandbox_error(
                    f"Sandbox bridge broken: {error}. The script may have corrupted the sandbox.")
                return
            except Exception as error:
                state["error"] = _sandbox_error(f"Sandbox host failed: {type(error).__name__}: {error}")
                return

    async def _start_call(message):
        # Payload validation before any state changes: a malformed call fails
        # the run instead of flowing into the call pipeline. An unknown tool
        # name stays a normal call error (the executor rejects it).
        call_id = message.get("id")
        if isinstance(call_id, bool) or not isinstance(call_id, int):
            raise BridgeError("call message has no integer id")
        if call_id in pending:
            raise BridgeError(f"duplicate call id {call_id}")
        name = message.get("tool")
        if not isinstance(name, str) or not name:
            raise BridgeError("call message has no tool name")
        args = message.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise BridgeError("tool arguments are not an object")
        index = len(calls) + 1
        try:
            preview = json.dumps(args, ensure_ascii=False)[:ARGS_PREVIEW_CHARS]
        except (TypeError, ValueError):
            preview = ""
        record = CallRecord(id=f"{root_id}/{index}", name=name, args=preview)
        calls.append(record)
        publish()
        started = asyncio.get_event_loop().time()
        future = asyncio.ensure_future(on_call(name, args, signal, record))
        pending[call_id] = future

        def done(fut):
            record.duration_ms = round((asyncio.get_event_loop().time() - started) * 1000)
            if fut.cancelled():
                record.status = "cancelled"
            elif fut.exception() is not None:
                record.status = "cancelled" if signal is not None and signal.aborted else "error"
                record.error = str(fut.exception())[:ERROR_PREVIEW_CHARS]
            else:
                record.status = "ok"
            pending.pop(call_id, None)
            publish()
            asyncio.ensure_future(_write_reply(process, _reply_from_future(call_id, fut)))

        future.add_done_callback(done)

    def _reply_from_future(message_id, fut) -> dict:
        if fut.cancelled():
            return {"id": message_id, "ok": False, "error": "aborted"}
        if fut.exception() is not None:
            return {"id": message_id, "ok": False, "error": str(fut.exception())}
        return {"id": message_id, "ok": True, "value": fut.result()}

    feeder = asyncio.ensure_future(feed())
    pump_task = asyncio.ensure_future(pump())
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_ms / 1000 if timeout_ms else None
    aborted = False
    try:
        if signal is not None:
            signal.throw_if_aborted()
        while not pump_task.done():
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                state["timed_out"] = True
                break
            wait_timeout = 0.5 if remaining is None else min(remaining, 0.5)
            done_set, _ = await asyncio.wait({pump_task}, timeout=wait_timeout)
            if pump_task in done_set:
                break
            if signal is not None and signal.aborted:
                aborted = True
                break
        if pump_task.done() and not pump_task.cancelled() and pump_task.exception() is not None:
            state["error"] = _sandbox_error(f"codemode bridge failed: {pump_task.exception()!r}")
    finally:
        feeder.cancel()
        pump_task.cancel()
        for future in list(pending.values()):
            future.cancel()
        if not feeder.done():
            try:
                await feeder
            except (asyncio.CancelledError, OSError, ConnectionError, RuntimeError):
                pass
        for task in (feeder, pump_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        for future in list(pending.values()):
            if future.cancelled():
                continue
            try:
                await future
            except BaseException:
                pass
        if process.returncode is None:
            process.kill()
        await process.wait()
        stdin = process.stdin
        if stdin is not None:
            # Drain the stdin transport: a killed worker can leave a pending
            # pipe write whose BrokenPipeError must be retrieved here instead
            # of surfacing later as an unretrieved-future warning.
            stdin.close()
            try:
                await asyncio.wait_for(stdin.wait_closed(), timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError,
                    OSError, RuntimeError):
                pass
    for record in calls:
        if record.status == "running":
            record.status = "cancelled"

    wall_time = asyncio.get_event_loop().time() - started_at
    if aborted:
        status: str = STATUS_ABORTED
    elif state["timed_out"]:
        status = STATUS_TIMEOUT
    elif state["crashed"] and not state["finished"] and state["error"] is None:
        status = STATUS_CRASHED
    elif state["error"] is not None:
        status = STATUS_SCRIPT_ERROR
    else:
        status = STATUS_OK
    return CodemodeOutcome(
        status=status,
        value=state["value"],
        output_items=output_items,
        calls=[record.as_dict() for record in calls],
        writes=writes,
        wall_time=wall_time,
        error=state["error"],
        exit_code=process.returncode if process.returncode is not None else "unknown",
    )


async def _write_reply(process, reply: dict) -> None:
    if process.stdin is None or process.returncode is not None:
        return
    try:
        process.stdin.write((json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode())
        await process.stdin.drain()
    except (ConnectionError, RuntimeError):
        pass
