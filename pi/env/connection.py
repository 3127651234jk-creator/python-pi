"""Client of the pi-env daemon's wire protocol (port of ``packages/env/src/connection.ts``).

Speaks the frame protocol of ``docs/protocol.md`` (version 1) over a daemon
started on the daemon's stdin/stdout: ``command`` names the program, and the
connection appends ``serve --token <hex>``. The daemon's first output is the
sync line ``PI-ENV <token>``; everything before it is startup noise.

A connection is started lazily on the first request and started again after it
is lost. Requests in flight when it is lost fail with ``RemoteError`` code
``unknown`` and ``lost=True``; for mutations their outcome is then unknown.

Porting deviations from the TS implementation:
- abort signals are ``asyncio.Event`` instead of ``AbortSignal``;
- ``stderr`` is always drained (a full pipe would block the daemon) and forwarded to
  ``on_log`` in chunks.
"""

from __future__ import annotations

import asyncio
import json as json_module
import secrets
import struct
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Union

# Frame types (docs/protocol.md).
REQUEST = 1
RESULT = 2
ERROR = 3
EVENT = 4
CANCEL = 5
PING = 6

MAX_FRAME = 16 * 1024 * 1024
PING_INTERVAL_MS = 5000
# The daemon pings every five seconds; this much silence means the connection is gone.
SILENCE_LIMIT_MS = 30_000
# How long starting the daemon and its ``hello`` may take.
START_TIMEOUT_MS = 60_000

Json = Dict[str, Any]

CommandSpec = Union[Sequence[str], Callable[[], Awaitable[Sequence[str]]]]


@dataclass
class RemoteError(Exception):
    """A failure reported by the daemon, with its Node-style code."""

    fields: Json

    def __post_init__(self) -> None:
        message = self.fields.get("message")
        super().__init__(message if isinstance(message, str) else "Remote operation failed")
        self.name = "RemoteError"
        self.code = self.fields["code"] if isinstance(self.fields.get("code"), str) else "unknown"
        self.path: Optional[str] = self.fields["path"] if isinstance(self.fields.get("path"), str) else None


@dataclass
class RemoteInfo:
    """What ``hello`` reports about the remote machine."""

    protocol: int
    version: str
    os: str
    arch: str
    home: str
    tmpdir: str
    # Path separator: ``/``, or ``\`` on Windows.
    separator: str
    # The daemon's working directory, where ``path.resolve`` falls back to ``cwd``.
    cwd: str
    # Windows: per-drive working directories (``C:`` -> ``C:\work``) of the ``=C:`` variables.
    drive_cwds: Dict[str, str]
    pid: int


@dataclass
class Reply:
    json: Json
    payload: bytes
    # The daemon session that answered; handles from it are only valid while it lives.
    session: int


@dataclass
class ConnectionOptions:
    """
    Command that starts the daemon, before its ``serve --token <hex>`` arguments, e.g.
    ``["ssh", "-T", "--", "host", "~/.pi/mobile/tools/pi-env"]``. A function computes it at each
    start (detecting and deploying first, for example); its failure fails that start, and the next
    request tries again.
    """

    command: CommandSpec
    # Receives the daemon's and the transport's diagnostic output.
    on_log: Optional[Callable[[str], None]] = None


def _well_formed(value: Any) -> Any:
    """Lone surrogates become U+FFFD, as Node encodes strings for system calls."""
    if isinstance(value, str):
        return value.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    if isinstance(value, dict):
        return {_well_formed(key) if isinstance(key, str) else key: _well_formed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_well_formed(item) for item in value]
    return value


def frame(kind: int, request_id: int, json: Json, payload: bytes = b"") -> bytes:
    body = json_module.dumps(_well_formed(json), ensure_ascii=False).encode("utf-8")
    header = struct.pack(">IBII", 9 + len(body) + len(payload), kind, request_id, len(body))
    return header + body + payload


# The connection failed or ended; a request may succeed once it is started again.
def lost(message: str) -> RemoteError:
    return RemoteError({"code": "unknown", "message": message, "lost": True})


def is_connection_lost(error: Any) -> bool:
    """Whether an error means the connection was lost, not that the daemon refused the request."""
    return isinstance(error, RemoteError) and error.fields.get("lost") is True


def _closed_error() -> RemoteError:
    return RemoteError({"code": "unknown", "message": "Connection closed"})


class _Pending:
    __slots__ = ("future", "on_event")

    def __init__(
        self, future: "asyncio.Future[Reply]", on_event: Optional[Callable[[Json, bytes], None]]
    ) -> None:
        self.future = future
        self.on_event = on_event


class _Session:
    """One running daemon."""

    def __init__(self, session_id: int, process: asyncio.subprocess.Process, token: str) -> None:
        self.id = session_id
        self.process = process
        self.token = token
        self.pending: Dict[int, _Pending] = {}
        self.buffer = bytearray()
        self.synced = False
        self.last_seen = 0.0
        self.timer: Optional[asyncio.Task] = None
        self.read_task: Optional[asyncio.Task] = None
        self.exit_task: Optional[asyncio.Task] = None
        self.stderr_task: Optional[asyncio.Task] = None
        self.live = True


@dataclass
class _RequestOptions:
    payload: Optional[bytes] = None
    # Aborting sends ``cancel``; the request still settles with the daemon's result or error.
    abort_signal: Optional[asyncio.Event] = None
    # Progress events of the request, such as ``exec`` output.
    on_event: Optional[Callable[[Json, bytes], None]] = None
    # The request's id and session, for ``kill``.
    on_start: Optional[Callable[[int, int], None]] = None
    # Only run in this daemon session, for requests on handles it opened. After the connection was
    # lost and started again, such requests fail instead of reaching a daemon that never opened the
    # handle.
    session: Optional[int] = None


class Connection:
    """
    One connection to a pi-env daemon: started lazily on the first request and started again after
    it is lost. Requests in flight when it is lost fail with code ``unknown``; for mutations their
    outcome is then unknown.
    """

    def __init__(self, options: Union[ConnectionOptions, CommandSpec], on_log: Optional[Callable[[str], None]] = None) -> None:
        if isinstance(options, ConnectionOptions):
            self._options = options
        else:
            self._options = ConnectionOptions(command=options, on_log=on_log)
        self._session: Optional[_Session] = None
        self._last_session: Optional[_Session] = None
        self._ready: Optional["asyncio.Task"] = None
        self._sessions = 0
        self._next_id = 1
        self._closed = False

    # -- lifecycle -------------------------------------------------------------

    async def info(self) -> RemoteInfo:
        """Connect if needed and return what the daemon reported about its machine."""
        return (await self._connect())[0]

    async def session_id(self) -> int:
        """Connect if needed and return the live session's id."""
        return (await self._connect())[1].id

    def close(self) -> None:
        """Stop the daemon; it kills everything it started."""
        self._closed = True
        if self._session is not None:
            self._teardown(self._session, _closed_error())

    async def wait_closed(self) -> None:
        """Wait for the daemon process to exit after ``close()``; a no-op when it never ran."""
        session = self._session or self._last_session
        if session is None:
            return
        try:
            await session.process.wait()
        except (OSError, RuntimeError):
            return

    def kill(self, request_id: int, session_id: int) -> None:
        """Kill one ``exec`` without aborting it."""
        current = self._session
        if current is not None and current.id == session_id and current.live:
            self._write(current, frame(CANCEL, request_id, {"mode": "kill"}))

    def _kill_aborted(self, request_id: int, session_id: int) -> None:
        current = self._session
        if current is not None and current.id == session_id and current.live:
            self._write(current, frame(CANCEL, request_id, {}))

    # -- requests --------------------------------------------------------------

    async def request(
        self,
        op: str,
        json: Optional[Json] = None,
        *,
        payload: Optional[bytes] = None,
        abort_signal: Optional[asyncio.Event] = None,
        on_event: Optional[Callable[[Json, bytes], None]] = None,
        on_start: Optional[Callable[[int, int], None]] = None,
        session: Optional[int] = None,
    ) -> Reply:
        options = _RequestOptions(
            payload=payload,
            abort_signal=abort_signal,
            on_event=on_event,
            on_start=on_start,
            session=session,
        )
        if options.session is not None:
            current = self._session
            if current is None or current.id != options.session or not current.live:
                raise lost("pi-env connection lost")
            return await self._send(current, op, json or {}, options)
        info_session = await self._connect()
        return await self._send(info_session[1], op, json or {}, options)

    async def _connect(self) -> "tuple[RemoteInfo, _Session]":
        if self._closed:
            raise _closed_error()
        if self._ready is None:
            ready = asyncio.ensure_future(self._start())
            self._ready = ready

            def _observe(task: "asyncio.Task") -> None:
                if task.cancelled():
                    return
                failure = task.exception()  # Retrieved here, and by any waiter's own await.
                if failure is not None and self._ready is task:
                    # A failed start is not remembered: the next request tries again.
                    self._ready = None

            ready.add_done_callback(_observe)
        # Shielded: a caller that gives up (timeout, cancellation) must not cancel the start
        # every later request shares.
        return await asyncio.shield(self._ready)

    async def _send(self, session: _Session, op: str, json: Json, options: Optional[_RequestOptions] = None) -> Reply:
        options = options or _RequestOptions()
        if not session.live:
            raise lost("pi-env connection lost")
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Reply]" = loop.create_future()
        session.pending[request_id] = _Pending(future, options.on_event)

        async def _cancel_when_aborted() -> None:
            if options.abort_signal is None:
                return
            try:
                await options.abort_signal.wait()
            except asyncio.CancelledError:
                raise
            self._kill_aborted(request_id, session.id)

        abort_task = asyncio.ensure_future(_cancel_when_aborted()) if options.abort_signal else None
        try:
            self._write(session, frame(REQUEST, request_id, {**json, "op": op}, options.payload or b""))
            if options.on_start is not None:
                options.on_start(request_id, session.id)
            return await future
        finally:
            session.pending.pop(request_id, None)
            if abort_task is not None:
                abort_task.cancel()
                try:
                    await abort_task
                except asyncio.CancelledError:
                    pass

    def _write(self, session: _Session, data: bytes) -> None:
        if session.live and session.process.stdin is not None:
            try:
                session.process.stdin.write(data)
            except (ConnectionResetError, RuntimeError, OSError):
                pass  # Teardown reports the loss.

    # -- starting --------------------------------------------------------------

    async def _start(self) -> "tuple[RemoteInfo, _Session]":
        try:
            command = self._options.command
            resolved = list(await command()) if callable(command) else list(command)
        except Exception as error:  # noqa: BLE001 - a failed command fails that start
            raise RemoteError({"code": "spawn_error", "message": str(error), "lost": True}) from None
        if self._closed:
            raise _closed_error()
        if not resolved:
            raise RemoteError({"code": "spawn_error", "message": "No daemon command"})
        program, args = resolved[0], resolved[1:]
        token = secrets.token_hex(16)
        try:
            process = await asyncio.create_subprocess_exec(
                program,
                *args,
                "serve",
                "--token",
                token,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise RemoteError({"code": "spawn_error", "message": str(error), "lost": True}) from None
        self._sessions += 1
        session = _Session(self._sessions, process, token)
        self._session = session
        self._last_session = session
        if self._closed:
            # A ``close()`` while the start was in flight: do not leave the daemon running.
            self._teardown(session, _closed_error())
            raise _closed_error()
        loop = asyncio.get_running_loop()
        session.last_seen = loop.time()
        failed: "asyncio.Future[None]" = loop.create_future()

        def _retrieve_failure(_future: "asyncio.Future") -> None:
            # The failure may be reported after the start settled; still retrieve it.
            if not _future.cancelled():
                _future.exception()

        failed.add_done_callback(_retrieve_failure)

        def _fail(message: str) -> None:
            if not failed.done():
                failed.set_exception(lost(message))

        start_timer = loop.call_later(
            START_TIMEOUT_MS / 1000, lambda: _fail(f"pi-env did not answer within {START_TIMEOUT_MS // 1000} s")
        )

        async def _watch_exit() -> None:
            code = await process.wait()
            _fail(f"pi-env exited with code {code} before it was ready")
            self._teardown(session, lost("pi-env connection lost"))

        # The tasks are stored on the session: a task no one references can be garbage-collected.
        session.exit_task = asyncio.ensure_future(_watch_exit())
        session.stderr_task = asyncio.ensure_future(self._pump_stderr(session))
        session.read_task = asyncio.ensure_future(self._read_loop(session))
        session.timer = asyncio.ensure_future(self._ping_loop(session))
        hello: Optional["asyncio.Task"] = None
        try:
            hello = asyncio.ensure_future(self._send(session, "hello", {"protocol": 1}))
            await asyncio.wait({hello, failed}, return_when=asyncio.FIRST_COMPLETED)
            if failed.done() and not hello.done():
                failure = failed.exception()
                self._teardown(session, failure if isinstance(failure, Exception) else lost(str(failure)))
                raise failure or lost("pi-env start failed")
            reply = await hello
            if reply.json.get("protocol") != 1:
                raise lost(f"Unsupported protocol {reply.json.get('protocol')}")
            json = reply.json
            return (
                RemoteInfo(
                    protocol=json.get("protocol", 0),
                    version=json.get("version", ""),
                    os=json.get("os", ""),
                    arch=json.get("arch", ""),
                    home=json.get("home", ""),
                    tmpdir=json.get("tmpdir", ""),
                    separator=json.get("separator", "/"),
                    cwd=json.get("cwd", ""),
                    drive_cwds={
                        key: value
                        for key, value in (json.get("driveCwds") or {}).items()
                        if isinstance(value, str)
                    },
                    pid=json.get("pid", 0),
                ),
                session,
            )
        except BaseException as error:
            self._teardown(session, error if isinstance(error, Exception) else lost(str(error)))
            raise
        finally:
            start_timer.cancel()
            if hello is not None:
                if not hello.done():
                    # The start failed first: this request never settles on its own.
                    hello.cancel()
                elif not hello.cancelled():
                    hello.exception()  # Retrieve it; the caller saw the start's own failure.
            if failed.done() and not failed.cancelled():
                failed.exception()  # A late exit is reported through teardown; retrieve it.
            else:
                failed.cancel()
            # Keep the exit watcher: it tears the session down on daemon exit. Nothing else leaks.

    async def _pump_stderr(self, session: _Session) -> None:
        stream = session.process.stderr
        if stream is None:
            return
        # Always read: a full stderr pipe would block the daemon, whether or not anyone logs it.
        # Chunks, not lines: a line longer than the stream limit would end the pump.
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    return
                if self._options.on_log is not None:
                    self._options.on_log(chunk.decode("utf-8", "replace"))
        except (ConnectionResetError, RuntimeError, OSError, ValueError):
            return

    async def _ping_loop(self, session: _Session) -> None:
        loop = asyncio.get_running_loop()
        while session.live:
            await asyncio.sleep(PING_INTERVAL_MS / 1000)
            if not session.live:
                return
            self._write(session, frame(PING, 0, {}))
            if loop.time() - session.last_seen > SILENCE_LIMIT_MS / 1000:
                self._teardown(session, lost("pi-env connection timed out"))

    def _teardown(self, session: _Session, error: Exception) -> None:
        if not session.live:
            return
        session.live = False
        if session.timer is not None:
            session.timer.cancel()
        if self._session is session:
            self._session = None
            self._ready = None
        if session.process.stdin is not None:
            try:
                session.process.stdin.close()
            except (OSError, RuntimeError):
                pass
        if session.process.returncode is None:
            try:
                session.process.kill()
            except (OSError, ProcessLookupError):
                pass
        # Close the pipes now, so their transports do not warn after the loop is gone.
        transport = getattr(session.process, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        pending = list(session.pending.values())
        session.pending.clear()
        for request in pending:
            if not request.future.done():
                request.future.set_exception(error)

    # -- reading ---------------------------------------------------------------

    async def _read_loop(self, session: _Session) -> None:
        stream = session.process.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk or not session.live:
                    return
                self._on_data(session, chunk)
        except (ConnectionResetError, RuntimeError, OSError):
            return

    def _on_data(self, session: _Session, chunk: bytes) -> None:
        if not session.live:
            return
        session.last_seen = asyncio.get_running_loop().time()
        session.buffer.extend(chunk)
        if not session.synced:
            # Shell startup files may print before the daemon runs; skip everything before its sync line.
            marker = f"PI-ENV {session.token}\n".encode("utf-8")
            index = session.buffer.find(marker)
            if index == -1:
                if len(session.buffer) > 1024 * 1024:
                    del session.buffer[: len(session.buffer) - len(marker)]
                return
            noise = bytes(session.buffer[:index]).decode("utf-8", "replace").strip()
            if noise and self._options.on_log is not None:
                self._options.on_log(noise)
            del session.buffer[: index + len(marker)]
            session.synced = True
        while session.live and len(session.buffer) >= 4:
            length = int.from_bytes(session.buffer[:4], "big")
            if length < 9 or length > MAX_FRAME:
                self._teardown(session, lost("Corrupt frame from pi-env"))
                return
            if len(session.buffer) < 4 + length:
                return
            # A view over the buffer; released before the buffer is resized.
            view = memoryview(session.buffer)
            body = view[4 : 4 + length]
            kind = body[0]
            request_id = int.from_bytes(body[1:5], "big")
            json_length = int.from_bytes(body[5:9], "big")
            try:
                if 9 + json_length > length:
                    raise ValueError("JSON length out of range")
                parsed = json_module.loads(bytes(body[9 : 9 + json_length]).decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("not an object")
            except (ValueError, UnicodeDecodeError):
                body.release()
                view.release()
                self._teardown(session, lost("Corrupt frame from pi-env"))
                return
            # A plain bytes copy, like the reader results, not a view of the receive buffer.
            payload = bytes(body[9 + json_length :])
            body.release()
            view.release()
            del session.buffer[: 4 + length]
            try:
                self._dispatch(session, kind, request_id, parsed, payload)
            except Exception:
                # A failing event callback must not stop the read loop; the request itself reported it.
                pass

    def _dispatch(self, session: _Session, kind: int, request_id: int, json: Json, payload: bytes) -> None:
        pending = session.pending.get(request_id)
        if kind == EVENT:
            if pending is not None and pending.on_event is not None:
                pending.on_event(json, payload)
            return
        if kind not in (RESULT, ERROR) or pending is None:
            return
        session.pending.pop(request_id, None)
        if pending.future.done():
            return
        if kind == RESULT:
            pending.future.set_result(Reply(json=json, payload=payload, session=session.id))
        else:
            pending.future.set_exception(RemoteError(json))
