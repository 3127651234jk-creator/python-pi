"""Async routed Pi v8 server. Application hosts own service payload semantics.

host.attach_client(presentation) returns a server endpoint;
host.open_session(await host.resolve_session(id)) returns a shared session handle.
Endpoints implement invoke_service(call, publish, cancelled) and release().
Session handles implement attach_client() and close(); a handle may expose a
``terminated`` future (an Exception instance on unexpected termination, None on
an expected close) which the server watches to invalidate the session.
"""

import asyncio
import uuid

from .chord import StateEncoder, decode_control_call, parse_service_call, parse_subscription_snapshot
from .protocol import MessageDecoder, encode_message, PROTOCOL_VERSION, MAX_FRAME_LENGTH, ProtocolValidationError, validate_target


class SubscriptionPump:
    """Ordered sender for one chord subscription: buffers endpoint updates until
    the subscribe response has been sent, then encodes each update with the
    subscription's state codec and forwards it in publication order."""

    def __init__(self, subscription_id, send):
        self.subscription_id = subscription_id
        self.send = send
        self.encoder = None
        self.dead = False
        self.ready = asyncio.Event()
        self.queue = asyncio.Queue()
        self.task = asyncio.create_task(self._run())

    def set_encoder(self, encoder):
        self.encoder = encoder

    def enqueue(self, update):
        self.queue.put_nowait(update)

    def stop(self):
        self.task.cancel()

    async def _run(self):
        try:
            await self.ready.wait()
            while True:
                update = await self.queue.get()
                encoded = self.encoder.encode_update(update)
                await self.send({"type": "service_update", "subscriptionId": self.subscription_id,
                                 "update": encoded})
        except asyncio.CancelledError:
            raise
        except Exception:
            # A broken state stream ends this pump; TS closes the whole connection
            # here, the compact port drops the stream and further updates.
            self.dead = True


class RoutingError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Presentation:
    def __init__(self, server, send):
        self.server, self.send = server, send
        self.attachment = None
        self.endpoint = None
        self.retired = []
        self.pending = 0  # in-flight invoke_service calls served by this presentation
        self.lock = asyncio.Lock()

    async def attach_session(self, session_id):
        async with self.lock:
            if self.attachment and self.attachment["sessionId"] == session_id:
                return
            handle = await self.server.session(session_id)
            endpoint = await handle.attach_client()
            if self.endpoint is not None:
                self.retired.append(self.endpoint)
            self.endpoint = endpoint
            self.attachment = {"serverId": self.server.server_id, "sessionId": session_id,
                               "attachmentId": str(uuid.uuid4())}
            await self.send({"type": "attachment", "attachment": self.attachment})

    async def detach_session(self):
        async with self.lock:
            if self.endpoint is not None:
                self.retired.append(self.endpoint)
            self.endpoint = None
            self.attachment = None
            await self.send({"type": "attachment", "attachment": None})

    async def prepare_session_removal(self, session_id):
        """Release the session across every connection; management services call this
        before removing the application's durable metadata."""
        await self.server.remove_session(session_id)

    async def release_retired(self):
        retired, self.retired = self.retired, []
        for owned in retired:
            await owned.release()


class Server:
    def __init__(self, host, server_id=None, max_frame_length=MAX_FRAME_LENGTH, handshake_timeout=10):
        self.host = host
        self.server_id = server_id or str(uuid.uuid4())
        validate_target({"serverId": self.server_id})
        self.limit, self.handshake_timeout = max_frame_length, handshake_timeout
        self.listener = None
        self.connections = set()
        self.writers = set()
        self.presentations = set()
        self.sessions = {}
        self.terminated_watchers = {}
        self.session_lock = asyncio.Lock()

    async def session(self, session_id):
        async with self.session_lock:
            if session_id not in self.sessions:
                metadata = await self.host.resolve_session(session_id)
                handle = await self.host.open_session(metadata)
                self.sessions[session_id] = handle
                self._watch_terminated(session_id, handle)
            return self.sessions[session_id]

    def _watch_terminated(self, session_id, handle):
        terminated = getattr(handle, "terminated", None)
        if terminated is None:
            return

        async def watch():
            error = await terminated
            if error is not None:
                await self.invalidate_session(session_id, error)

        self.terminated_watchers[session_id] = asyncio.create_task(watch())

    async def remove_session(self, session_id):
        """Release every client attachment of a session, close its handle, and evict it.

        New attachments race with removal only until the metadata itself is deleted;
        the caller (management service) removes durable metadata after this returns.
        """
        async with self.session_lock:
            handle = self.sessions.pop(session_id, None)
            watcher = self.terminated_watchers.pop(session_id, None)
        if watcher is not None:
            watcher.cancel()
        await self._release_attachments(session_id)
        if handle is not None:
            await handle.close()

    async def invalidate_session(self, session_id, error):
        """Drop a session whose handle terminated unexpectedly; its attachments are
        released with ``attachment: null``. The handle is not closed (it is already dead)."""
        async with self.session_lock:
            if session_id not in self.sessions:
                return
            del self.sessions[session_id]
            watcher = self.terminated_watchers.pop(session_id, None)
        if watcher is not None:
            watcher.cancel()
        await self._release_attachments(session_id)

    async def _release_attachments(self, session_id):
        for presentation in list(self.presentations):
            if not (presentation.attachment and presentation.attachment["sessionId"] == session_id):
                continue
            try:
                async with presentation.lock:
                    # Re-check under the lock: the connection may have attached or detached meanwhile.
                    if not (presentation.attachment and presentation.attachment["sessionId"] == session_id):
                        continue
                    if presentation.endpoint is not None:
                        presentation.retired.append(presentation.endpoint)
                    presentation.endpoint = None
                    presentation.attachment = None
                    await presentation.send({"type": "attachment", "attachment": None})
            except (ConnectionError, OSError):
                continue  # dead connection; its own finally path releases the endpoints
            if presentation.pending == 0:
                await presentation.release_retired()

    async def start_tcp(self, host="127.0.0.1", port=0):
        self.listener = await asyncio.start_server(self._accept, host, port)
        return self.listener.sockets[0].getsockname()

    async def start_unix(self, path):
        self.listener = await asyncio.start_unix_server(self._accept, path=path)
        return path

    async def start_streams(self, reader, writer):
        """Serve exactly one routed connection over the given byte streams (worker over stdio)."""
        task = asyncio.create_task(self._accept(reader, writer))
        self.connections.add(task)
        return task

    async def _accept(self, reader, writer):
        current = asyncio.current_task()
        self.connections.add(current)
        self.writers.add(writer)
        send_lock = asyncio.Lock()
        async def send(message):
            data = encode_message(message, "server", self.limit)
            async with send_lock:
                writer.write(data)
                await writer.drain()
        presentation = Presentation(self, send)
        self.presentations.add(presentation)
        pending = {}
        encoders = {}  # subscription id -> SubscriptionPump for chord subscriptions
        endpoint = None
        decoder = MessageDecoder("client", self.limit)
        ready = False
        async def invoke(message, target_endpoint, cancelled):
            target = message["target"]
            async def publish(subscription_id, update):
                if "sessionId" in target and target != presentation.attachment:
                    return
                pump = encoders.get(subscription_id)
                if pump is not None:
                    if not pump.dead:
                        pump.enqueue(update)
                    return
                await send({"type": "service_update", "subscriptionId": subscription_id, "update": update})
            pump = None
            try:
                try:
                    call = parse_service_call(message["call"])
                except (TypeError, ValueError) as exc:
                    await send({"type": "response", "id": message["id"], "ok": False,
                                "error": {"code": "invalid_request", "message": str(exc)}})
                    return
                control = decode_control_call(call)
                if control is not None and control["type"] == "subscribe":
                    if control["subscriptionId"] in encoders:
                        await send({"type": "response", "id": message["id"], "ok": False,
                                    "error": {"code": "invalid_request",
                                              "message": f"Duplicate service subscription {control['subscriptionId']}"}})
                        return
                    # Created before the endpoint installs the subscription so updates
                    # published during activation buffer until the snapshot response.
                    pump = SubscriptionPump(control["subscriptionId"], send)
                result = await target_endpoint.invoke_service(call, publish, cancelled)
                stopped = None
                if control is not None and control["type"] == "subscribe":
                    if result is None:
                        raise ProtocolValidationError("Service subscription did not return a snapshot")
                    encoder = StateEncoder()
                    result = encoder.encode_snapshot(parse_subscription_snapshot(result))
                    pump.set_encoder(encoder)
                    encoders[control["subscriptionId"]] = pump
                elif control is not None and control["type"] == "unsubscribe":
                    stopped = encoders.pop(control["subscriptionId"], None)
                await send({"type": "response", "id": message["id"], "ok": True, "result": result}
                           if result is not None else {"type": "response", "id": message["id"], "ok": True})
                if pump is not None:
                    pump.ready.set()
                if stopped is not None:
                    stopped.stop()
            except Exception as exc:
                if pump is not None and encoders.get(pump.subscription_id) is not pump:
                    pump.stop()  # created for a failed subscribe; nothing drains it
                try:
                    code = getattr(exc, "code", None)
                    if code is None:
                        if isinstance(exc, (TypeError, ValueError)):
                            code, text = "invalid_request", str(exc)
                        else:
                            code, text = "internal_error", "Internal server error"
                    else:
                        text = str(exc)
                    await send({"type": "response", "id": message["id"], "ok": False,
                                "error": {"code": code, "message": text}})
                except (ConnectionError, OSError):
                    pass
            finally:
                pending.pop(message["id"], None)
                presentation.pending -= 1
                if not pending:
                    await presentation.release_retired()
        try:
            while True:
                chunk = await asyncio.wait_for(reader.read(65536), self.handshake_timeout) if not ready else await reader.read(65536)
                if not chunk:
                    decoder.end()
                    break
                for message in decoder.push(chunk):
                    if not ready:
                        if message["type"] != "hello" or message["version"] != PROTOCOL_VERSION:
                            await send({"type": "hello_error", "error": {"code": "version_mismatch", "message": "Expected Pi protocol v8 hello"}})
                            return
                        endpoint = await self.host.attach_client(presentation)
                        await send({"type": "hello", "version": PROTOCOL_VERSION, "serverId": self.server_id})
                        ready = True
                        continue
                    if message["type"] == "hello":
                        raise ProtocolValidationError("Repeated hello")
                    target = message["target"]
                    selected = endpoint if "sessionId" not in target else presentation.endpoint
                    valid = target["serverId"] == self.server_id and ("sessionId" not in target or target == presentation.attachment)
                    if not valid or selected is None:
                        if message["type"] == "request":
                            await send({"type": "response", "id": message["id"], "ok": False,
                                        "error": {"code": "invalid_target", "message": "Stale or mismatched route"}})
                        continue
                    if message["type"] == "cancel":
                        active = pending.get(message["id"])
                        if active and active[2] == target:
                            active[1].set()
                    else:
                        if message["id"] in pending:
                            await send({"type": "response", "id": message["id"], "ok": False,
                                        "error": {"code": "invalid_request", "message": "Request ID is already active"}})
                            continue
                        cancelled = asyncio.Event()
                        # Count before the task runs so removal sees the in-flight call even
                        # while the invoke task has not started yet.
                        presentation.pending += 1
                        task = asyncio.create_task(invoke(message, selected, cancelled))
                        pending[message["id"]] = (task, cancelled, target)
        except (ConnectionError, OSError, ValueError, asyncio.TimeoutError):
            pass
        finally:
            writer.close()
            for pump in encoders.values():
                pump.stop()
            if pending:
                await asyncio.gather(*(entry[0] for entry in list(pending.values())), return_exceptions=True)
            presentation.retired.extend([item for item in (endpoint, presentation.endpoint) if item is not None])
            presentation.endpoint = None
            await presentation.release_retired()
            self.presentations.discard(presentation)
            self.writers.discard(writer)
            self.connections.discard(current)
            await writer.wait_closed()

    async def close(self):
        if self.listener:
            self.listener.close()
            await self.listener.wait_closed()
        for writer in list(self.writers):
            writer.close()
        for watcher in list(self.terminated_watchers.values()):
            watcher.cancel()
        self.terminated_watchers.clear()
        await asyncio.gather(*list(self.connections), return_exceptions=True)
        for handle in self.sessions.values():
            await handle.close()
        self.sessions.clear()
