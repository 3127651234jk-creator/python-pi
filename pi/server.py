"""Async routed Pi v8 server. Application hosts own service payload semantics.

host.attach_client(presentation) returns a server endpoint;
host.open_session(await host.resolve_session(id)) returns a shared session handle.
Endpoints implement invoke_service(call, publish, cancelled) and release().
Session handles implement attach_client() and close().
"""

import asyncio
import uuid

from .protocol import MessageDecoder, encode_message, PROTOCOL_VERSION, MAX_FRAME_LENGTH, ProtocolValidationError, validate_target


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


class Server:
    def __init__(self, host, server_id=None, max_frame_length=MAX_FRAME_LENGTH, handshake_timeout=10):
        self.host = host
        self.server_id = server_id or str(uuid.uuid4())
        validate_target({"serverId": self.server_id})
        self.limit, self.handshake_timeout = max_frame_length, handshake_timeout
        self.listener = None
        self.connections = set()
        self.writers = set()
        self.sessions = {}
        self.session_lock = asyncio.Lock()

    async def session(self, session_id):
        async with self.session_lock:
            if session_id not in self.sessions:
                metadata = await self.host.resolve_session(session_id)
                self.sessions[session_id] = await self.host.open_session(metadata)
            return self.sessions[session_id]

    async def start_tcp(self, host="127.0.0.1", port=0):
        self.listener = await asyncio.start_server(self._accept, host, port)
        return self.listener.sockets[0].getsockname()

    async def start_unix(self, path):
        self.listener = await asyncio.start_unix_server(self._accept, path=path)
        return path

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
        pending = {}
        endpoint = None
        decoder = MessageDecoder("client", self.limit)
        ready = False
        async def invoke(message, target_endpoint, cancelled):
            async def publish(subscription_id, update):
                if "sessionId" in message["target"] and message["target"] != presentation.attachment:
                    return
                await send({"type": "service_update", "subscriptionId": subscription_id, "update": update})
            try:
                result = await target_endpoint.invoke_service(message["call"], publish, cancelled)
                await send({"type": "response", "id": message["id"], "ok": True, "result": result})
            except Exception as exc:
                try:
                    await send({"type": "response", "id": message["id"], "ok": False,
                                "error": {"code": getattr(exc, "code", "service_error"), "message": str(exc)}})
                except (ConnectionError, OSError):
                    pass
            finally:
                pending.pop(message["id"], None)
                if not pending:
                    retired, presentation.retired = presentation.retired, []
                    for owned in retired:
                        await owned.release()
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
                            raise ProtocolValidationError("Duplicate request ID")
                        cancelled = asyncio.Event()
                        task = asyncio.create_task(invoke(message, selected, cancelled))
                        pending[message["id"]] = (task, cancelled, target)
        except (ConnectionError, OSError, ValueError, asyncio.TimeoutError):
            pass
        finally:
            writer.close()
            if pending:
                await asyncio.gather(*(entry[0] for entry in list(pending.values())), return_exceptions=True)
            for owned in [endpoint, presentation.endpoint, *presentation.retired]:
                if owned is not None:
                    await owned.release()
            self.writers.discard(writer)
            self.connections.discard(current)
            await writer.wait_closed()

    async def close(self):
        if self.listener:
            self.listener.close()
            await self.listener.wait_closed()
        for writer in list(self.writers):
            writer.close()
        await asyncio.gather(*list(self.connections), return_exceptions=True)
        for handle in self.sessions.values():
            await handle.close()
        self.sessions.clear()
