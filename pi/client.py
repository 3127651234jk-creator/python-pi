"""Async Pi v8 byte-stream client, with explicit TCP/Unix reconnect."""

import asyncio

from .protocol import MessageDecoder, encode_message, PROTOCOL_VERSION, MAX_FRAME_LENGTH


class ServerError(RuntimeError):
    def __init__(self, error):
        super().__init__(error["message"])
        self.code = error["code"]


class Client:
    def __init__(self, server_id, max_frame_length=MAX_FRAME_LENGTH):
        self.server_id, self.limit = server_id, max_frame_length
        self.pending = {}
        self.updates = asyncio.Queue()
        self.attachment = None
        self.hello = None
        self.reader_task = None
        self.writer = None
        self.sequence = 0
        self.send_lock = asyncio.Lock()

    async def connect_tcp(self, host, port, timeout=10):
        return await self._connect(lambda: asyncio.open_connection(host, port), timeout)

    async def connect_unix(self, path, timeout=10):
        return await self._connect(lambda: asyncio.open_unix_connection(path), timeout)

    async def connect_streams(self, reader, writer, timeout=10):
        """Connect over an existing byte-stream pair, e.g. a worker process's stdio."""
        async def factory():
            return reader, writer

        return await self._connect(factory, timeout)

    async def _connect(self, factory, timeout):
        if self.writer is not None:
            raise RuntimeError("Client is already connected")
        self.handshake = asyncio.get_running_loop().create_future()
        reader, self.writer = await asyncio.wait_for(factory(), timeout)
        self.reader_task = asyncio.create_task(self._read(reader))
        try:
            await self._send({"type": "hello", "version": PROTOCOL_VERSION})
            self.hello = await asyncio.wait_for(self.handshake, timeout)
            return self
        except BaseException:
            await self.close()
            raise

    async def _send(self, message):
        if self.writer is None:
            raise ConnectionError("Client disconnected")
        data = encode_message(message, "client", self.limit)
        async with self.send_lock:
            self.writer.write(data)
            await self.writer.drain()

    async def _read(self, reader):
        decoder = MessageDecoder("server", self.limit)
        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    decoder.end()
                    raise ConnectionError("Server disconnected")
                for message in decoder.push(chunk):
                    kind = message["type"]
                    if not self.handshake.done():
                        if kind == "hello_error":
                            raise ServerError(message["error"])
                        if kind != "hello" or message["serverId"] != self.server_id:
                            raise ConnectionError("Server identity mismatch")
                        self.handshake.set_result(message)
                    elif kind == "response":
                        future = self.pending.get(message["id"])
                        if future is not None and not future.done():
                            if message["ok"]:
                                future.set_result(message.get("result"))
                            else:
                                future.set_exception(ServerError(message["error"]))
                    elif kind == "attachment":
                        route = message["attachment"]
                        if route and route["serverId"] != self.server_id:
                            raise ConnectionError("Attachment server mismatch")
                        self.attachment = route
                    elif kind == "service_update":
                        self.updates.put_nowait(message)
                    else:
                        raise ConnectionError("Unexpected server envelope")
        except (Exception, asyncio.CancelledError) as exc:
            for future in [self.handshake, *self.pending.values()]:
                if not future.done():
                    future.set_exception(exc if isinstance(exc, ServerError) else ConnectionError(str(exc)))
        finally:
            self.attachment = None
            if self.writer is not None:
                self.writer.close()

    async def request(self, target, call, timeout=60):
        if self.reader_task is None or self.reader_task.done():
            raise ConnectionError("Client disconnected")
        self.sequence += 1
        request_id = str(self.sequence)
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._send({"type": "request", "id": request_id, "target": target, "call": call})
            return await asyncio.wait_for(future, timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            try:
                await self._send({"type": "cancel", "id": request_id, "target": target})
            except (ConnectionError, OSError):
                pass
            raise
        finally:
            self.pending.pop(request_id, None)

    async def close(self):
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
        if self.reader_task is not None:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)
        self.writer = None
        self.reader_task = None
        self.attachment = None
