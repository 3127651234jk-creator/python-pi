import asyncio
import struct
import uuid
import shutil
import subprocess
from pathlib import Path

import pytest

from pi.protocol import MessageDecoder, ProtocolValidationError, encode_message
from pi.client import Client, ServerError
from pi.server import Server


def test_fragmented_coalesced_frames_and_validation():
    messages = [{"type": "hello", "version": 8}, {"type": "request", "id": "1",
        "target": {"serverId": str(uuid.uuid4())}, "call": {"value": [None, True, 2, -3, 1.5, "中文"]}}]
    encoded = b"".join(encode_message(m, "client") for m in messages)
    decoder = MessageDecoder("client")
    result = []
    for byte in encoded:
        result.extend(decoder.push(bytes([byte])))
    decoder.end()
    assert result == messages
    assert MessageDecoder("client").push(encoded) == messages
    with pytest.raises(ProtocolValidationError):
        encode_message({"type": "hello", "version": 8, "extra": 1}, "client")
    with pytest.raises(ProtocolValidationError):
        encode_message({**messages[1], "call": float("nan")}, "client")
    decoder = MessageDecoder("client", 10)
    with pytest.raises(ProtocolValidationError):
        decoder.push(struct.pack(">I", 11))
    with pytest.raises(ProtocolValidationError):
        decoder.push(b"")
    decoder = MessageDecoder("client")
    decoder.push(encoded[:5])
    with pytest.raises(ProtocolValidationError):
        decoder.end()


class Endpoint:
    def __init__(self, host, presentation=None):
        self.host, self.presentation = host, presentation
        self.released = False

    async def invoke_service(self, call, publish, cancelled):
        if call["member"] == "attach":
            await self.presentation.attach_session(call["args"][0])
        elif call["member"] == "detach":
            await self.presentation.detach_session()
        elif call["member"] == "wait":
            await cancelled.wait()
            self.host.cancelled.set()
        elif call["member"] == "observe":
            await publish("sub", {"value": 2})
            return {"value": 1}
        else:
            return call["args"]

    async def release(self):
        self.released = True


class Host:
    def __init__(self):
        self.endpoints = []
        self.cancelled = asyncio.Event()
        self.closed = False

    async def attach_client(self, presentation=None):
        endpoint = Endpoint(self, presentation)
        self.endpoints.append(endpoint)
        return endpoint

    async def resolve_session(self, identifier):
        return {"id": identifier}

    async def open_session(self, metadata):
        return self

    async def close(self):
        self.closed = True


def test_real_tcp_routing_cancellation_updates_and_reconnect():
    async def check():
        host = Host()
        server = Server(host)
        address = await server.start_tcp()
        client = Client(server.server_id)
        other = Client(server.server_id)
        target = {"serverId": server.server_id}
        try:
            await client.connect_tcp(*address[:2])
            await other.connect_tcp(*address[:2])
            def call(member, *args):
                return {"serviceId": "test", "member": member, "args": list(args)}
            assert await client.request(target, call("echo", "hi")) == ["hi"]
            await client.request(target, call("attach", "session"))
            old = dict(client.attachment)
            await other.request(target, call("attach", "session"))
            assert old != other.attachment
            with pytest.raises(ServerError) as error:
                await other.request(old, call("echo"))
            assert error.value.code == "invalid_target"
            assert await client.request(old, call("observe")) == {"value": 1}
            assert (await client.updates.get())["update"] == {"value": 2}
            with pytest.raises(asyncio.TimeoutError):
                await client.request(old, call("wait"), timeout=0.05)
            await asyncio.wait_for(host.cancelled.wait(), 2)
            await client.request(target, call("detach"))
            assert client.attachment is None
            with pytest.raises(ServerError):
                await client.request(old, call("echo"))
            await client.close()
            await client.connect_tcp(*address[:2])
            assert client.attachment is None
            assert await client.request(target, call("echo", 3)) == [3]
        finally:
            await client.close()
            await other.close()
            await server.close()
        assert host.closed and all(e.released for e in host.endpoints)
    asyncio.run(asyncio.wait_for(check(), 10))


def test_identity_mismatch():
    async def check():
        server = Server(Host())
        address = await server.start_tcp()
        client = Client(str(uuid.uuid4()))
        try:
            with pytest.raises(ConnectionError, match="identity"):
                await client.connect_tcp(*address[:2])
        finally:
            await client.close()
            await server.close()
    asyncio.run(check())


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_typescript_codec_interop(tmp_path):
    codec = (Path(__file__).resolve().parents[2] / "packages/protocol/src/cbor/index.ts").as_uri()
    script = tmp_path / "interop.mjs"
    script.write_text(f'''import {{ encodeCbor, decodeCbor }} from "{codec}";
const message = {{type:"hello",version:8}};
const encoded = Buffer.from(encodeCbor(message));
const decoded = decodeCbor(Buffer.from(process.argv[2], "hex").subarray(4));
if (JSON.stringify(decoded) !== JSON.stringify(message)) throw Error("decode mismatch");
const prefix = Buffer.alloc(4); prefix.writeUInt32BE(encoded.length);
process.stdout.write(Buffer.concat([prefix, encoded]).toString("hex"));
''', encoding="utf-8")
    message = {"type": "hello", "version": 8}
    result = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", str(script),
                             encode_message(message, "client").hex()], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert MessageDecoder("client").push(bytes.fromhex(result.stdout)) == [message]
