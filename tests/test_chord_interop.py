"""TypeScript interoperability tests: the real TS Server and Client (loaded from
the packages source tree via ``tests/interop/register.mjs``, no build) against
the Python stack over real loopback TCP.

Direction A: Python :class:`pi.client.Client` + chord binding against the TS server
(catalogue, subscribe, TS-encoded delta updates, methods, errors, attachments).
Direction B: TS ``Client`` against the Python durable server (subscribe to
``pi.session-directory`` with TS delta decoding, session create/remove updates).
Plus a byte-level chord delta codec cross-check in both directions.
"""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pi.chord import create_catalogue_call, parse_service_catalogue
from pi.chord.consumer import ClientServiceTransport, RemoteServiceBinding
from pi.client import Client, ServerError
from pi.delta import WireDecoder, WireEncoder
from pi.durable.server import DurableServerHost, SessionWorkerManager
from pi.server import Server

INTEROP = Path(__file__).resolve().parent / "interop"
NODE = shutil.which("node")

REQUIRES_NODE = pytest.mark.skipif(NODE is None, reason="Node unavailable")


async def start_node(script, *args):
    register = (INTEROP / "register.mjs").as_uri()
    process = await asyncio.create_subprocess_exec(
        NODE, "--experimental-strip-types", "--no-warnings", "--import", register, str(script), *args,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    return process


async def read_event(process, timeout=30):
    line = await asyncio.wait_for(process.stdout.readline(), timeout)
    if not line:
        stderr = await process.stderr.read() if process.stderr else b""
        raise AssertionError("Node peer closed stdout before the expected event; stderr:\n"
                             + stderr.decode(errors="replace"))
    return json.loads(line.decode())


async def send(process, command):
    process.stdin.write((json.dumps(command) + "\n").encode())
    await process.stdin.drain()


async def until(predicate, timeout=10.0):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


def management(member, *args):
    return {"serviceId": "pi.session-management", "member": member, "args": list(args)}


@REQUIRES_NODE
def test_python_client_against_typescript_server():
    async def check():
        process = await start_node(INTEROP / "ts_server.mjs")
        client = None
        binding = None
        transport = None
        try:
            hello = await read_event(process)
            assert hello["event"] == "listening"
            server_id, port = hello["serverId"], hello["port"]

            client = Client(server_id)
            await client.connect_tcp("127.0.0.1", port)
            target = {"serverId": server_id}

            catalogue = parse_service_catalogue(await client.request(target, create_catalogue_call()))
            assert {"serviceId": "pi.example", "mode": "singleton"} in catalogue
            assert {"serviceId": "pi.session-management", "mode": "singleton"} in catalogue

            transport = ClientServiceTransport(client, lambda: target)
            binding = RemoteServiceBinding(["pi.example", "pi.session-management"], transport)
            example = binding.use("pi.example")
            management_facade = binding.use("pi.session-management")
            await binding.ready()
            assert example.state.value == {"count": 0, "label": "alpha", "items": []}

            # TS-encoded delta updates (path interning on second use, short forms)
            # decode into the replica and apply in order.
            await send(process, {"op": "bump"})
            await send(process, {"op": "bump"})
            await until(lambda: (example.state.value or {}).get("count") == 2)
            await send(process, {"op": "set", "value": "beta"})
            await until(lambda: (example.state.value or {}).get("label") == "beta")
            await send(process, {"op": "push", "value": "x"})
            await until(lambda: (example.state.value or {}).get("items") == ["x"])

            # Method invocation, member errors, and unknown services.
            assert await example.echo("hello", 7) == ["hello", 7]
            with pytest.raises(ServerError) as member_error:
                await example.missing()
            assert member_error.value.code == "service_member_not_found"
            with pytest.raises(ServerError) as service_error:
                await client.request(target, {"serviceId": "pi.nope", "member": "echo", "args": []})
            assert service_error.value.code == "service_not_allowed"  # TS checks the allowlist first

            # Attachment envelopes precede the management response.
            await management_facade.attach("session-1")
            assert client.attachment["sessionId"] == "session-1"
            await management_facade.detach()
            await until(lambda: client.attachment is None)

            # Cancellation: the client aborts a slow call and the connection stays usable.
            with pytest.raises(asyncio.TimeoutError):
                await client.request(target, {"serviceId": "pi.example", "member": "wait",
                                              "args": [30000]}, timeout=0.3)
            assert await example.echo("after-cancel") == ["after-cancel"]

            await send(process, {"op": "stop"})
            # Drain the per-op change echoes, then confirm the shutdown event.
            while True:
                stopping = await read_event(process)
                if stopping["event"] == "stopping":
                    break
        finally:
            if binding is not None:
                await binding.close()
            if transport is not None:
                await transport.dispose()
            if client is not None:
                await client.close()
            if process.returncode is None:
                process.kill()
            await asyncio.wait_for(process.wait(), 10)
    asyncio.run(asyncio.wait_for(check(), 60))


@REQUIRES_NODE
def test_typescript_client_against_python_server(tmp_path):
    async def check():
        root = str(tmp_path / "sessions")
        manager = SessionWorkerManager(root, model_args=["--provider", "faux", "--model", "faux"])
        host = DurableServerHost(manager)
        server = Server(host)
        address = await server.start_tcp()
        host.server_id = server.server_id
        process = None
        try:
            process = await start_node(INTEROP / "ts_client.mjs", address[0], str(address[1]),
                                       server.server_id)
            connected = await read_event(process)
            assert connected["event"] == "connected"

            await send(process, {"op": "catalogue"})
            message = await read_event(process)
            assert message["event"] == "catalogue"
            assert {"serviceId": "pi.session-directory", "mode": "singleton"} in message["entries"]
            assert {"serviceId": "pi.session-management", "mode": "singleton"} in message["entries"]

            # The TS client's chord subscription decodes the Python-encoded snapshot.
            await send(process, {"op": "subscribe", "serviceId": "pi.session-directory",
                                 "mode": "singleton"})
            message = await read_event(process)
            assert message["event"] == "subscribed"
            snapshot = message["snapshot"]
            assert snapshot["serviceId"] == "pi.session-directory"
            state_member = snapshot["instances"][0]["members"][0]
            assert state_member["name"] == "state" and state_member["sequence"] == 0
            assert state_member["ops"][0][0] == "r"

            # Session create/remove flow through as decoded state updates.
            await send(process, {"op": "request", "call": management("create", {"cwd": str(tmp_path)})})
            created = await read_event(process)
            assert created["event"] == "result"
            summary = created["result"]
            assert summary["sessionId"] and summary["serverId"] == server.server_id
            update = await read_event(process)
            assert update["event"] == "update"
            assert update["update"]["type"] == "state"
            assert update["update"]["sequence"] == 1
            sessions = update["update"]["ops"][0][1]["sessions"]
            assert [item["sessionId"] for item in sessions] == [summary["sessionId"]]

            await send(process, {"op": "request", "call": management("remove", {"sessionId": summary["sessionId"]})})
            removed = await read_event(process)
            assert removed["event"] == "result"
            update = await read_event(process)
            assert update["update"]["ops"][0][1]["sessions"] == []

            # Error codes pass through to the TS client.
            await send(process, {"op": "request",
                                 "call": {"serviceId": "pi.nope", "member": "echo", "args": []}})
            message = await read_event(process)
            assert message["event"] == "error" and message["code"] == "service_not_found"

            await send(process, {"op": "stop"})
            stopping = await read_event(process)
            assert stopping["event"] == "stopping"
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await asyncio.wait_for(process.wait(), 10)
            await server.close()
            await manager.shutdown()
    asyncio.run(asyncio.wait_for(check(), 120))


@REQUIRES_NODE
def test_delta_wire_codec_matches_typescript(tmp_path):
    delta = (Path(__file__).resolve().parents[2] / "packages/chord/src/delta/index.ts").as_uri()
    script = tmp_path / "delta_codec.mjs"
    script.write_text(
        "import { encoder, decoder } from \"%s\";\n"
        "const batches = JSON.parse(Buffer.from(process.argv[2], \"hex\").toString(\"utf8\"));\n"
        "if (process.argv[3] === \"encode\") {\n"
        "  const enc = encoder();\n"
        "  process.stdout.write(JSON.stringify(batches.map((batch) => enc.encode(batch))));\n"
        "} else {\n"
        "  const dec = decoder();\n"
        "  process.stdout.write(JSON.stringify(batches.map((batch) => dec.decode(batch))));\n"
        "}\n" % delta,
        encoding="utf-8")
    batches = [
        [["s", ["v"], 1], ["s", ["v"], 2]],
        [["a", ["v"], "-tail"]],
        [["t", ["v"], 3]],
        [["p", ["items"], 0, 0, [1, 2]]],
        [["m", ["items"], [1, 0]]],
        [["r", {"v": 0}]],
        [["s", ["v"], 5]],
        [["d", ["items", 0]]],
    ]
    payload = json.dumps(batches, separators=(",", ":")).encode("utf-8").hex()

    # Python wire form == TS wire form for the same decoded batches (one encoder).
    result = subprocess.run([NODE, "--experimental-strip-types", "--no-warnings", str(script),
                             payload, "encode"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    python_encoder = WireEncoder()
    assert json.loads(result.stdout) == [python_encoder.encode(batch) for batch in batches]

    # The TS decoder resolves Python's wire form exactly like the Python decoder.
    result = subprocess.run([NODE, "--experimental-strip-types", "--no-warnings", str(script),
                             json.dumps([python_encoder.encode(batch) for batch in batches],
                                        separators=(",", ":")).encode("utf-8").hex(),
                             "decode"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    python_decoder = WireDecoder()
    assert json.loads(result.stdout) == [python_decoder.decode(batch) for batch in batches]
