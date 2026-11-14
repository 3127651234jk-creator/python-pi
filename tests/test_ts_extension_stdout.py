"""Non-protocol JSON printed by extensions must not stop the bridge."""

import asyncio
from types import SimpleNamespace

import pytest

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.coding_agent.extensions import ExtensionAPI, ExtensionContext
from pi.coding_agent.ts_extension import TsExtensionBridge, node_command


@pytest.mark.parametrize("line", [b"42", b"null", b"true", b"false", b'"log"', b"[]"])
def test_reader_skips_json_logs_and_delivers_next_reply(tmp_path, line):
    async def scenario():
        api = ExtensionAPI(tmp_path / "extension.mjs", ExtensionContext(Agent(resolve_model("faux")), tmp_path))
        bridge = TsExtensionBridge(tmp_path / "extension.mjs", api, [])
        reader = asyncio.StreamReader()
        bridge.process = SimpleNamespace(stdout=reader)
        reply = asyncio.get_running_loop().create_future()
        bridge.pending[1] = reply
        reader.feed_data(line + b'\n{"id":1,"result":"ok"}\n')
        reader.feed_eof()
        await bridge._read_loop()
        assert reply.result() == "ok"
        assert not bridge.pending

    asyncio.run(scenario())


def test_node_console_json_logs_during_setup_and_command(tmp_path):
    argv = node_command()
    if argv is None:
        pytest.skip("node unavailable")
    path = tmp_path / "logging.mjs"
    path.write_text('''export function setup(pi) {
        console.log(42); console.log(null); console.log(true);
        pi.registerCommand("echo", {handler: () => {
            console.log(42); console.log(null); console.log(true);
            return "ok";
        }});
    }''', encoding="utf-8")

    async def scenario():
        api = ExtensionAPI(path, ExtensionContext(Agent(resolve_model("faux")), tmp_path))
        bridge = TsExtensionBridge(path, api, argv)
        try:
            await asyncio.wait_for(bridge.start(), 10)
            for _ in range(2):
                assert await asyncio.wait_for(api.commands["echo"][0](""), 5) == "ok"
            assert not bridge.reader_task.done()
        finally:
            await bridge.close()

    asyncio.run(scenario())
