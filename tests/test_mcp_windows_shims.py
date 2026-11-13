"""Windows npm-style batch shims must work as stdio MCP commands."""

import asyncio
import os
import sys

import pytest

from pi.coding_agent.mcp import McpClient


@pytest.mark.skipif(sys.platform != "win32", reason="Windows command shims")
@pytest.mark.parametrize("command", ["npx", "npm", "npx.cmd", "absolute", "relative", "batch"])
def test_stdio_cmd_shim(tmp_path, command):
    directory = tmp_path / "shim directory"
    directory.mkdir()
    server = directory / "server.py"
    server.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    if 'id' not in req: continue\n"
        "    result = {'protocolVersion': '2025-03-26'}\n"
        "    if req['method'] == 'tools/list': result = {'tools': []}\n"
        "    if req['method'] == 'echo': result = {'args': sys.argv[1:]}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    for name in ("npx", "npm"):
        (directory / (name + ".cmd")).write_text(
            f'@echo off\n"{sys.executable}" "{server}" %*\n', encoding="utf-8"
        )
    (directory / "server.bat").write_bytes((directory / "npx.cmd").read_bytes())
    commands = {"absolute": str(directory / "npx.cmd"), "relative": "shim directory/npx.cmd",
                "batch": str(directory / "server.bat")}
    args = ["-y", "@example/mcp", "argument with spaces", "", "a&b", "x|y", "(value)",
            'embedded"quote', "trailing\\", "%PI_MCP_TEST_LITERAL%", "!literal!", "a^b"]
    settings = {"command": commands.get(command, command),
                    "args": args, "env": {"PATH": str(directory) + os.pathsep + os.environ["PATH"],
                                          "PI_MCP_TEST_LITERAL": "must-not-expand"},
                "timeout": 5}

    async def scenario():
        client = McpClient("shim", settings, tmp_path)
        try:
            await client.connect()
            assert (await client.request("echo", {}))["args"] == args
        finally:
            await client.close()

    asyncio.run(scenario())
