"""``pi mcp list`` and ``/mcp`` status display (B-CA-09).

The connection never lists resources; the display only marks the capability
(#10526): "offers resources" in text output and ``resources: true`` in JSON.
"""

import asyncio
import json

import httpx
import pytest

from pi.coding_agent.config import write_json
from pi.coding_agent.mcp import McpManager
from pi.coding_agent.mcp_cli import mcp_list


def _install_fake_mcp(monkeypatch, capabilities):
    async def send(self, request, **kwargs):
        url = str(request.url)
        if url != "https://example.test/mcp":
            return httpx.Response(404, request=request)
        try:
            body = json.loads(request.content)
        except ValueError:
            return httpx.Response(202, request=request)
        if "id" not in body:
            return httpx.Response(202, request=request)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": capabilities}
        else:
            result = {"tools": [{"name": "echo", "description": "Echo",
                                 "inputSchema": {"type": "object", "properties": {}}}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                              request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)


def _write_config(tmp_path, url="https://example.test/mcp"):
    write_json(tmp_path / "agent" / "mcp.json", {"mcpServers": {"dev": {"url": url}}})


def test_list_marks_resources_without_listing_them(tmp_path, monkeypatch, capsys):
    _install_fake_mcp(monkeypatch, capabilities={"resources": {}})
    _write_config(tmp_path)
    assert asyncio.run(mcp_list(str(tmp_path), str(tmp_path / "agent"))) == 0
    out = capsys.readouterr().out
    assert "dev: connected, 1 tool (direct, global)" in out
    assert "  offers resources" in out


def test_list_json_reports_resources_true(tmp_path, monkeypatch, capsys):
    _install_fake_mcp(monkeypatch, capabilities={"resources": {}})
    _write_config(tmp_path)
    assert asyncio.run(mcp_list(str(tmp_path), str(tmp_path / "agent"), json_output=True)) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["servers"][0]["resources"] is True
    assert parsed["servers"][0]["state"] == "connected"
    assert parsed["servers"][0]["tools"] == ["echo"]


def test_list_omits_resources_marker_without_capability(tmp_path, monkeypatch, capsys):
    _install_fake_mcp(monkeypatch, capabilities={})
    _write_config(tmp_path)
    assert asyncio.run(mcp_list(str(tmp_path), str(tmp_path / "agent"), json_output=True)) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert "resources" not in parsed["servers"][0]


def test_server_states_marks_resources(tmp_path, monkeypatch):
    _install_fake_mcp(monkeypatch, capabilities={"resources": {}})
    _write_config(tmp_path)
    config_manager = {}

    async def scenario():
        manager = McpManager(config_manager["config"])
        config_manager["manager"] = manager
        try:
            await manager.reload()
            assert manager.server_states()["dev"] == "connected, 1 tools · resources"
        finally:
            await manager.close()

    from pi.coding_agent.config import Configuration

    config_manager["config"] = Configuration(str(tmp_path), str(tmp_path / "agent"))
    asyncio.run(scenario())
