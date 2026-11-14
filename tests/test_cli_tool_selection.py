"""CLI tool selection: --tools / --exclude-tools / --no-tools / --no-mcp.

Port of the v1.0.4 coding-agent change (04b97ef00): `*` patterns, MCP tools stay
registered with --tools unless an entry starts with mcp__, --no-tools removes
everything, and sync_mcp_tools re-applies the selection on every tool refresh.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.agent.types import AgentTool
from pi.coding_agent.cli import (
    ToolSelection,
    apply_tool_selection,
    build_parser,
    skill_file_read_tool,
    sync_mcp_tools,
    tool_selection_from_args,
)


def tool(name: str) -> AgentTool:
    return AgentTool(name=name, label=name, description=name, parameters={}, execute=None)


def select(argv: list) -> ToolSelection:
    return tool_selection_from_args(build_parser().parse_args(argv))


def agent_with(*names, selection=None) -> SimpleNamespace:
    return SimpleNamespace(tools=[tool(name) for name in names], _tool_selection=selection)


def names_of(agent) -> list:
    return [item.name for item in agent.tools]


def test_parser_accepts_tool_flags():
    args = build_parser().parse_args(["--tools", "read,codemode"])
    assert args.tools == "read,codemode" and not args.no_tools and not args.no_mcp
    args = build_parser().parse_args(["-t", "read", "-xt", "bash", "-nt", "--no-mcp"])
    assert args.tools == "read" and args.exclude_tools == "bash"
    assert args.no_tools and args.no_mcp
    assert tool_selection_from_args(build_parser().parse_args([])) is None


def test_tools_allowlist_keeps_matches_and_mcp_tools():
    selection = select(["--tools", "read,codemode"])
    assert selection.keeps("read") and selection.keeps("codemode")
    assert not selection.keeps("bash")
    # The allowlist names no MCP tool, so MCP tools stay usable.
    assert selection.keeps("mcp__dev__echo")
    assert selection.keeps("read_mcp_resource")


def test_tools_allowlist_declares_kept_but_unnamed_mcp_tools_as_undeclared():
    selection = select(["--tools", "read,codemode"])
    assert selection.declares("read") and selection.declares("codemode")
    # Kept for codemode, but the model does not see their declarations.
    assert not selection.declares("mcp__dev__echo")
    assert not selection.declares("read_mcp_resource")


def test_named_mcp_tools_are_declared_and_unmatched_ones_are_dropped():
    selection = select(["--tools", "read,mcp__radius__*"])
    assert selection.declares("read")
    assert selection.declares("mcp__radius__search")
    # Naming MCP tools filters MCP: the other server is not even registered.
    assert not selection.keeps("mcp__other__search")


def test_exclude_tools_alone_declares_every_retained_tool():
    selection = select(["--exclude-tools", "bash"])
    assert selection.declares("read")
    assert selection.declares("mcp__dev__echo")


def test_tools_naming_mcp_filters_other_mcp_tools():
    selection = select(["--tools", "read,mcp__radius__search"])
    assert selection.keeps("read")
    assert selection.keeps("mcp__radius__search")
    assert not selection.keeps("mcp__other__search")
    assert not selection.keeps("mcp__radius__other")

    pattern = select(["--tools", "read,mcp__radius__*"])
    assert pattern.keeps("mcp__radius__search")
    # The pattern covers every tool of that server.
    assert pattern.keeps("mcp__radius__other")
    assert not pattern.keeps("mcp__other__search")


def test_exclude_tools_applies_to_everything_including_mcp():
    selection = select(["--exclude-tools", "bash,mcp__radius__*"])
    assert not selection.keeps("bash")
    assert not selection.keeps("mcp__radius__search")
    assert selection.keeps("mcp__other__search")
    assert selection.keeps("read")


def test_no_tools_removes_everything_including_mcp():
    selection = select(["--no-tools"])
    assert not selection.keeps("read")
    assert not selection.keeps("mcp__dev__echo")
    assert not selection.keeps("read_mcp_resource")


def test_exclude_tools_wins_over_tools():
    selection = select(["--tools", "read,bash", "--exclude-tools", "bash"])
    assert not selection.keeps("bash")
    assert selection.keeps("read")


def test_apply_tool_selection_filters_agent_tools():
    agent = agent_with("read", "bash", "mcp__dev__echo", selection=select(["--tools", "read,codemode"]))
    apply_tool_selection(agent)
    assert names_of(agent) == ["read", "mcp__dev__echo"]

    no_flags = agent_with("read", "bash")
    apply_tool_selection(no_flags)
    assert names_of(no_flags) == ["read", "bash"]


def test_apply_tool_selection_marks_kept_unnamed_mcp_tools_undeclared():
    agent = agent_with("read", "codemode", "mcp__dev__echo",
                       selection=select(["--tools", "read,codemode"]))
    apply_tool_selection(agent)
    assert {item.name: item.declared for item in agent.tools} == {
        "read": True, "codemode": True, "mcp__dev__echo": False}


def test_sync_mcp_tools_honors_selection():
    agent = agent_with("codemode", selection=select(["--tools", "codemode"]))
    mcp = SimpleNamespace(tools=[tool("mcp__dev__echo"), tool("read_mcp_resource")])
    sync_mcp_tools(agent, mcp)
    # MCP tools stay registered when the allowlist does not name MCP tools.
    assert names_of(agent) == ["codemode", "mcp__dev__echo", "read_mcp_resource"]
    # Registered for codemode, but not declared to the model.
    assert {item.name: item.declared for item in agent.tools} == {
        "codemode": True, "mcp__dev__echo": False, "read_mcp_resource": False}


def test_sync_mcp_tools_pattern_keeps_only_matching_server():
    agent = agent_with("codemode", selection=select(["--tools", "codemode,mcp__radius__*"]))
    mcp = SimpleNamespace(tools=[tool("mcp__radius__search"), tool("mcp__dev__echo")])
    sync_mcp_tools(agent, mcp)
    assert names_of(agent) == ["codemode", "mcp__radius__search"]


def test_sync_mcp_tools_without_selection_keeps_everything():
    agent = agent_with("codemode")
    mcp = SimpleNamespace(tools=[tool("mcp__dev__echo")])
    sync_mcp_tools(agent, mcp)
    assert names_of(agent) == ["codemode", "mcp__dev__echo"]


def test_sync_mcp_tools_none_manager_is_noop():
    agent = agent_with("read", selection=select(["--tools", "read"]))
    sync_mcp_tools(agent, None)
    assert names_of(agent) == ["read"]


def test_skill_file_read_tool_follows_declared_readers():
    assert skill_file_read_tool(agent_with("read", "bash")) == "read"
    assert skill_file_read_tool(agent_with("bash", "grep")) == "bash"
    assert skill_file_read_tool(agent_with("codemode")) is None


def test_extension_refresh_tools_honors_selection(tmp_path):
    """Tools an extension registers after setup also run through the --tools selection."""
    from pi.agent.agent import Agent
    from pi.ai.models import resolve_model
    from pi.coding_agent.config import Configuration
    from pi.coding_agent.extensions import ExtensionManager

    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    agent = Agent(resolve_model("faux"))
    agent._tool_selection = select(["--tools", "read"])
    manager = ExtensionManager(config, agent)
    manager.context.tool_catalog["codemode"] = tool("codemode")
    agent.tools[:] = [tool("read")]
    manager.refresh_tools()
    assert names_of(agent) == ["read"]
