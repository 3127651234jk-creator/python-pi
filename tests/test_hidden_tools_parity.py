"""Hidden tools stay out of prompt declarations while codemode keeps them (H01-H07)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from pi.codemode.tool import create_codemode_description, create_codemode_tool
from pi.coding_agent.cli import ToolSelection, apply_tool_selection, skill_file_read_tool
from pi.coding_agent.config import Configuration
from pi.coding_agent.resources import ResourceCatalog
from pi.coding_agent.system_prompt import SystemPromptOptions, build_system_prompt


def tool(name, description="", declared=True, guidelines=None):
    async def execute(call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=name)])

    return AgentTool(name=name, description=description, declared=declared,
                     prompt_snippet=description, prompt_guidelines=list(guidelines or []),
                     execute=execute)


def prompt(selected, hidden=(), snippets=None, guidelines=None):
    return build_system_prompt(SystemPromptOptions(
        cwd="/tmp/project", selected_tools=list(selected), hidden_tools=list(hidden),
        tool_snippets=dict(snippets or {}), tool_guidelines=dict(guidelines or {}),
        include_pi_documentation=False))


def test_h01_declared_reader_appears_in_tools_and_rules():
    text = prompt(["read", "bash"], snippets={"read": "Read files", "bash": "Run commands"},
                  guidelines={"read": ["Read a file before editing it."]})
    assert "- read: Read files" in text
    assert "Read a file before editing it." in text


def test_h02_hidden_reader_leaves_tools_and_rules():
    text = prompt(["read", "bash"], hidden=["read"],
                  snippets={"read": "Read files", "bash": "Run commands"},
                  guidelines={"read": ["Read a file before editing it."]})
    assert "read:" not in text
    assert "Read a file before editing it." not in text
    assert "- bash: Run commands" in text


def test_h02_skills_hint_uses_bash_when_read_is_hidden():
    agent = SimpleNamespace(tools=[tool("read", declared=False), tool("bash")])
    assert skill_file_read_tool(agent) == "bash"


def test_h03_hidden_readers_use_indirect_skills_hint(tmp_path):
    agent = SimpleNamespace(tools=[tool("read", declared=False), tool("bash", declared=False)])
    assert skill_file_read_tool(agent) == "indirect"

    config = Configuration(str(tmp_path), str(tmp_path / "user"))
    skill_dir = config.agent_dir / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\nDemo body\n", encoding="utf-8")
    resources = ResourceCatalog(config, home=tmp_path / "home")
    section = resources.system_prompt("indirect")
    assert "<available_skills>" in section
    assert "use read" not in section and "use bash" not in section
    assert "load its SKILL.md" in section


def test_h04_no_reader_omits_the_skills_hint():
    agent = SimpleNamespace(tools=[tool("edit")])
    assert skill_file_read_tool(agent) is None


def test_h03_codemode_reads_a_real_skill_file(tmp_path):
    from pi.coding_agent.tools.read import create_read_tool

    skill = tmp_path / "SKILL.md"
    skill.write_text("SKILL-BODY-MARKER\n", encoding="utf-8")
    read = create_read_tool(str(tmp_path))
    read.declared = False
    tools = [read]
    codemode = create_codemode_tool(tools)

    async def scenario():
        source = f"result = tools.read({{'path': {json.dumps(str(skill))}}})"
        return await codemode.execute("call", {"code": source}, None, None)

    result = asyncio.run(scenario())
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    assert "SKILL-BODY-MARKER" in text
    assert not result.is_error


def test_h04_codemode_cannot_call_a_disabled_tool():
    codemode = create_codemode_tool([])

    async def scenario():
        return await codemode.execute("call", {"code": "tools.read({'path': 'x'})"}, None, None)

    result = asyncio.run(scenario())
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    assert result.is_error
    assert "read" in text


def test_h05_hidden_guidelines_move_to_codemode_only():
    marker = "Only for codemode: prefer the indexed path."
    hidden = tool("indexed", "Indexed reader", declared=False, guidelines=[marker])
    text = prompt(["read", "indexed"], hidden=["indexed"],
                  snippets={"read": "Read files"},
                  guidelines={"indexed": [marker]})
    assert marker not in text

    description = create_codemode_description([hidden])
    assert description.count(marker) == 1


def test_h06_codemode_description_follows_tool_changes():
    first = tool("first", "First tool")
    tools = [first]
    codemode = create_codemode_tool(tools)
    assert "tools.first" in codemode.description

    second = tool("second", "Second tool")
    tools.append(second)
    codemode._codemode_refresh()
    assert "tools.second" in codemode.description

    tools.remove(second)
    codemode._codemode_refresh()
    assert "tools.second" not in codemode.description


def test_h06_codemode_budget_update_rebuilds_description():
    codemode = create_codemode_tool([tool("first", "First tool")],
                                    settings={"inline_budget": 1000})
    assert "tools.first" in codemode.description
    codemode._codemode_update_settings({"inline_budget": 0})
    assert "tools.first" not in codemode.description
    codemode._codemode_update_settings({"inline_budget": 1000})
    assert "tools.first" in codemode.description


def test_h06_selection_change_rebuilds_prompt():
    read, bash = tool("read", "read"), tool("bash", "bash")
    agent = SimpleNamespace(tools=[read, bash], system_prompt="", _resources=None,
                            messages=[SimpleNamespace(role="system", content="", sections=None)])
    agent._system_prompt_options = SystemPromptOptions(
        cwd="/tmp/project", selected_tools=["read", "bash"],
        tool_snippets={"read": "Read files", "bash": "Run commands"},
        include_pi_documentation=False)
    agent._tool_selection = None
    apply_tool_selection(agent)
    assert "read:" in agent.messages[0].content
    assert "bash:" in agent.messages[0].content

    read.declared = False
    apply_tool_selection(agent)
    assert "read:" not in agent.messages[0].content
    assert "bash:" in agent.messages[0].content
    assert agent._system_prompt_options.hidden_tools == ["read"]


def test_h07_no_tools_keeps_everything_out():
    agent = SimpleNamespace(tools=[tool("read"), tool("bash")], system_prompt="", _resources=None,
                            messages=[SimpleNamespace(role="system", content="", sections=None)])
    agent._system_prompt_options = SystemPromptOptions(
        cwd="/tmp/project", tool_snippets={"read": "Read files", "bash": "Run commands"},
        include_pi_documentation=False)
    agent._tool_selection = ToolSelection([], [])
    apply_tool_selection(agent)
    assert agent.tools == []
    assert "read:" not in agent.messages[0].content
    assert "bash:" not in agent.messages[0].content


def test_h07_hidden_does_not_re_add_excluded_tools():
    selection = ToolSelection(None, ["read"])
    assert not selection.keeps("read")
    assert selection.keeps("bash")

    wildcard = ToolSelection(["mcp__radius__*"], [])
    assert wildcard.keeps("mcp__radius__search")
    assert wildcard.declares("mcp__radius__search")
    assert not wildcard.keeps("read")
