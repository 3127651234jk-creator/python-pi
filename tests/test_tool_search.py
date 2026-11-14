import asyncio
from types import SimpleNamespace

import pytest

from pi.agent.types import AgentTool
from pi.coding_agent.cli import ToolSelection, apply_tool_selection
from pi.coding_agent.tool_search import create_tool_search_tool, rank_tools, tokenize


def test_bm25_schema_and_stemming():
    tools = [AgentTool(name="tickets", description="List issues"),
             AgentTool(name="lookup", description="", parameters={"properties": {"repository": {"description": "Search branches"}}})]
    assert rank_tools("issue", tools) == [tools[0]]
    assert rank_tools("repository", tools) == [tools[1]]
    assert tokenize("getHTTPResponses and searches") == ["get", "http", "response", "search"]


def test_search_loads_next_declaration_and_selection_cannot_restore_excluded():
    tool = AgentTool(name="mcp__demo__issues", description="", exposure="deferred", declared=False)
    agent = SimpleNamespace(tools=[tool], _tool_selection=ToolSelection(["tool_search"], []), _loaded_tools=[])
    def activate(names):
        agent._loaded_tools = names
        apply_tool_selection(agent)
    search = create_tool_search_tool(agent.tools, activate)
    result = asyncio.run(search.execute("search", {"query": "issues"}))
    assert result.details == {"loaded": [tool.name]}
    assert tool.declared
    assert asyncio.run(search.execute("search", {"query": "issues"})).details == {"loaded": []}
    agent._tool_selection = ToolSelection(["tool_search"], [tool.name])
    apply_tool_selection(agent)
    assert agent.tools == []


@pytest.mark.parametrize("params", [{"query": " "}, {"query": "issue", "limit": True},
                                   {"query": "issue", "limit": 0}])
def test_invalid_search(params):
    with pytest.raises(ValueError):
        asyncio.run(create_tool_search_tool([], lambda names: None).execute("search", params))
