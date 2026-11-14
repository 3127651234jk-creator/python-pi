"""BM25 discovery and branch-local activation of deferred tools."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, List

from ..agent.types import AgentTool, AgentToolResult
from ..ai.types import TextContent

STORE_TYPE = "tool_search.loaded"
STOP_WORDS = set("a an and are as at be by for from in is it of on or that the this to with".split())


def tokenize(text: str) -> List[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    result = []
    for term in re.split(r"[^a-z0-9]+", text.lower()):
        if not term or term in STOP_WORDS:
            continue
        if len(term) > 4 and term.endswith("ies"):
            term = term[:-3] + "y"
        elif len(term) > 4 and re.search(r"(ches|shes|sses|xes|zes)$", term):
            term = term[:-2]
        elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            term = term[:-1]
        result.append(term)
    return result


def schema_text(schema: object) -> str:
    if not isinstance(schema, dict):
        return ""
    parts = [schema.get("description", "")]
    for name, child in (schema.get("properties") or {}).items():
        parts.extend([name, schema_text(child)])
    parts.append(schema_text(schema.get("items")))
    for key in ("anyOf", "oneOf", "allOf"):
        parts.extend(schema_text(child) for child in schema.get(key, []))
    return " ".join(part for part in parts if isinstance(part, str))


def rank_tools(query: str, tools: List[AgentTool], limit: int = 8) -> List[AgentTool]:
    terms = set(tokenize(query))
    if not terms or not tools or limit <= 0:
        return []
    counts = [Counter(tokenize(f"{tool.name} {tool.name.replace('_', ' ')} {tool.description} "
                               + schema_text(tool.parameters))) for tool in tools]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(tools) or 1
    scores = [0.0] * len(tools)
    for term in terms:
        frequency = sum(term in count for count in counts)
        idf = math.log(1 + (len(tools) - frequency + 0.5) / (frequency + 0.5))
        for index, count in enumerate(counts):
            tf = count[term]
            if tf:
                scores[index] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[index] / average))
    return [tools[index] for index in sorted(range(len(tools)), key=lambda i: -scores[i])
            if scores[index] > 0][:limit]


def create_tool_search_tool(tools: List[AgentTool], activate: Callable[[List[str]], None]) -> AgentTool:
    async def execute(call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
        query, limit = params.get("query"), params.get("limit", 8)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if signal:
            signal.throw_if_aborted()
        candidates = [tool for tool in tools if not tool.declared
                      and tool.exposure in ("deferred", "codemode")]
        matches = rank_tools(query, candidates, limit)
        loaded = [tool.name for tool in matches]
        if loaded:
            activate(loaded)
        text = "No matching tools found."
        if matches:
            text = f"Loaded {len(matches)} tools. They are available from your next call:\n" + "\n".join(
                f"- {tool.name}: {tool.description.strip().splitlines()[0] if tool.description.strip() else ''}"
                for tool in matches)
        return AgentToolResult(content=[TextContent(text=text)], details={"loaded": loaded})

    return AgentTool(name="tool_search", description="Search deferred tool metadata with BM25 and load matching "
                     "tools for the next model call. Use this tool for MCP tool discovery.",
                     parameters={"type": "object", "properties": {"query": {"type": "string"},
                                 "limit": {"type": "integer", "minimum": 1}}, "required": ["query"]},
                     exposure="model-only", execute=execute)
