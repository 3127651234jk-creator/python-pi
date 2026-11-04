"""Cumulative token and cost accounting.

Port of ``packages/coding-agent/src/core/usage-totals.ts`` (totals and the
per-model cost breakdown) plus the ``getSessionStats`` summary. Totals are
derived from assistant messages on the transcript, so they survive resume and
need no sidecar state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from ..ai.types import AssistantMessage, ToolCall, ToolResultMessage, Usage


@dataclass
class ModelUsage:
    requests: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0

    def add(self, usage: Usage, cost: float) -> None:
        self.requests += 1
        self.input += usage.input
        self.output += usage.output
        self.cache_read += usage.cache_read
        self.cache_write += usage.cache_write
        self.cost += cost


@dataclass
class UsageTotals:
    requests: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    by_model: Dict[str, ModelUsage] = field(default_factory=dict)

    def add(self, usage: Usage, cost: float, model_key: str) -> None:
        self.requests += 1
        self.input += usage.input
        self.output += usage.output
        self.cache_read += usage.cache_read
        self.cache_write += usage.cache_write
        self.cost += cost
        entry = self.by_model.setdefault(model_key, ModelUsage())
        entry.add(usage, cost)


def _usage_cost(usage: Usage, model, registry) -> float:
    """Recorded cost when present, else recompute from the model catalog."""
    recorded = usage.cost.get("total", 0.0) if isinstance(usage.cost, dict) else 0.0
    if recorded:
        return recorded
    lookup = getattr(model, "calculate_cost", None)
    if lookup is not None:
        return lookup(usage).get("total", 0.0)
    return 0.0


def usage_cost_breakdown(messages: Iterable, registry=None) -> UsageTotals:
    """Total usage grouped by provider/model; unpriced calls fall back to the catalog."""
    totals = UsageTotals()
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        usage = message.usage
        if usage.total_tokens == 0 and not usage.cost.get("total"):
            continue
        key = f"{message.provider}/{message.response_model or message.model}"
        model = None
        if registry is not None:
            model = registry.models.get(key) or registry.models.get(f"{message.provider}/{message.model}")
        totals.add(usage, _usage_cost(usage, model, registry), key)
    return totals


def context_usage(messages: Iterable, model) -> Optional[dict]:
    """Last assistant turn's token footprint against the context window."""
    last_usage = None
    for message in messages:
        if isinstance(message, AssistantMessage) and message.stop_reason != "error":
            last_usage = message.usage
    if last_usage is None or last_usage.total_tokens == 0:
        return None
    window = getattr(model, "context_window", 0) or 0
    percent = (last_usage.total_tokens / window * 100.0) if window else 0.0
    return {"tokens": last_usage.total_tokens, "context_window": window, "percent": round(percent, 1)}


def format_usage(totals: UsageTotals, context: Optional[dict] = None) -> str:
    lines = [
        "requests: {}  input: {}  output: {}  cache read: {}  cache write: {}".format(
            totals.requests, f"{totals.input:,}", f"{totals.output:,}",
            f"{totals.cache_read:,}", f"{totals.cache_write:,}"),
        "total tokens: {}  cost: ${:.4f}".format(f"{totals.input + totals.output + totals.cache_read + totals.cache_write:,}", totals.cost),
    ]
    if context is not None:
        lines.append("context: {} / {} tokens ({:.1f}%)".format(
            f"{context['tokens']:,}", f"{context['context_window']:,}", context["percent"]))
    if totals.by_model:
        lines.append("by model:")
        for key, entry in sorted(totals.by_model.items(), key=lambda item: item[1].cost, reverse=True):
            lines.append("  {}: {} requests, in {}, out {}, cache {}/{}, ${:.4f}".format(
                key, entry.requests, f"{entry.input:,}", f"{entry.output:,}",
                f"{entry.cache_read:,}", f"{entry.cache_write:,}", entry.cost))
    return "\n".join(lines)


def session_stats(messages: list, model=None, session_file: Optional[str] = None) -> dict:
    """Port of AgentSession.getSessionStats()."""
    user = sum(1 for m in messages if getattr(m, "role", None) == "user")
    assistant = sum(1 for m in messages if isinstance(m, AssistantMessage))
    tool_calls = sum(1 for m in messages if isinstance(m, AssistantMessage)
                     for block in m.content if isinstance(block, ToolCall))
    tool_results = sum(1 for m in messages if isinstance(m, ToolResultMessage))
    totals = usage_cost_breakdown(messages)
    return {
        "session_file": session_file,
        "user_messages": user,
        "assistant_messages": assistant,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "total_messages": len(messages),
        "tokens": {"input": totals.input, "output": totals.output,
                   "cache_read": totals.cache_read, "cache_write": totals.cache_write,
                   "total": totals.input + totals.output + totals.cache_read + totals.cache_write},
        "cost": totals.cost,
        "context_usage": context_usage(messages, model) if model is not None else None,
    }
