"""Prompt cache warming (F-CA-13, port of packages/coding-agent/src/core/cache-warmer.ts).

When enabled, the last real provider request is replayed with a one-token
output cap shortly before the prompt cache TTL expires, refreshing the
provider-side cache so the next real request still bills at cache-read rates.
This module owns the state machine: capture, scheduling, the economic
decision, replay, and cancellation. See docs/specs/prompt-cache-warming.md.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from ..ai.models import Model
from ..ai.types import AssistantMessage, DoneEvent, TranscriptContext
from ..util import AbortSignal

# TTL must exceed this margin before a warm is worth scheduling.
MIN_TTL_SECONDS = 10
# Scheduled delay: fire at 90% of the TTL, but never within 10s of expiry.
WARM_FRACTION = 0.9
# A replay that fires after the planned time plus half the remaining margin is
# too late to help and is dropped instead of sent.
LATE_SLACK_FRACTION = 0.5
# Warm replaying is capped from the first real request, not reset by refreshes.
MAX_ACTIVE_SECONDS = 60 * 60
MAX_IDLE_SECONDS = 30 * 60
# The warm decision: p(more prompts coming) times the miss cost must beat the
# warm cost by this margin (USD) before a replay is worth sending.
DECISION_THRESHOLD_USD = 0.05
IDLE_CONTINUATION_PROBABILITY = 0.15

WARM_OUTPUT_TOKENS = 1


@dataclass
class CacheWarmRequest:
    """Immutable snapshot of one real provider request (spec section 4)."""

    model: Model
    context: TranscriptContext
    options: Any
    request_id: int
    generation: int
    started_at: float  # monotonic seconds of the real request's start


@dataclass
class CacheWarmingDecision:
    phase: str
    warm_cost: float = 0.0
    miss_cost: float = 0.0
    continuation_probability: float = 0.0
    expected_savings: float = 0.0
    economics_available: bool = False
    action: str = "stop"


@dataclass
class CacheWarmingStatus:
    state: str  # inactive | scheduled | refreshing
    reason: Optional[str] = None
    next_warm_at: Optional[float] = None
    decision: Optional[CacheWarmingDecision] = None


def _per_million(costs: Any, key: str) -> Optional[float]:
    value = getattr(costs, key, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _cache_ttl_seconds(model: Model, retention: Optional[str]) -> Optional[int]:
    """The model's advertised cache TTL for the retention, from metadata;
    an unknown retention or missing metadata stops warming."""
    metadata = model.prompt_cache
    if not metadata or not retention:
        return None
    value = metadata.get(retention)
    if isinstance(value, bool) or not isinstance(value, int) or value <= MIN_TTL_SECONDS:
        return None
    return value


def replay_safe(options: Any, model: Model) -> Optional[str]:
    """Why the request cannot be replayed with a one-token cap, or None when safe.

    Budget-based thinking derives its thinking parameters from the output
    ceiling, so shrinking the cap would change the cached prefix (spec section
    6); adaptive-thinking effort models keep their thinking parameters.
    """
    reasoning = getattr(options, "reasoning", None)
    if reasoning not in (None, "off") and not (
            "haiku-5" in model.id.lower()
            or any(marker in model.id.lower() for marker in ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5"))):
        return "budget-based thinking derives its parameters from the output cap"
    return None


def decide(model: Model, prompt_tokens: int, idle: bool) -> CacheWarmingDecision:
    """The economic decision: warm when p * miss_cost - warm_cost clears the
    threshold; zero prices are not mistaken for free (unknown stops)."""
    phase = "idle" if idle else "streaming"
    input_price = _per_million(model.costs, "input")
    cache_read_price = _per_million(model.costs, "cache_read")
    cache_write_price = _per_million(model.costs, "cache_write")
    output_price = _per_million(model.costs, "output")
    if input_price is None or output_price is None or cache_read_price is None:
        return CacheWarmingDecision(phase=phase, economics_available=False, action="stop")
    hit = prompt_tokens * cache_read_price / 1_000_000
    miss = prompt_tokens * (cache_write_price if cache_write_price else input_price) / 1_000_000
    warm_cost = hit + output_price * WARM_OUTPUT_TOKENS / 1_000_000
    miss_cost = max(0.0, miss - hit)
    probability = IDLE_CONTINUATION_PROBABILITY if idle else 1.0
    expected_savings = probability * miss_cost - warm_cost
    action = "warm" if expected_savings >= DECISION_THRESHOLD_USD else "stop"
    return CacheWarmingDecision(
        phase=phase,
        warm_cost=warm_cost, miss_cost=miss_cost, continuation_probability=probability,
        expected_savings=expected_savings,
        economics_available=prompt_tokens > 0 and (hit > 0 or miss > 0), action=action)


class CacheWarmer:
    """Schedules and runs one cache-warm replay for the latest real request."""

    def __init__(self, stream: Callable[..., Any], get_session_name: Callable[[], Optional[str]] = lambda: None,
                 clock: Callable[[], float] = time.monotonic,
                 decision_hook: Optional[Callable[[CacheWarmingDecision], Any]] = None) -> None:
        self.stream = stream
        self.get_session_name = get_session_name
        self.clock = clock
        self.decision_hook = decision_hook
        self.mode: str = "streaming"
        self.status: CacheWarmingStatus = CacheWarmingStatus(state="inactive", reason="no request")
        self.request: Optional[CacheWarmRequest] = None
        # Phase of the latest run (TS ActiveRun.phase): a run is streaming
        # until its agent_end arrives, then idle.
        self.phase: str = "streaming"
        self.started_at: Optional[float] = None
        self._generation = 0
        self._timer: Optional[Any] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._closed = False

    # -- capture -------------------------------------------------------------

    def capture(self, model: Model, context: TranscriptContext, options: Any) -> None:
        """Record the latest real request; a new request cancels any pending
        refresh (the real request has priority over warming)."""
        if self._closed:
            return
        self._generation += 1
        self.request = CacheWarmRequest(
            model=model, context=copy.deepcopy(context), options=copy.deepcopy(options),
            request_id=self._generation, generation=self._generation, started_at=self.clock())
        # Each real request starts a new run with its own lifetime window.
        self.phase = "streaming"
        self.started_at = self.request.started_at
        self._cancel_timer()
        self._cancel_refresh()
        self._schedule()

    # -- lifecycle hooks -----------------------------------------------------

    def on_agent_settled(self) -> None:
        """The run ended: streaming mode stops, idle mode applies its shorter cap."""
        if self.mode == "streaming":
            self._stop("streaming mode does not keep the cache warm after the run")
            return
        self.phase = "idle"
        if self.started_at is not None and self.clock() - self.started_at > MAX_IDLE_SECONDS:
            self._stop("idle warm window expired")
            return
        self._schedule()

    def cancel(self) -> None:
        self._cancel_timer()
        self._cancel_refresh()
        self.request = None
        self.phase = "streaming"
        self.started_at = None
        self.status = CacheWarmingStatus(state="inactive", reason="cancelled")

    async def close(self) -> None:
        self._closed = True
        refresh = self._refresh_task
        self.cancel()
        if refresh is not None:
            await asyncio.gather(refresh, return_exceptions=True)

    # -- scheduling ----------------------------------------------------------

    def _stop(self, reason: str) -> None:
        self._cancel_timer()
        self._cancel_refresh()
        self.request = None
        self.status = CacheWarmingStatus(state="inactive", reason=reason)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _cancel_refresh(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None

    def _schedule(self) -> None:
        if self._closed or self.request is None:
            return
        retention = getattr(self.request.options, "cache_retention", None)
        ttl = _cache_ttl_seconds(self.request.model, retention)
        if ttl is None:
            self.status = CacheWarmingStatus(state="inactive", reason="unknown or too-short cache TTL")
            return
        unsafe = replay_safe(self.request.options, self.request.model)
        if unsafe is not None:
            self.status = CacheWarmingStatus(state="inactive", reason=unsafe)
            return
        prompt_tokens = self._prompt_tokens()
        if prompt_tokens is None:
            self.status = CacheWarmingStatus(state="inactive", reason="no usable usage on the last response")
            return
        decision = decide(self.request.model, prompt_tokens, idle=self.phase == "idle")
        if self.decision_hook is not None:
            outcome = self.decision_hook(decision)
            if asyncio.iscoroutine(outcome):
                asyncio.ensure_future(outcome)
        if decision.action != "warm":
            self.status = CacheWarmingStatus(state="inactive", reason="economics", decision=decision)
            return
        delay_ms = max(1, int(min(WARM_FRACTION * ttl, ttl - MIN_TTL_SECONDS) * 1000))
        due = self.clock() + delay_ms / 1000
        self.status = CacheWarmingStatus(state="scheduled", next_warm_at=due, decision=decision)
        self._cancel_timer()
        self._timer = asyncio.get_event_loop().call_later(delay_ms / 1000, self._fire, delay_ms / 1000, ttl)

    def _prompt_tokens(self) -> Optional[int]:
        context = self.request.context
        for message in reversed(context.messages):
            if getattr(message, "role", None) == "assistant":
                usage = getattr(message, "usage", None)
                if usage is None:
                    continue
                total = (usage.input + usage.cache_read + usage.cache_write + usage.cache_write_1h)
                if total > 0:
                    return total
                return None
        return None

    def _fire(self, planned_delay: float, ttl: int) -> None:
        if self._closed or self.request is None:
            return
        # A late timer (event loop stalled past the deadline) is dropped.
        slack = LATE_SLACK_FRACTION * max(0.0, ttl - planned_delay) + MIN_TTL_SECONDS * 0
        if self.clock() > self.status.next_warm_at + max(slack, MIN_TTL_SECONDS * LATE_SLACK_FRACTION):
            self._stop("warm window passed while the loop was stalled")
            return
        window = MAX_IDLE_SECONDS if self.phase == "idle" else MAX_ACTIVE_SECONDS
        if self.clock() - self.request.started_at > window:
            self._stop("warm window expired")
            return
        self._cancel_refresh()
        self.status = CacheWarmingStatus(state="refreshing", next_warm_at=self.status.next_warm_at,
                                         decision=self.status.decision)
        self._refresh_task = asyncio.ensure_future(self._refresh())

    # -- replay --------------------------------------------------------------

    async def _refresh(self) -> None:
        request = self.request
        generation = self._generation
        signal = AbortSignal()
        try:
            options = replace(request.options, max_tokens=WARM_OUTPUT_TOKENS, signal=signal)
            options = replace(options, api_key=getattr(request.options, "api_key", None))
            async for event in self.stream(request.model, request.context, options):
                if isinstance(event, DoneEvent):
                    break
            if self._closed or generation != self._generation:
                return
            self._schedule()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - warming failures never touch the agent run
            if not self._closed and generation == self._generation:
                # Not yet invalid and not expired: retry once after a fixed delay.
                self._timer = asyncio.get_event_loop().call_later(30.0, self._schedule)
