"""Prompt cache warming: capture, scheduling, economics, replay (F-CA-13)."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.ai.models import Model, ModelCost
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import AssistantMessage, DoneEvent, TextContent, TranscriptContext, Usage, UserMessage
from pi.coding_agent.cache_warmer import CacheWarmer, decide

# Prices: input 3, output 15, cache_read 0.3, cache_write 3.75 per million.
BIG_USAGE = Usage(input=500_000, output=10, cache_read=0, cache_write=500_000)


def make_model(costs=None, prompt_cache=None) -> Model:
    return Model(id="claude-sonnet-4-5-20250929", name="Sonnet", api="anthropic-messages",
                 provider="anthropic", base_url="https://api.anthropic.com", reasoning=True,
                 context_window=200_000, max_tokens=64_000,
                 costs=costs or ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
                 prompt_cache=prompt_cache)


def captured_context():
    """Last real request projection: the user prompt and the assistant usage
    the warmer uses as the prompt size."""
    return TranscriptContext(messages=[request_message(), assistant_message()])


def request_message():
    return UserMessage(content="x" * 400, timestamp=0)


def assistant_message():
    return AssistantMessage(content=[TextContent(text="ok")], api="anthropic-messages", provider="anthropic",
                            model="claude-sonnet-4-5-20250929", usage=BIG_USAGE,
                            stop_reason="stop", timestamp=0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_warmer(clock=None, calls=None):
    clock = clock or FakeClock()
    calls = calls if calls is not None else []

    async def stream(sent_model, context, options):
        calls.append((sent_model, context, options))
        yield DoneEvent(reason="stop", message=assistant_message())

    warmer = CacheWarmer(stream, clock=clock)
    warmer.mode = "idle"
    return warmer, clock, calls


def capture_request(warmer, model=None, options=None):
    model = model or make_model(prompt_cache={"5m": 300})
    warmer.capture(model, captured_context(), options or SimpleStreamOptions(cache_retention="5m"))
    return model


def test_capture_schedules_replay_with_one_token_cap():
    """TTL=300s schedules at 270s and replays the same context with max_tokens=1."""
    async def scenario():
        warmer, clock, calls = make_warmer()
        capture_request(warmer)
        assert warmer.status.state == "scheduled"
        assert abs(warmer.status.next_warm_at - (clock.now + 270)) < 0.01
        clock.advance(270)
        warmer._timer._run()  # execute the pending call_later callback
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        sent_model, context, options = calls[0]
        assert options.max_tokens == 1
        assert options.cache_retention == "5m"
        assert context.messages[0].content == "x" * 400
        await warmer.close()

    asyncio.run(scenario())


def test_economics_threshold_and_idle_probability():
    model = make_model()
    # 1000 prompt tokens: miss_cost = 1000*(3.75-0.3)/1e6 = 0.00345; the warm
    # request costs ~0.000315 — below the 0.05 threshold even when streaming.
    assert decide(model, 1000, idle=False).action == "stop"
    # A long prompt clears the threshold at p=0.15 too.
    assert decide(model, 1_000_000, idle=True).action == "warm"
    assert decide(model, 200_000, idle=False).action == "warm"
    # Zero prices are not free: a fully zero catalog stops warming.
    decision = decide(make_model(costs=ModelCost()), 1_000_000, idle=False)
    assert decision.action == "stop"
    assert decision.economics_available is False
    assert decide(model, 1_000_000, idle=False).economics_available is True


def test_unknown_ttl_and_unsafe_replay_stay_inactive():
    async def scenario():
        # No prompt_cache metadata: the request intent alone is not proof.
        warmer, _, calls = make_warmer()
        capture_request(warmer, make_model())
        assert warmer.status.state == "inactive" and "TTL" in warmer.status.reason
        await warmer.close()
        assert calls == []

        # Budget-based thinking derives its parameters from the output cap.
        warmer, _, calls = make_warmer()
        model = make_model(prompt_cache={"5m": 300})
        options = SimpleStreamOptions(cache_retention="5m", reasoning="high")
        warmer.capture(model, captured_context(), options)
        assert warmer.status.state == "inactive" and "thinking" in warmer.status.reason
        await warmer.close()
        assert calls == []

    asyncio.run(scenario())


def test_new_real_request_cancels_pending_refresh():
    async def scenario():
        warmer, clock, calls = make_warmer()
        capture_request(warmer)
        capture_request(warmer)  # a second real request replaces the first
        clock.advance(270)
        warmer._timer._run()
        await asyncio.sleep(0.05)
        assert len(calls) == 1  # one replay for the newest capture
        await warmer.close()

    asyncio.run(scenario())


def test_streaming_settled_stops_and_cancel_clears():
    async def scenario():
        warmer, clock, _ = make_warmer()
        capture_request(warmer)
        warmer.mode = "streaming"
        warmer.on_agent_settled()
        assert warmer.status.state == "inactive" and warmer.request is None
        await warmer.close()

        warmer, clock, _ = make_warmer()
        capture_request(warmer)
        warmer.cancel()
        clock.advance(300)
        assert warmer.request is None and warmer.status.state == "inactive"
        await warmer.close()

    asyncio.run(scenario())


def test_idle_mode_uses_streaming_probability_until_settled():
    """During a run the decision uses p=1.0; only agent_end switches to idle."""

    async def scenario():
        warmer, _, _ = make_warmer()
        capture_request(warmer)
        assert warmer.status.decision.phase == "streaming"
        assert warmer.status.decision.continuation_probability == 1.0
        warmer.on_agent_settled()
        assert warmer.status.decision.phase == "idle"
        assert warmer.status.decision.continuation_probability == 0.15
        await warmer.close()

    asyncio.run(scenario())


def test_idle_window_restarts_per_run():
    """started_at is per run: an expired run must not disable later runs."""

    async def scenario():
        warmer, clock, _ = make_warmer()
        capture_request(warmer)
        clock.advance(31 * 60)
        warmer.on_agent_settled()
        assert warmer.request is None  # this run's 30-minute window expired
        capture_request(warmer)
        warmer.on_agent_settled()
        assert warmer.request is not None  # a new request starts a new window
        await warmer.close()

    asyncio.run(scenario())


def test_agent_capture_hook_records_last_request():
    """The Agent's capture hook fires on the request that produced the response."""
    from pi.agent.agent import Agent
    from pi.ai.providers import faux

    captured = []

    async def scenario():
        os.environ["FAUX_SCRIPT"] = '[{"text": "ok"}]'
        faux.reset_script_index()
        model = Model(id="faux", name="Faux", api="faux", provider="faux", base_url="",
                      costs=ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
                      prompt_cache={"5m": 300})
        agent = Agent(model)
        agent.on_real_request = lambda model, context, options: captured.append((model, context, options))
        try:
            await agent.prompt("hello")
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())
    assert len(captured) == 1
    model, context, options = captured[0]
    assert model.prompt_cache == {"5m": 300}
    assert context.messages[-1].content == "hello"
