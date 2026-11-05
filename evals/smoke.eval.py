"""Smoke host eval: one complete conversation round-trip.

Port of ``packages/evals/evals/smoke.eval.ts``. Host evals run on this machine
through ``python -m pi.evals.host`` (the replacement for TS
``vitest run --project host``); the model comes from PI_PROVIDER / PI_MODEL.
This suite is the first thing that should pass in ``eval:host``: it checks the
bare prompt -> model -> text path with every tool disabled.
"""

import os

from pi.evals.framework import describe_eval
from pi.evals.harness import PiCodingAgentHarnessOptions, create_pi_coding_agent_harness

harness = create_pi_coding_agent_harness(PiCodingAgentHarnessOptions(no_tools=True))


@describe_eval("Answer a basic prompt", harness=harness)
async def returns_the_expected_answer(run):
    result = await run("What's the capital of France? Respond with only the city name.")
    assert result["output"].strip() == "Paris"
    assert result["errors"] == []
    usage = result["usage"]
    assert usage["provider"] == os.environ.get("PI_PROVIDER")
    assert usage["model"] == os.environ.get("PI_MODEL")
    assert usage["totalTokens"] > 0
