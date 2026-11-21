"""Eval case runner limits: every case is bounded, including async runners."""

import asyncio

import pi.evals as evals
from pi.evals import EvalCase, run_case


def test_run_case_timeout_covers_async_runners(monkeypatch):
    # The timeout used to wrap only non-coroutine awaitables, so the normal
    # `async def runner` path could hang an eval run forever.
    monkeypatch.setattr(evals, "CASE_TIMEOUT_SECONDS", 0.02)

    async def hung(prompt):
        await asyncio.sleep(10)
        return []

    result = asyncio.run(run_case(EvalCase(name="hung", prompt="p"), hung))
    assert not result.passed
    assert "Timeout" in result.failures[0]
