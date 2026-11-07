"""Provider failures must be visible: the CLI renderer reports them instead of
returning to the prompt with no output."""

import asyncio

from pi.agent.types import MessageEndEvent
from pi.ai.types import AssistantMessage, Usage
from pi.coding_agent.cli import CliRenderer


def assistant(**overrides) -> AssistantMessage:
    fields = dict(content=[], api="openai-completions", provider="deepseek-cn",
                  model="cn:deepseek-v4.1-flash", usage=Usage(), stop_reason="error", timestamp=0)
    fields.update(overrides)
    return AssistantMessage(**fields)


def test_renderer_prints_provider_error(capsys):
    message = assistant(error_message="HTTP 502 from deepseek-cn: (empty response body)")
    asyncio.run(CliRenderer()(MessageEndEvent(message=message)))
    assert "error: HTTP 502 from deepseek-cn: (empty response body)" in capsys.readouterr().out


def test_renderer_falls_back_when_the_error_has_no_message(capsys):
    asyncio.run(CliRenderer()(MessageEndEvent(message=assistant())))
    assert "error: Unknown provider error" in capsys.readouterr().out


def test_renderer_stays_silent_for_a_successful_empty_message(capsys):
    asyncio.run(CliRenderer()(MessageEndEvent(message=assistant(stop_reason="stop"))))
    assert capsys.readouterr().out == ""
