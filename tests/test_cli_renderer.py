"""Provider failures must be visible: the CLI renderer reports them instead of
returning to the prompt with no output."""

import asyncio
import pytest

from pi.agent.types import AgentToolResult, MessageEndEvent, ToolExecutionStartEvent, ToolExecutionEndEvent
from pi.ai.types import AssistantMessage, ImageContent, TextContent, Usage
from pi.coding_agent.cli import CliRenderer


@pytest.mark.parametrize("with_text", [False, True])
@pytest.mark.parametrize("is_error", [False, True])
def test_renderer_handles_image_first_tool_result(monkeypatch, capsys, with_text, is_error):
    renderer = CliRenderer()
    image = ImageContent(data="image-data", mime_type="image/png")
    content = [image]
    if with_text:
        content.extend([TextContent(text="first text"), TextContent(text="second text")])
    rendered = []
    monkeypatch.setattr(renderer, "print_image_block", rendered.append)
    asyncio.run(renderer(ToolExecutionEndEvent(
        tool_call_id="screenshot", tool_name="screenshot",
        result=AgentToolResult(content=content), is_error=is_error,
    )))
    assert rendered == [image]
    output = capsys.readouterr().out
    if with_text:
        assert "first text" in output
        assert "second text" in output
    if is_error:
        assert "error:" in output


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


def test_renderer_labels_a_search_with_its_pattern(capsys):
    event = ToolExecutionStartEvent(tool_call_id="t", tool_name="find", args={"pattern": "*.py"})
    asyncio.run(CliRenderer()(event))
    assert "[find] *.py" in capsys.readouterr().out
