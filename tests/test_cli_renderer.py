"""Provider failures must be visible: the CLI renderer reports them instead of
returning to the prompt with no output."""

import asyncio
import pytest

from pi.agent.types import AgentToolResult, MessageEndEvent, ToolExecutionStartEvent, ToolExecutionEndEvent
from pi.ai.types import AssistantMessage, ImageContent, TextContent, Usage
from pi.coding_agent.cli import CliRenderer, format_duration


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


def test_renderer_shows_took_for_shell_results(capsys):
    """Shell results show the recorded execute() duration (#10549: B-CA-01)."""
    event = ToolExecutionEndEvent(
        tool_call_id="t", tool_name="bash",
        result=AgentToolResult(content=[TextContent(text="out")]), is_error=False, duration_ms=1500,
    )
    asyncio.run(CliRenderer()(event))
    assert "Took 1.5s" in capsys.readouterr().out


def test_renderer_shows_took_for_errors_and_nested_calls(capsys):
    event = ToolExecutionEndEvent(
        tool_call_id="t", tool_name="powershell",
        result=AgentToolResult(content=[TextContent(text="nope")]), is_error=True,
        duration_ms=62_000, parent_tool_call_id="p1",
    )
    asyncio.run(CliRenderer()(event))
    output = capsys.readouterr().out
    assert "error: nope" in output
    assert "Took 1m 2s" in output


def test_renderer_took_is_shell_only_and_needs_a_duration(capsys):
    other = ToolExecutionEndEvent(
        tool_call_id="t", tool_name="read",
        result=AgentToolResult(content=[TextContent(text="x")]), is_error=False, duration_ms=1500,
    )
    asyncio.run(CliRenderer()(other))
    assert "Took" not in capsys.readouterr().out
    untimed = ToolExecutionEndEvent(
        tool_call_id="t", tool_name="bash",
        result=AgentToolResult(content=[TextContent(text="x")]), is_error=False,
    )
    asyncio.run(CliRenderer()(untimed))
    assert "Took" not in capsys.readouterr().out


def test_renderer_passes_duration_ms_to_extension_renderers():
    """Extension tool renderers get durationMs in the render context (#10549: F-CA-03)."""
    seen = []

    class Extensions:
        async def render_tool(self, tool_name, phase, payload, options=None, width=80):
            seen.append(dict(options or {}))
            return ["lines"]

    renderer = CliRenderer()
    renderer.extensions = Extensions()
    asyncio.run(renderer(ToolExecutionEndEvent(
        tool_call_id="t", tool_name="bash",
        result=AgentToolResult(content=[TextContent(text="out")]), is_error=False, duration_ms=42,
    )))
    assert seen[0]["durationMs"] == 42
    asyncio.run(renderer(ToolExecutionEndEvent(
        tool_call_id="t2", tool_name="bash",
        result=AgentToolResult(content=[TextContent(text="out")]), is_error=False,
    )))
    assert "durationMs" not in seen[1]


def test_renderer_pads_transcript_content():
    """outputPad indents transcript lines; 0 keeps them flush (F-TUI-03)."""
    padded = CliRenderer(settings={"output_pad": 1})
    assert padded.output_pad == 1
    assert padded.pad("line one\n\nline two") == " line one\n\n line two"
    flush = CliRenderer(settings={"output_pad": 0})
    assert flush.output_pad == 0 and flush.pad("line") == "line"
    default = CliRenderer()
    assert default.output_pad == 1


def test_renderer_passes_output_pad_to_extension_renderers():
    """Extension tool renderers get outputPad in the render context (#10557: F-TUI-03)."""
    seen = []

    class Extensions:
        async def render_tool(self, tool_name, phase, payload, options=None, width=80):
            seen.append(dict(options or {}))
            return ["lines"]

    renderer = CliRenderer(settings={"output_pad": 0})
    renderer.extensions = Extensions()
    asyncio.run(renderer(ToolExecutionEndEvent(
        tool_call_id="t", tool_name="bash",
        result=AgentToolResult(content=[TextContent(text="out")]), is_error=False,
    )))
    assert seen[0]["outputPad"] == 0


def test_format_duration_matches_the_ts_shell_renderers():
    assert format_duration(0) == "0.0s"
    assert format_duration(1500) == "1.5s"
    assert format_duration(59_900) == "59.9s"
    assert format_duration(60_000) == "1m 0s"
    assert format_duration(125_000) == "2m 5s"
    assert format_duration(3_600_000) == "1h 0m 0s"
    assert format_duration(3_725_000) == "1h 2m 5s"
