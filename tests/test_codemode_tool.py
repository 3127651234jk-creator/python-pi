"""Codemode tool tests: schema/description rendering, script-facing conversion,
store commits, output ordering, truncation and image files."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import ImageContent, TextContent
from pi.codemode import tool as codemode_tool
from pi.codemode.tool import (
    _save_images,
    create_codemode_description,
    create_codemode_tool,
    load_codemode_store,
    render_tool_sample,
)
from pi.coding_agent.nested_tool_calls import NestedToolCallOutcome
from pi.coding_agent.session import SessionManager

PNG_1PX = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
           "AAAABJRU5ErkJggg==")


def leaf(name, description="leaf tool", parameters=None, output_schema=None, text="leaf"):
    async def execute(tool_call_id, params, signal=None, on_update=None):
        execute.calls.append((tool_call_id, params))
        return AgentToolResult(content=[TextContent(text=f"{text} {tool_call_id}")])

    execute.calls = []
    return AgentTool(name=name, label=name, description=description,
                     parameters=parameters or {"type": "object"}, execute=execute,
                     output_schema=output_schema)


def outcome(result, is_error=False, tool_call_id="runner/9"):
    return NestedToolCallOutcome(tool_call_id=tool_call_id, result=result, is_error=is_error)


def texts_of(result):
    return [block.text for block in result.content if isinstance(block, TextContent)]


def joined(result):
    return "\n".join(texts_of(result))


# ---------------------------------------------------------------------------
# Schema and description rendering
# ---------------------------------------------------------------------------


def test_render_tool_sample_renders_parameter_types():
    assert render_tool_sample("plain", "", {}) == "tools.plain() -> str"
    assert render_tool_sample("s", "", {"type": "object", "properties": {"a": {"type": "string"}},
                                        "required": ["a"]}) == "tools.s(a: str) -> str"
    assert render_tool_sample("s", "", {"type": "object", "properties": {
        "a": {"type": "string"}, "n": {"type": "integer"}, "f": {"type": "number"},
        "b": {"type": "boolean"}, "any": {"type": ["string", "null"]}}}) == \
        "tools.s(a: str = None, n: int = None, f: float = None, b: bool = None, any: str | Any = None) -> str"
    assert render_tool_sample("s", "", {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "integer"}}}}) == \
        "tools.s(items: list[int] = None) -> str"
    assert render_tool_sample("s", "", {"type": "object", "properties": {
        "obj": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}) == \
        "tools.s(obj: {x: str} = None) -> str"
    assert render_tool_sample("s", "", {"type": "object", "properties": {
        "e": {"enum": ["a", "b"]}, "u": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}}) == \
        "tools.s(e: 'a', 'b' = None, u: str | int = None) -> str"
    assert render_tool_sample("s", "Do things.", {"type": "object"}) == "tools.s() -> str\n\nDo things."
    assert render_tool_sample("s", "", True) == "tools.s() -> str"
    # A False parameter schema renders as Never at the property level.
    assert render_tool_sample("s", "", {"type": "object", "properties": {"x": False}}) == \
        "tools.s(x: Never = None) -> str"
    assert render_tool_sample("s", "", {"type": "array"}) == "tools.s() -> str"


def test_render_tool_sample_describes_output_schema():
    schema = {"type": "object", "properties": {"echoed": {"type": "object"},
                                               "id": {"type": "string"}},
              "required": ["echoed"]}
    assert " -> { echoed, id? }" in render_tool_sample("s", "", {}, schema)
    assert " -> str" in render_tool_sample("s", "", {}, {"type": "string"})
    assert " -> str" in render_tool_sample("s", "", {}, {"type": "object"})


def test_description_lists_namespaces_and_helpers():
    tools = [leaf("echo", "Echo it."), leaf("mcp__dev__query", "Query."),
             leaf("mcp__dev__other", "Other."), leaf("mcp__stats__count", "Count.")]
    description = create_codemode_description(tools)
    assert description.startswith(codemode_tool.DESCRIPTION_INTRO)
    assert codemode_tool.DESCRIPTION_GLOBALS in description
    body = description.split("Nested tools:\n", 1)[1]
    assert body.lstrip("\n").startswith("### `echo`")
    assert "## dev\n\n### `mcp__dev__query`" in body
    assert "## stats" in body
    assert body.index("## dev") < body.index("## stats")
    assert "tools.echo() -> str\n\nEcho it." in body
    # With no budget everything is listed.
    assert "(tools not listed)" not in body and "(some tools not listed)" not in body


def test_description_excludes_codemode_and_groups_unlisted():
    tools = [leaf("codemode", "self"), leaf("echo", "Echo it."),
             leaf("mcp__dev__query", "Query."), leaf("mcp__dev__other", "Other.")]
    description = create_codemode_description(tools, inline_budget=1)
    assert "self" not in description
    body = description.split("Nested tools:\n", 1)[1]
    # Budget 1 affords nothing: namespace groups keep their heading with a
    # marker, and unaffordable ungrouped samples are dropped silently.
    assert "## dev (tools not listed)" in body
    assert "### `echo`" not in body


def test_description_budget_keeps_every_namespace():
    tools = [leaf("mcp__a__big_one", "x" * 400), leaf("mcp__a__big_two", "y" * 400),
             leaf("mcp__b__small", "z")]
    description = create_codemode_description(tools, inline_budget=10)
    body = description.split("Nested tools:\n", 1)[1]
    # The a-section samples exceed the budget; the cheap b tool still shows.
    assert "## a (tools not listed)" in body
    assert "## b" in body and "(tools not listed)" not in body.split("## b", 1)[1]
    assert "### `mcp__b__small`" in body


def test_description_marks_identifier_collisions():
    tools = [leaf("a-b", "dash"), leaf("a_b", "under")]
    body = create_codemode_description(tools).split("Nested tools:\n", 1)[1]
    assert "### `a_b` (`a-b`)" in body
    assert "### `a_b`\ntools.a_b() -> str\n\nunder" in body


def test_tool_declaration():
    tool = create_codemode_tool([leaf("echo")])
    assert tool.name == "codemode"
    assert tool.parameters == {"type": "object", "properties": {
        "code": {"type": "string", "description": "Raw Python source."}}, "required": ["code"]}
    assert tool.description.startswith("Run Python that calls other tools.")


# ---------------------------------------------------------------------------
# Script-facing conversion of nested outcomes
# ---------------------------------------------------------------------------


def test_structured_content_preferred_over_text():
    async def scenario():
        async def execute_tool(name, args, signal):
            return outcome(AgentToolResult(
                content=[TextContent(text="ignored text")],
                structured_content={"data": args}))

        tool = create_codemode_tool([leaf("echo")], execute_tool=execute_tool)
        result = await tool.execute("c", {"code": "result = tools.echo({'x': 1})"}, None, None)
        assert result.is_error is False
        assert result.structured_content is None  # conversion stays script-side
        assert "Script completed" in texts_of(result)[0]
        assert json.loads(texts_of(result)[1]) == {"data": {"x": 1}}

    asyncio.run(scenario())


def test_text_fallback_when_no_structured_content():
    async def scenario():
        async def execute_tool(name, args, signal):
            return outcome(AgentToolResult(content=[TextContent(text="line1"),
                                                    TextContent(text="line2")]))

        tool = create_codemode_tool([leaf("echo")], execute_tool=execute_tool)
        result = await tool.execute("c", {"code": "result = tools.echo({})"}, None, None)
        # String values reach the script (and the output) verbatim.
        assert texts_of(result)[1] == "line1\nline2"

    asyncio.run(scenario())


def test_error_outcome_is_catchable_and_reports_text():
    async def scenario():
        async def execute_tool(name, args, signal):
            return outcome(AgentToolResult(content=[TextContent(text="boom detail")], is_error=True),
                           is_error=True)

        tool = create_codemode_tool([leaf("echo")], execute_tool=execute_tool)
        code = (
            "try:\n"
            "    tools.echo({})\n"
            "    failed = None\n"
            "except RuntimeError as exc:\n"
            "    failed = str(exc)\n"
            "result = failed\n"
        )
        result = await tool.execute("c", {"code": code}, None, None)
        assert result.is_error is False
        assert texts_of(result)[1] == "boom detail"

    asyncio.run(scenario())


def test_error_outcome_without_text_uses_tool_name():
    async def scenario():
        async def execute_tool(name, args, signal):
            return outcome(AgentToolResult(content=[], is_error=True), is_error=True)

        tool = create_codemode_tool([leaf("echo")], execute_tool=execute_tool)
        code = (
            "try:\n"
            "    tools.echo({})\n"
            "except RuntimeError as exc:\n"
            "    result = str(exc)\n"
        )
        result = await tool.execute("c", {"code": code}, None, None)
        assert texts_of(result)[1] == 'tool "echo" failed'

    asyncio.run(scenario())


def test_details_record_carries_runner_assigned_id():
    async def scenario():
        async def execute_tool(name, args, signal):
            return outcome(AgentToolResult(content=[TextContent(text="ok")],
                                           structured_content=None), tool_call_id="runner/7")

        updates = []

        async def on_update(partial):
            updates.append(partial)

        tool = create_codemode_tool([leaf("echo")], execute_tool=execute_tool)
        result = await tool.execute("c", {"code": "tools.echo({})"}, None, on_update)
        assert updates, "progress updates should reach the agent"
        assert updates[0].details["calls"][0]["id"] == "c/1"
        assert updates[0].details["calls"][0]["status"] == "running"
        assert result.details["calls"][0]["id"] == "runner/7"
        assert result.details["calls"][0]["status"] == "ok"
        assert isinstance(result.details["calls"][0]["duration_ms"], int)

    asyncio.run(scenario())


def test_standalone_fallback_calls_tool_execute_directly():
    async def scenario():
        echo = leaf("echo", text="leaf says")
        tool = create_codemode_tool([echo])
        result = await tool.execute("c", {"code": "result = tools.echo({'k': 'v'})"}, None, None)
        assert result.is_error is False
        assert texts_of(result)[1] == "leaf says c/1"
        # The nested-style id is what the runner would have assigned.
        assert echo.execute.calls == [("c/1", {"k": "v"})]
        assert result.details["calls"][0]["id"] == "c/1"

    asyncio.run(scenario())


def test_codemode_excludes_itself_from_the_catalog():
    async def scenario():
        tool = create_codemode_tool([leaf("echo"), leaf("codemode")])
        result = await tool.execute("c", {"code": "result = dir(tools)"}, None, None)
        assert json.loads(texts_of(result)[1]) == ["echo"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_commits_on_success_only(tmp_path):
    async def scenario():
        session = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path))
        session.start_new()
        tool = create_codemode_tool([leaf("echo")], session=session)

        result = await tool.execute("c1", {"code": "store('a', {'n': 1})\nresult = 'ok'"}, None, None)
        assert result.is_error is False
        entries = session.custom_entries("codemode-store")
        assert len(entries) == 1
        assert entries[0].custom["data"] == {"set": {"a": {"n": 1}}, "delete": []}

        result = await tool.execute("c2", {"code": "store('b', 1)\n1/0"}, None, None)
        assert result.is_error is True
        assert len(session.custom_entries("codemode-store")) == 1  # failure wrote nothing
        assert load_codemode_store(session) == {"a": {"n": 1}}

        result = await tool.execute("c3", {"code": "store('a')"}, None, None)
        assert result.is_error is False
        entries = session.custom_entries("codemode-store")
        assert entries[-1].custom["data"] == {"set": {}, "delete": ["a"]}
        assert load_codemode_store(session) == {}

    asyncio.run(scenario())


def test_store_replay_reaches_the_next_script(tmp_path):
    async def scenario():
        session = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path))
        session.start_new()
        session.append_custom("codemode-store", {"set": {"keep": "me"}, "delete": []})
        session.append_custom("codemode-store", {"set": {"other": [1]}, "delete": ["gone"]})
        session.append_custom("codemode-store", {"set": {"gone": "x"}, "delete": []})

        tool = create_codemode_tool([leaf("echo")], session=session)
        result = await tool.execute("c", {"code": "result = {'keep': load('keep'),\n"
                                                  "          'other': load('other'),\n"
                                                  "          'gone': load('gone'),\n"
                                                  "          'missing': load('nope')}"}, None, None)
        # Replay applies each entry in order: the delete in entry two is later
        # overwritten by the set in entry three.
        assert json.loads(texts_of(result)[1]) == {"keep": "me", "other": [1], "gone": "x",
                                                   "missing": None}

    asyncio.run(scenario())


def test_load_codemode_store_without_session():
    assert load_codemode_store(None) == {}


# ---------------------------------------------------------------------------
# Output: ordering, truncation, images
# ---------------------------------------------------------------------------


def test_output_ordering_texts_result_then_images():
    async def scenario():
        tool = create_codemode_tool([leaf("echo")])
        code = (
            "text('t1')\n"
            "print('t2')\n"
            "image('data:image/png;base64," + PNG_1PX + "')\n"
            "text('t3')\n"
            "result = 'RES'\n"
        )
        result = await tool.execute("c", {"code": code}, None, None)
        texts = texts_of(result)
        assert texts[0].startswith("Script completed\nWall time ")
        assert texts[0].endswith(" seconds\nOutput:")
        # Produced text first, then the result value, then the image group.
        assert texts[1:5] == ["t1", "t2", "t3", "RES"]
        assert isinstance(result.content[-1], ImageContent)
        save_notes = [text for text in texts if text.startswith("[Image saved to ")]
        assert len(save_notes) == 1
        assert save_notes[0].startswith("[Image saved to ") and "image/png, 70 bytes)]" in save_notes[0]

    asyncio.run(scenario())


def test_truncation_keeps_head_tail_and_spills(tmp_path):
    async def scenario():
        tool = create_codemode_tool([leaf("echo")], session=None)
        code = '# @options: {"max_output_tokens": 1}\ntext("abcdefghij")\n'
        result = await tool.execute("c", {"code": code}, None, None)
        warning = texts_of(result)[1]
        assert warning.startswith("Warning: truncated output (original token count: 3)\n"
                                  "Total output lines: 1\n\nab...2 tokens truncated...ij")
        marker = "[Full output: "
        start = warning.index(marker) + len(marker)
        spill = Path(warning[start:warning.index(" (read with offset/limit)]")])
        assert spill.name == "codemode_c_full_output.txt"
        assert spill.read_text(encoding="utf-8") == "abcdefghij"

    asyncio.run(scenario())


def test_truncation_preserves_image_blocks():
    async def scenario():
        tool = create_codemode_tool([leaf("echo")])
        code = ('# @options: {"max_output_tokens": 1}\n'
                'image("data:image/png;base64,' + PNG_1PX + '")\n'
                'text("abcdefghij")\n')
        result = await tool.execute("c", {"code": code}, None, None)
        assert isinstance(result.content[-1], ImageContent)
        assert result.content[-1].data == PNG_1PX

    asyncio.run(scenario())


def test_save_images_writes_files_and_reuses_identical_data(tmp_path):
    items = [ImageContent(data=PNG_1PX, mime_type="image/png"),
             ImageContent(data=PNG_1PX, mime_type="image/png")]
    saved = _save_images(items, tmp_path, "call9")
    notes = [item.text for item in saved if isinstance(item, TextContent)]
    assert notes == [f"[Image saved to {tmp_path / 'codemode_call9_img1.png'} (image/png, 70 bytes)]",
                     notes[0]]
    assert (tmp_path / "codemode_call9_img1.png").read_bytes() == base64.b64decode(PNG_1PX)
    assert [item for item in saved if isinstance(item, ImageContent)] == items


def test_save_images_reports_unknown_extension(tmp_path):
    items = [ImageContent(data="aGk=", mime_type="image/x-unknown")]
    saved = _save_images(items, tmp_path, "c")
    assert "[Image (image/x-unknown) has no known file extension]" in saved[0].text
    assert saved[1] is items[0]


def test_workdir_is_shared_across_executions(tmp_path):
    async def scenario():
        tool = create_codemode_tool([leaf("echo")])

        def spill_path(result):
            text = texts_of(result)[1]
            marker = "[Full output: "
            start = text.index(marker) + len(marker)
            return Path(text[start:text.index(" (read with offset/limit)]")])

        first = await tool.execute("c1", {
            "code": '# @options: {"max_output_tokens": 1}\ntext("first")\n'}, None, None)
        second = await tool.execute("c2", {
            "code": '# @options: {"max_output_tokens": 1}\ntext("second")\n'}, None, None)
        spill1, spill2 = spill_path(first), spill_path(second)
        assert spill1.parent == spill2.parent  # one scratch dir per tool
        assert spill1.exists() and spill2.exists()  # nothing is auto-deleted
        assert spill1.read_text(encoding="utf-8") == "first"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Input validation and settings
# ---------------------------------------------------------------------------


def test_code_is_required():
    async def scenario():
        tool = create_codemode_tool([leaf("echo")])
        with pytest.raises(ValueError) as excinfo:
            await tool.execute("c", {}, None, None)
        assert str(excinfo.value) == "codemode requires a code string"
        with pytest.raises(ValueError):
            await tool.execute("c", {"code": "   "}, None, None)
        with pytest.raises(ValueError):
            await tool.execute("c", {"code": 5}, None, None)

    asyncio.run(scenario())


def test_options_parse_errors_become_failed_results():
    async def scenario():
        tool = create_codemode_tool([leaf("echo")])
        for code, message in [
            ('# @options: {"timeout_ms": -1}\nx = 1', "codemode @options timeout_ms must be a positive integer"),
            ('#@options: {"unknown": 1}\nx = 1', 'codemode @options supports "timeout_ms" and "max_output_tokens"'),
            ("# @options: [1]\nx = 1", 'codemode @options supports "timeout_ms" and "max_output_tokens"'),
            ("# @options: oops\nx = 1", "codemode @options line is not valid JSON"),
        ]:
            result = await tool.execute("c", {"code": code}, None, None)
            assert result.is_error is True
            assert "Script error:\n" + message in joined(result)
            assert "No tool calls were made." in joined(result)
            assert result.details == {"calls": []}

    asyncio.run(scenario())


def test_settings_timeout_applies_and_options_override_it():
    async def scenario():
        never = asyncio.Event()

        async def blocking(name, args, signal):
            await never.wait()
            return outcome(AgentToolResult(content=[TextContent(text="unreachable")]))

        tool = create_codemode_tool([leaf("echo")], settings={"timeout_ms": 400},
                                    execute_tool=blocking)
        result = await tool.execute("c", {"code": "result = wait(tools.echo.start({}))"},
                                    None, None)
        assert result.is_error is True
        assert "Script timed out: exceeded the 400 ms deadline" in joined(result)
        assert result.details["calls"][0]["status"] == "cancelled"

        async def quick(name, args, signal):
            return outcome(AgentToolResult(content=[TextContent(text="quick")],
                                           structured_content={"quick": True}))

        overrider = create_codemode_tool([leaf("echo")], settings={"timeout_ms": 200},
                                         execute_tool=quick)
        result = await overrider.execute("c", {
            "code": '# @options: {"timeout_ms": 30000}\nresult = tools.echo({})'}, None, None)
        assert result.is_error is False  # the @options timeout wins over settings
        assert json.loads(texts_of(result)[1]) == {"quick": True}

    asyncio.run(scenario())


def test_settings_inline_budget_changes_the_catalog():
    tool = create_codemode_tool([leaf("mcp__dev__query", "Query.")],
                                settings={"inline_budget": 1})
    assert "## dev (tools not listed)" in tool.description
    wide = create_codemode_tool([leaf("mcp__dev__query", "Query.")],
                                settings={"inline_budget": None})
    assert "(tools not listed)" not in wide.description
