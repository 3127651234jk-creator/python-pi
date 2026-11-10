"""Regression coverage for argument validation and truncation boundaries."""

import pytest

from pi.ai.types import Tool
from pi.ai.utils import validate_tool_arguments
from pi.coding_agent.tools.truncate import truncate_head, truncate_tail


@pytest.mark.parametrize("truncate", [truncate_head, truncate_tail])
@pytest.mark.parametrize("text,limit,expected", [("abc\n", 3, "abc"), ("abcdef", 3, "def")])
def test_byte_truncation_reason(truncate, text, limit, expected):
    result = truncate(text, max_lines=1, max_bytes=limit)
    assert result.truncated_by == "bytes"
    if truncate is truncate_tail or text.endswith("\n"):
        assert result.content == expected


@pytest.mark.parametrize("value", [True, {"nested": [False]}])
def test_enum_distinguishes_booleans_from_numbers(value):
    allowed = 1 if value is True else {"nested": [0]}
    tool = Tool(name="check", description="", parameters={
        "type": "object", "properties": {"value": {"enum": [allowed]}}
    })
    with pytest.raises(ValueError, match="must be one of"):
        validate_tool_arguments(tool, {"value": value})


def test_array_items_without_explicit_array_type():
    tool = Tool(name="check", description="", parameters={
        "properties": {"values": {"items": {"type": "string"}}}
    })
    with pytest.raises(ValueError, match=r"values\[0\]"):
        validate_tool_arguments(tool, {"values": [123]})


def test_enum_accepts_equivalent_json_numbers():
    tool = Tool(name="check", description="", parameters={
        "properties": {"value": {"enum": [{"nested": [1]}]}}
    })
    assert validate_tool_arguments(tool, {"value": {"nested": [1.0]}})
