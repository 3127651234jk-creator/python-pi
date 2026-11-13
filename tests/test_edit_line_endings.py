"""Line ending detection follows the first newline, as in durable edits."""

import asyncio

import pytest

from pi.coding_agent.tools.edit import create_edit_tool
from pi.coding_agent.tools.edit_diff import detect_line_ending
from pi.durable.tools.edit_diff import detect_line_ending as durable_detect_line_ending


@pytest.mark.parametrize("content,expected", [
    ("", "\n"),
    ("no newline", "\n"),
    ("a\nb\n", "\n"),
    ("a\r\nb\r\n", "\r\n"),
    ("a\nb\r\nc\n", "\n"),
    ("a\r\nb\nc\n", "\r\n"),
    ("a\rb\r\nc", "\r\n"),
])
def test_detect_line_ending_matches_durable(content, expected):
    assert detect_line_ending(content) == expected
    assert detect_line_ending(content) == durable_detect_line_ending(content)


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_edit_lf_file_with_one_crlf_does_not_convert_to_crlf(tmp_path, bom):
    path = tmp_path / "mixed.txt"
    path.write_bytes(bom + b"first\nsecond\r\ntarget\nfourth\n")

    async def scenario():
        result = await create_edit_tool(str(tmp_path)).execute(
            "edit", {"path": "mixed.txt", "edits": [{"oldText": "target", "newText": "changed"}]}
        )
        assert not result.is_error

    asyncio.run(scenario())
    assert path.read_bytes() == bom + b"first\nsecond\nchanged\nfourth\n"
