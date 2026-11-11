"""Changed tool interfaces must reach the next provider request."""

from dataclasses import replace

import pytest

from pi.ai.types import SystemMessage, Tool, TranscriptContext
from pi.ai.utils import get_current_tools, get_tool_state_changes, normalize_context


@pytest.mark.parametrize("changes", [
    {"description": "new description"},
    {"parameters": {"type": "object", "required": ["path"]}},
    {"hosted": {"type": "web_search"}},
])
def test_redefinition_is_replayed_with_new_interface(changes):
    original = Tool(name="tool", description="original", parameters={"type": "object"})
    updated = replace(original, **changes)
    delta = get_tool_state_changes([original], [updated])
    assert delta.tools_added == [updated]
    assert [ref.name for ref in delta.tools_removed] == ["tool"]
    messages = [SystemMessage(content="", timestamp=0, tools_added=[original]),
                SystemMessage(content="", timestamp=1, tools_added=delta.tools_added,
                              tools_removed=delta.tools_removed)]
    assert get_current_tools(messages) == [updated]
    system, _ = normalize_context(TranscriptContext(messages=messages))
    assert system.tools_added == [updated]
