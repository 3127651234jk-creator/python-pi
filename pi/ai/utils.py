"""Transcript normalization and tool-argument validation.

Python port of ``normalizeContext``, ``getCurrentTools``, ``getToolStateChanges``
and ``validateToolArguments`` from ``packages/ai/src``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Dict, List, Optional, Tuple

from ..util import AbortError
from .types import (
    AssistantMessage,
    Message,
    SystemMessage,
    Tool,
    ToolReference,
    TranscriptContext,
    UserMessage,
)

_SYSTEM_ROLES = ("system", "user", "assistant", "toolResult")


def is_llm_message(message: Any) -> bool:
    role = getattr(message, "role", None)
    return role in _SYSTEM_ROLES


@dataclass
class ToolStateChanges:
    tools_added: List[Tool]
    tools_removed: List[ToolReference]


NO_CHANGES = ToolStateChanges(tools_added=[], tools_removed=[])


def normalize_tool_call_id(call_id: str) -> str:
    """Use the call portion of compound IDs and satisfy Claude's ID limits."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", call_id.split("|", 1)[0])[:64] or "call"


def normalize_context(context: TranscriptContext) -> Tuple[Optional[SystemMessage], List[Message]]:
    """Replay system messages into one leading system message.

    Returns ``(system, messages)`` where ``system`` carries the accumulated
    prompt text, sections, and the effective tool declarations, and
    ``messages`` contains the non-system transcript in order. Providers that
    accept mid-conversation system messages can ignore the fold and send
    ``context.messages`` directly; the replayed leading message is equivalent.
    """

    prompt_parts: List[str] = []
    sections: Dict[str, Optional[str]] = {}
    tools: List[Tool] = []
    removed: set = set()
    rest: List[Message] = []

    for message in context.messages:
        if isinstance(message, SystemMessage):
            if isinstance(message.content, str):
                if message.content:
                    prompt_parts.append(message.content)
            else:
                text = "\n".join(block.text for block in message.content)
                if text:
                    prompt_parts.append(text)
            if message.sections:
                for name, section in message.sections.items():
                    sections[name] = section
            for ref in message.tools_removed or []:
                removed.add(ref.name)
                tools = [t for t in tools if t.name != ref.name]
            for tool in message.tools_added or []:
                removed.discard(tool.name)
                tools = [t for t in tools if t.name != tool.name]
                tools.append(tool)
        else:
            rest.append(message)

    system: Optional[SystemMessage] = None
    if prompt_parts or sections or tools:
        content = "\n\n".join(prompt_parts)
        system = SystemMessage(
            content=content,
            timestamp=0,
            sections=dict(sections) or None,
            tools_added=tools or None,
        )
    return system, rest


def get_current_tools(messages: List[Message]) -> List[Tool]:
    """Replay the transcript's system messages and return the declared tools."""
    _, _, tools, _ = _replay_system(messages)
    return tools


def _replay_system(messages: List[Message]) -> Tuple[List[str], Dict[str, Optional[str]], List[Tool], set]:
    prompt_parts: List[str] = []
    sections: Dict[str, Optional[str]] = {}
    tools: List[Tool] = []
    removed: set = set()
    for message in messages:
        if not isinstance(message, SystemMessage):
            continue
        if isinstance(message.content, str):
            if message.content:
                prompt_parts.append(message.content)
        else:
            text = "\n".join(block.text for block in message.content)
            if text:
                prompt_parts.append(text)
        if message.sections:
            for name, section in message.sections.items():
                sections[name] = section
        for ref in message.tools_removed or []:
            removed.add(ref.name)
            tools = [t for t in tools if t.name != ref.name]
        for tool in message.tools_added or []:
            removed.discard(tool.name)
            tools = [t for t in tools if t.name != tool.name]
            tools.append(tool)
    return prompt_parts, sections, tools, removed


def get_tool_state_changes(declared: List[Tool], executable: List[Tool]) -> ToolStateChanges:
    """Diff tools declared in the transcript against the executable set."""
    declared_names = {t.name for t in declared}
    executable_names = {t.name for t in executable}
    previous = {t.name: t for t in declared}
    changed = {t.name for t in executable if t.name in previous and not _json_equal(
        [previous[t.name].description, previous[t.name].parameters, previous[t.name].hosted],
        [t.description, t.parameters, t.hosted],
    )}
    added = [t for t in executable if t.name not in declared_names or t.name in changed]
    removed = [ToolReference(name=n) for n in sorted((declared_names - executable_names) | changed)]
    return ToolStateChanges(tools_added=added, tools_removed=removed)


def with_tool_changes(message: SystemMessage, changes: ToolStateChanges) -> SystemMessage:
    """Copy a system message with its tool fields replaced by ``changes``."""
    return replace(
        message,
        tools_added=list(changes.tools_added) if changes.tools_added else None,
        tools_removed=list(changes.tools_removed) if changes.tools_removed else None,
    )


# ---------------------------------------------------------------------------
# JSON Schema validation (subset sufficient for tool arguments)
# ---------------------------------------------------------------------------


def validate_tool_arguments(tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tool call arguments against the tool's JSON schema.

    Supports the subset pi's tools declare: object/string/number/integer/
    boolean/array/enum, properties, required, items. Raises ``ValueError``
    with a model-readable message on violation, like the TS loop does.
    """
    schema = tool.parameters or {}
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool {tool.name} arguments must be a JSON object")
    _validate_value(tool.name, arguments, schema, path=tool.name)
    return arguments


def _validate_value(tool_name: str, value: Any, schema: Dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_type_matches(value, t) for t in expected):
            raise ValueError(f"{path}: expected one of type {expected}")
        expected = next(t for t in expected if _type_matches(value, t))
    elif expected is not None:
        if not _type_matches(value, expected):
            raise ValueError(f"{path}: expected {expected}, got {_type_name(value)}")

    if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
        raise ValueError(f"{path}: must be one of {schema['enum']!r}")

    if expected == "object" or (expected is None and isinstance(value, dict)):
        properties = schema.get("properties", {})
        for required_key in schema.get("required", []):
            if required_key not in value:
                raise ValueError(f"{path}: missing required property '{required_key}'")
        for key, item in value.items():
            if key in properties:
                _validate_value(tool_name, item, properties[key], path=f"{path}.{key}")
    elif expected == "array" or (expected is None and isinstance(value, list)):
        items = schema.get("items")
        if items:
            for index, item in enumerate(value):
                _validate_value(tool_name, item, items, path=f"{path}[{index}]")


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's boolean/number equivalence."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def abort_if_signalled(signal: Any) -> None:
    if signal is not None and getattr(signal, "aborted", False):
        raise AbortError("Operation aborted")


def empty_user_message(text: str, timestamp: int) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


def assistant_error(api: str, provider: str, model: str, message: str, timestamp: int) -> AssistantMessage:
    from .types import Usage

    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model,
        usage=Usage(),
        stop_reason="error",
        timestamp=timestamp,
        error_message=message,
    )
