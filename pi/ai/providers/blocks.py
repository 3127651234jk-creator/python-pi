"""Shared normalized content-block events for provider adapters."""

import json

from ..types import (TextContent, ThinkingContent, ToolCall, TextStartEvent, TextDeltaEvent,
    TextEndEvent, ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent, ToolCallStartEvent,
    ToolCallDeltaEvent, ToolCallEndEvent, DoneEvent)


class BlockEvents:
    def __init__(self, state):
        self.state = state
        self.indices = {}
        self.arguments = {}

    def text(self, key, delta, thinking=False):
        events = []
        if key not in self.indices:
            self.indices[key] = len(self.state.partial.content)
            self.state.partial.content.append(ThinkingContent(thinking="") if thinking else TextContent(text=""))
            cls = ThinkingStartEvent if thinking else TextStartEvent
            events.append(cls(content_index=self.indices[key], partial=self.state.snapshot()))
        index = self.indices[key]
        block = self.state.partial.content[index]
        if thinking:
            block.thinking += delta
        else:
            block.text += delta
        cls = ThinkingDeltaEvent if thinking else TextDeltaEvent
        events.append(cls(content_index=index, delta=delta, partial=self.state.snapshot()))
        return events

    def tool(self, key, call_id, name, delta=""):
        events = []
        if key not in self.indices:
            self.indices[key] = len(self.state.partial.content)
            self.state.partial.content.append(ToolCall(id=call_id, name=name, arguments={}))
            self.arguments[key] = ""
            events.append(ToolCallStartEvent(content_index=self.indices[key], partial=self.state.snapshot()))
        index = self.indices[key]
        block = self.state.partial.content[index]
        if call_id:
            block.id = call_id
        if name:
            block.name = name
        self.arguments[key] += delta
        try:
            block.arguments = json.loads(self.arguments[key] or "{}")
        except ValueError:
            block.arguments = {"__raw__": self.arguments[key]}
        if delta:
            events.append(ToolCallDeltaEvent(content_index=index, delta=delta, partial=self.state.snapshot()))
        return events

    def finish(self, reason):
        events = []
        for index, block in enumerate(self.state.partial.content):
            if isinstance(block, TextContent):
                events.append(TextEndEvent(content_index=index, content=block.text, partial=self.state.snapshot()))
            elif isinstance(block, ThinkingContent):
                events.append(ThinkingEndEvent(content_index=index, content=block.thinking, partial=self.state.snapshot()))
            else:
                events.append(ToolCallEndEvent(content_index=index, tool_call=block, partial=self.state.snapshot()))
        self.state.partial.stop_reason = reason
        events.append(DoneEvent(reason=reason, message=self.state.snapshot()))
        return events
