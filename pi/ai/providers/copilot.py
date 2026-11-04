"""Request-local Copilot initiator and vision headers."""

from ..types import ImageContent


def copilot_headers(model, context):
    if model.provider != "github-copilot":
        return {}
    messages = [message for message in context.messages if message.role != "system"]
    headers = {"X-Initiator": "user" if messages and messages[-1].role == "user" else "agent",
               "Openai-Intent": "conversation-edits"}
    if any(isinstance(block, ImageContent) for message in messages if isinstance(message.content, list) for block in message.content):
        headers["Copilot-Vision-Request"] = "true"
    return headers
