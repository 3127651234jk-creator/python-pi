"""Typed errors.

Python port of ``packages/durable/src/errors.ts``.
"""

from __future__ import annotations

from typing import Any, Optional


def error_message(error: Any) -> str:
    """Format an error message without ever throwing (port of errorMessage)."""
    try:
        if isinstance(error, BaseException):
            return str(error) or type(error).__name__
        return str(error)
    except Exception:
        return "<unprintable error>"


class ReadAfterWrite(Exception):
    """A transaction read tables after its first table write."""


class StorageRejected(Exception):
    """Storage rejected an invalid read or commit request."""


class SessionFailed(Exception):
    """The session is terminal after a storage error; close and reopen it."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__("Session failed; close and reopen the storage", )
        self.cause = cause


class ConversationBusy(Exception):
    """A submission reached a busy conversation with when_busy="reject"."""

    def __init__(self, conversation_id: int) -> None:
        super().__init__(f"Conversation {conversation_id} is busy")
        self.conversation_id = conversation_id
