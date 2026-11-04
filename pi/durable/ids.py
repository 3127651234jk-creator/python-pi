"""Typed ID aliases.

Python port of ``packages/durable/src/ids.ts`` and the Id brands from
``types.ts``. Python has no nominal branding; ids are plain ints with alias
names, and ``id_from_number``/``seq_from_number`` stay as the trusted-boundary
assertion points (safe positive integers, zero runtime reinterpretation).
"""

from __future__ import annotations

from typing import NewType

ConversationId = NewType("ConversationId", int)
EntryId = NewType("EntryId", int)
TaskId = NewType("TaskId", int)
SubmissionId = NewType("SubmissionId", int)
DocumentId = NewType("DocumentId", int)
Seq = NewType("Seq", int)

ROOT_CONVERSATION_ID = ConversationId(1)


def id_from_number(value: int):
    """Trust-boundary assertion for a persisted id (safe positive integer)."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"Invalid id: {value!r}")
    return value


def seq_from_number(value: int) -> Seq:
    """Trust-boundary assertion for a persisted commit sequence."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"Invalid seq: {value!r}")
    return Seq(value)
