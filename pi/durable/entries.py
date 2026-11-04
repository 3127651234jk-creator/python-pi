"""Entry kinds.

Python port of ``packages/durable/src/entries.ts``: typed entry factories with
a narrowing guard, plus the six built-in ``pi.*`` entry kinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import EntryRecord


@dataclass
class Entry:
    """Typed entry kind; ``kind`` names the discriminator, ``data_type`` documents the payload."""

    kind: str
    data_type: Any = None

    def is_(self, entry: Optional[EntryRecord]) -> bool:
        return entry is not None and entry.kind == self.kind


def define_entry(kind: str, data_type: Any = None) -> Entry:
    return Entry(kind=kind, data_type=data_type)


USER_ENTRY = define_entry("pi.user")
ASSISTANT_ENTRY = define_entry("pi.assistant")
SYSTEM_ENTRY = define_entry("pi.system")
TOOL_RESULT_ENTRY = define_entry("pi.tool-result")
RESET_ENTRY = define_entry("pi.reset")
COMPACTION_ENTRY = define_entry("pi.compaction")

BUILTIN_ENTRIES: Dict[str, Entry] = {
    e.kind: e for e in (USER_ENTRY, ASSISTANT_ENTRY, SYSTEM_ENTRY, TOOL_RESULT_ENTRY, RESET_ENTRY, COMPACTION_ENTRY)
}
