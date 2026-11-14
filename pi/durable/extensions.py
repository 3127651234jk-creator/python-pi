"""Named extensions selected by each conversation, with ordered hooks and wrappers."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..agent.types import AgentTool
from .documents import DocDefinition, define_doc
from .types import TaskDefinition, write_task

AgentDoc = define_doc("pi.agent", 1, "conversation", lambda seed: {"extensions": [], "instructions": ""})
ProviderDoc = define_doc("pi.provider", 1, "conversation", lambda seed: {})
ProviderDoc.history, ProviderDoc.fork = "latest", "initial"


@dataclass
class Extension:
    name: str
    tools: List[AgentTool] = field(default_factory=list)
    tasks: List[TaskDefinition] = field(default_factory=list)
    hooks: Dict[str, Callable] = field(default_factory=dict)
    sections: Dict[str, Callable] = field(default_factory=dict)
    wraps: Dict[str, Callable] = field(default_factory=dict)
    documents: List[DocDefinition] = field(default_factory=list)


class Registry:
    def __init__(self, extensions: Optional[List[Extension]] = None) -> None:
        self._extensions: Dict[str, Extension] = {}
        for extension in extensions or []:
            self.install(extension)

    def install(self, extension: Extension) -> None:
        if not extension.name:
            raise ValueError("extension name must not be empty")
        self._extensions[extension.name] = extension

    def uninstall(self, extension: Extension) -> None:
        self._extensions.pop(extension.name, None)

    def installed(self) -> List[Extension]:
        return list(self._extensions.values())

    def extension(self, name: str) -> Optional[Extension]:
        return self._extensions.get(name)


class HookApi:
    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.task_id = runtime.task_id
        self.conversation_id = getattr(runtime, "conversation_id", None)
        self.session = runtime._harness.session

    async def snapshot(self, definition: DocDefinition, *args):
        return await self.session.snapshot(definition, *args)

    async def memo(self, name: str, *candidate):
        if len(candidate) > 1:
            raise TypeError("memo accepts at most one candidate")
        result = []
        async def change(tx, current):
            memos = dict(current.memos or {})
            if name not in memos and candidate:
                memos[name] = copy.deepcopy(candidate[0])
                current.memos = memos
                tx._writes.append(write_task(current))
                tx._mark_table_write()
            result.append(copy.deepcopy(memos.get(name)))
        await self.runtime.commit(change)
        return result[0]


async def invoke(callback: Callable, *args):
    value = callback(*args)
    return await value if inspect.isawaitable(value) else value
