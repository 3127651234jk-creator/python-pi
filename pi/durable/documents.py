"""Document definitions.

Python port of ``packages/durable/src/documents.ts`` (runtime parts): definition
validation, logical address resolution, and materialization. Porting deviation:
document content is stored as full base snapshots (no chord delta batches);
``checkpoint_when`` therefore always stores a base and ``snapshot_as_of`` reads
historical bases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import DocumentRecord, StoredDocument


@dataclass
class DocDefinition:
    """A document definition: identity, version, initial value, and migration."""

    kind: str
    version: int
    scope: str  # "session" | "conversation" | "task"
    initial: Callable[[Any], Dict[str, Any]]
    family: bool = False
    migrate: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or self.version <= 0:
            raise ValueError(f"Document definition {self.kind} version must be a positive integer")


def define_doc(kind: str, version: int, scope: str, initial: Callable[[Any], Dict[str, Any]], migrate=None) -> DocDefinition:
    return DocDefinition(kind=kind, version=version, scope=scope, initial=initial, migrate=migrate)


def define_doc_family(
    kind: str, version: int, scope: str, initial: Callable[[Any], Dict[str, Any]], migrate=None
) -> DocDefinition:
    return DocDefinition(kind=kind, version=version, scope=scope, initial=initial, family=True, migrate=migrate)


def scope_of(address_args: Tuple[Any, ...]) -> str:
    """Infer the scope kind from resolved address args (id-bearing or string key)."""
    for arg in address_args:
        if isinstance(arg, dict) and arg.get("kind") in ("session", "conversation", "task"):
            return arg["kind"]
    raise ValueError("Cannot infer document scope from arguments")


def resolve_address(definition: DocDefinition, args: List[Any]) -> Tuple[str, Dict[str, Any], int]:
    """Resolve a logical document address from positional args.

    Conversation/task scopes consume one owner id first; family definitions then
    consume a string key. Returns ``(address_id, address, next_arg_index)``.
    """

    index = 0
    scope: Dict[str, Any]
    if definition.scope == "session":
        scope = {"kind": "session"}
    elif definition.scope == "conversation":
        owner = args[index]
        index += 1
        scope = {"kind": "conversation", "conversation_id": int(owner)}
    elif definition.scope == "task":
        owner = args[index]
        index += 1
        scope = {"kind": "task", "task_id": int(owner)}
    else:
        raise ValueError(f"Unknown document scope: {definition.scope}")

    key: Optional[str] = None
    if definition.family:
        key = args[index]
        if not isinstance(key, str):
            raise ValueError(f"Family document {definition.kind} requires a string key")
        index += 1

    address = {"kind": definition.kind, "scope": scope, "key": key}
    return _address_id(address), address, index


def _address_id(address: Dict[str, Any]) -> str:
    scope = address["scope"]
    if scope["kind"] == "session":
        scope_id = "session"
    elif scope["kind"] == "conversation":
        scope_id = f"conversation:{scope['conversation_id']}"
    else:
        scope_id = f"task:{scope['task_id']}"
    key = f"family:{address['key']}" if address.get("key") is not None else "singleton"
    return f"{scope_id}|{address['kind']}|{key}"


def check_record_scope(definition: DocDefinition, record: DocumentRecord) -> None:
    scope_kind = record.scope.get("kind")
    if scope_kind != definition.scope:
        raise ValueError(f"Document {record.id} scope {scope_kind} does not match definition {definition.kind}")


def check_record_version(definition: DocDefinition, record: DocumentRecord, stored_version: int) -> None:
    if stored_version > definition.version:
        raise ValueError(
            f"Document {record.id} ({record.kind}) was stored by version {stored_version}, "
            f"newer than definition version {definition.version}"
        )


def initial_value(definition: DocDefinition, seed: Any) -> Dict[str, Any]:
    return definition.initial(seed)


def materialize_document(definition: DocDefinition, stored: StoredDocument) -> Dict[str, Any]:
    """Return the document value, migrating when the stored version is older."""
    value = stored.value
    if stored.version < definition.version and definition.migrate is not None:
        value = definition.migrate(dict(value), stored.version)
    return value
