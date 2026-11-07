"""Per-subscription state codec registry (port of ``chord/src/services/state-codec.ts``).

One pair of registries serves one service subscription: every replicated state
of every instance gets its own delta encoder/decoder, keyed by
``(instance key, generation, member)``. Snapshots, resets and replacements
rebuild the registry; ``closed`` drops one instance's codecs; encoding or
decoding for an unknown state is an error, never a silent skip.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Tuple

from ..delta import WireDecoder, WireEncoder


class _StateCodecRegistry:
    def __init__(self, create: Callable[[], Any]) -> None:
        self._create = create
        self._entries: Dict[str, Tuple[Optional[Dict[str, Any]], Any]] = {}

    def reset(self) -> None:
        self._entries.clear()

    def add(self, instance: Optional[Dict[str, Any]], member: str) -> Any:
        key = _state_key(instance, member)
        if key in self._entries:
            raise ValueError(f"Duplicate service state {_describe(instance, member)}")
        codec = self._create()
        self._entries[key] = (instance, codec)
        return codec

    def get(self, instance: Optional[Dict[str, Any]], member: str) -> Any:
        entry = self._entries.get(_state_key(instance, member))
        if entry is None:
            raise ValueError(f"Unknown service state {_describe(instance, member)}")
        return entry[1]

    def remove_instance(self, instance: Dict[str, Any]) -> None:
        for key, entry in list(self._entries.items()):
            if _same_address(entry[0], instance):
                del self._entries[key]


class StateEncoder:
    """Encodes snapshots and updates of one subscription onto the wire."""

    def __init__(self) -> None:
        self._codecs = _StateCodecRegistry(WireEncoder)

    def encode_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        self._codecs.reset()
        return {**snapshot, "instances": [self._encode_instance(i) for i in snapshot["instances"]]}

    def encode_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        kind = update["type"]
        if kind == "reset":
            self._codecs.reset()
            return {**update, "snapshot": {
                **update["snapshot"],
                "instances": [self._encode_instance(i) for i in update["snapshot"]["instances"]],
            }}
        if kind == "state":
            codec = self._codecs.get(update.get("instance"), update["member"])
            return {**update, "ops": codec.encode(update["ops"])}
        if kind == "replaced":
            self._codecs.reset()
            return {**update, "snapshot": self._encode_instance(update["snapshot"])}
        if kind == "spawned":
            return {**update, "instance": self._encode_instance(update["instance"])}
        if kind == "unavailable":
            self._codecs.reset()
            return update
        if kind == "closed":
            self._codecs.remove_instance(update["instance"])
            return update
        raise ValueError("Invalid service provider update")

    def _encode_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        members = []
        for member in instance["members"]:
            if member.get("kind") == "state":
                codec = self._codecs.add(instance.get("instance"), member["name"])
                members.append({**member, "ops": codec.encode(member["ops"])})
            else:
                members.append(member)
        return {**instance, "members": members}


class StateDecoder:
    """Decodes wire snapshots and updates of one subscription."""

    def __init__(self) -> None:
        self._codecs = _StateCodecRegistry(WireDecoder)

    def decode_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        self._codecs.reset()
        return {**snapshot, "instances": [self._decode_instance(i) for i in snapshot["instances"]]}

    def decode_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        kind = update["type"]
        if kind == "reset":
            self._codecs.reset()
            return {**update, "snapshot": {
                **update["snapshot"],
                "instances": [self._decode_instance(i) for i in update["snapshot"]["instances"]],
            }}
        if kind == "state":
            codec = self._codecs.get(update.get("instance"), update["member"])
            return {**update, "ops": codec.decode(update["ops"])}
        if kind == "replaced":
            self._codecs.reset()
            return {**update, "snapshot": self._decode_instance(update["snapshot"])}
        if kind == "spawned":
            return {**update, "instance": self._decode_instance(update["instance"])}
        if kind == "unavailable":
            self._codecs.reset()
            return update
        if kind == "closed":
            self._codecs.remove_instance(update["instance"])
            return update
        raise ValueError("Invalid service provider update")

    def _decode_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        members = []
        for member in instance["members"]:
            if member.get("kind") == "state":
                codec = self._codecs.add(instance.get("instance"), member["name"])
                members.append({**member, "ops": codec.decode(member["ops"])})
            else:
                members.append(member)
        return {**instance, "members": members}


def _state_key(instance: Optional[Dict[str, Any]], member: str) -> str:
    key = instance.get("key") if instance else None
    generation = instance.get("generation") if instance else None
    return json.dumps([key, generation, member])


def _same_address(left: Optional[Dict[str, Any]], right: Dict[str, Any]) -> bool:
    if left is None:
        return False
    return left.get("key") == right.get("key") and left.get("generation") == right.get("generation")


def _describe(instance: Optional[Dict[str, Any]], member: str) -> str:
    if instance is None:
        return member
    return f"{instance.get('key')}@{instance.get('generation')}.{member}"
