"""Chord service wire vocabulary (port of ``chord/src/services/wire.ts`` and ``errors.ts``).

Service calls are strict JSON: ``{serviceId, member, args}`` with an optional
``{key, generation}`` instance address. Subscription management rides on the
reserved pseudo-service ``$chord.service`` (catalogue/subscribe/unsubscribe).
There is no version field: compatibility is enforced by strict structural
validation, so unknown keys and unknown update types are rejected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..delta import assert_valid_op, assert_valid_wire_op

REMOTE_SERVICE_ERROR_CODES = (
    "service_not_allowed",
    "service_not_found",
    "service_mode_mismatch",
    "service_member_not_found",
    "service_member_mismatch",
    "service_instance_not_found",
    "service_stale_instance",
    "service_invalid_value",
)

SERVICE_CONTROL_ID = "$chord.service"


class RemoteServiceError(Exception):
    """A remote-service failure whose code travels as the protocol error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        if code not in REMOTE_SERVICE_ERROR_CODES:
            raise ValueError(f"Unknown remote service error code: {code}")
        self.code = code


def create_catalogue_call() -> Dict[str, Any]:
    return {"serviceId": SERVICE_CONTROL_ID, "member": "catalogue", "args": []}


def create_subscribe_call(subscription_id: str, service_id: str, mode: str) -> Dict[str, Any]:
    return {"serviceId": SERVICE_CONTROL_ID, "member": "subscribe",
            "args": [subscription_id, service_id, mode]}


def create_unsubscribe_call(subscription_id: str) -> Dict[str, Any]:
    return {"serviceId": SERVICE_CONTROL_ID, "member": "unsubscribe", "args": [subscription_id]}


def decode_control_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Returns the control-call description, or None for a business call."""
    if call.get("serviceId") != SERVICE_CONTROL_ID or call.get("instance") is not None:
        return None
    member = call.get("member")
    args = call.get("args")
    if member == "catalogue" and args == []:
        return {"type": "catalogue"}
    if member == "subscribe" and isinstance(args, list) and len(args) == 3 \
            and _is_id(args[0]) and _is_id(args[1]) and args[2] in ("singleton", "keyed"):
        return {"type": "subscribe", "subscriptionId": args[0], "serviceId": args[1], "mode": args[2]}
    if member == "unsubscribe" and isinstance(args, list) and len(args) == 1 and _is_id(args[0]):
        return {"type": "unsubscribe", "subscriptionId": args[0]}
    return None


def parse_service_call(value: Any) -> Dict[str, Any]:
    call = _record(value, "service call")
    _assert_keys(call, ("serviceId", "member", "args"), ("instance",), "service call")
    if not _is_id(call["serviceId"]) or not _is_id(call["member"]) or not isinstance(call["args"], list):
        raise TypeError("Invalid service call")
    if call.get("instance") is not None:
        _assert_address(call["instance"])
    return value


def parse_service_catalogue(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Invalid service catalogue")
    seen = set()
    for candidate in value:
        entry = _record(candidate, "service catalogue entry")
        _assert_keys(entry, ("serviceId", "mode"), (), "service catalogue entry")
        if not _is_id(entry["serviceId"]) or entry["mode"] not in ("singleton", "keyed") \
                or entry["serviceId"] in seen:
            raise TypeError("Invalid service catalogue")
        seen.add(entry["serviceId"])
    return value


def parse_subscription_snapshot(value: Any, validate_op=assert_valid_op) -> Dict[str, Any]:
    _assert_snapshot(value, validate_op)
    return value


def parse_wire_subscription_snapshot(value: Any) -> Dict[str, Any]:
    return parse_subscription_snapshot(value, assert_valid_wire_op)


def parse_provider_update(value: Any, validate_op=assert_valid_op) -> Dict[str, Any]:
    _assert_update(value, validate_op)
    return value


def parse_wire_provider_update(value: Any) -> Dict[str, Any]:
    return parse_provider_update(value, assert_valid_wire_op)


def _assert_snapshot(value: Any, validate_op) -> None:
    snapshot = _record(value, "service subscription snapshot")
    _assert_keys(snapshot, ("serviceId", "mode", "instances"), (), "service subscription snapshot")
    if not _is_id(snapshot["serviceId"]) or snapshot["mode"] not in ("singleton", "keyed") \
            or not isinstance(snapshot["instances"], list):
        raise TypeError("Invalid service subscription snapshot")
    for instance in snapshot["instances"]:
        _assert_instance(instance, validate_op)


def _assert_update(value: Any, validate_op) -> None:
    update = _record(value, "service provider update")
    kind = update.get("type")
    if kind == "state":
        _assert_keys(update, ("type", "member", "sequence", "ops"), ("instance",), "state update")
        if not _is_id(update["member"]) or not _is_integer(update["sequence"], 1) \
                or not isinstance(update["ops"], list):
            raise TypeError("Invalid service state update")
        if update.get("instance") is not None:
            _assert_address(update["instance"])
        for op in update["ops"]:
            validate_op(op)
        return
    if kind == "reset":
        _assert_keys(update, ("type", "snapshot"), (), "reset update")
        _assert_snapshot(update["snapshot"], validate_op)
        for instance in update["snapshot"]["instances"]:
            for member in instance["members"]:
                if member.get("kind") == "state" \
                        and (len(member["ops"]) != 1 or member["ops"][0][0] != "r"):
                    raise TypeError("Service reset must contain full root replacements")
        return
    if kind == "unavailable":
        _assert_keys(update, ("type",), (), "unavailable update")
        return
    if kind == "replaced":
        _assert_keys(update, ("type", "snapshot"), (), "replacement update")
        _assert_instance(update["snapshot"], validate_op)
        return
    if kind == "spawned":
        _assert_keys(update, ("type", "instance"), (), "spawn update")
        _assert_instance(update["instance"], validate_op)
        return
    if kind == "closed":
        _assert_keys(update, ("type", "instance"), (), "close update")
        _assert_address(update["instance"])
        return
    raise TypeError("Invalid service provider update")


def _assert_instance(value: Any, validate_op) -> None:
    instance = _record(value, "service instance snapshot")
    _assert_keys(instance, ("members",), ("instance",), "service instance snapshot")
    if instance.get("instance") is not None:
        _assert_address(instance["instance"])
    if not isinstance(instance["members"], list):
        raise TypeError("Invalid service instance snapshot")
    for candidate in instance["members"]:
        member = _record(candidate, "service member snapshot")
        kind = member.get("kind")
        if kind == "method":
            _assert_keys(member, ("name", "kind"), (), "service method snapshot")
            if not _is_id(member["name"]):
                raise TypeError("Invalid service method snapshot")
            continue
        if kind == "state":
            _assert_keys(member, ("name", "kind", "sequence", "ops"), (), "service state snapshot")
            if not _is_id(member["name"]) or not _is_integer(member["sequence"], 0) \
                    or not isinstance(member["ops"], list):
                raise TypeError("Invalid service state snapshot")
            for op in member["ops"]:
                validate_op(op)
            continue
        raise TypeError("Invalid service member snapshot")


def _assert_address(value: Any) -> None:
    address = _record(value, "service instance address")
    _assert_keys(address, ("key", "generation"), (), "service instance address")
    if not _is_id(address["key"]) or not _is_integer(address["generation"], 1):
        raise TypeError("Invalid service instance address")


def _record(value: Any, description: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Invalid {description}")
    return value


def _assert_keys(value: Dict[str, Any], required: Sequence[str], optional: Sequence[str],
                 description: str) -> None:
    allowed = set(required) | set(optional)
    if not set(required) <= value.keys() or not set(value) <= allowed:
        raise TypeError(f"Invalid {description}")


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def _is_integer(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum
