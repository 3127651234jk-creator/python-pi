"""Chord service provider (compact port of ``chord/src/services/provider.ts``).

Hosts one registry of services for one remote consumer. Methods are callables
and states are :class:`ReplicatedState` instances on the implementation object;
the provider classifies public members (``_``-prefixed attributes are private),
answers ``$chord.service`` control calls through :class:`RemoteServiceEndpoint`,
and fans state publications out to subscribers with a 100-update buffer that
collapses into a full ``reset`` on overflow.

Deviations from TS: the tracker/diff layer is not ported — states publish
explicit ops (``publish``) or root replacements (``replace``); singleton
``withdraw``/``replace`` are not ported.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Set

from .wire import RemoteServiceError, decode_control_call

SUBSCRIBER_BUFFER_LIMIT = 100


class ServiceContext:
    """Trailing argument every chord method receives; mirrors the TS ``Context``."""

    def __init__(self, cancelled: Optional[asyncio.Event] = None) -> None:
        self.cancelled = cancelled if cancelled is not None else asyncio.Event()


class ReplicatedState:
    """Authoritative replicated state; ops are supplied by the owner."""

    def __init__(self, initial: Any) -> None:
        self._value = initial
        self._sequence = 0
        self._listeners: List[Callable[[List[Any], int], None]] = []

    @property
    def value(self) -> Any:
        return self._value

    def snapshot(self) -> Dict[str, Any]:
        return {"value": self._value, "sequence": self._sequence}

    def publish(self, value: Any, ops: List[Any]) -> None:
        """Adopt the next value and publish one incremental revision."""
        if not ops:
            return
        self._value = value
        self._sequence += 1
        for listener in list(self._listeners):
            listener(list(ops), self._sequence)

    def replace(self, value: Any) -> None:
        """Publish a root replacement; a no-op when the value is deep-equal."""
        if value == self._value:
            return
        self.publish(value, [["r", value]])

    def subscribe_source(self, listener: Callable[[List[Any], int], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def remove() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return remove


class ServiceSubscription:
    """One consumer subscription: an atomic snapshot plus buffered updates until activation."""

    def __init__(self, snapshot: Dict[str, Any], activate: Callable[[], None],
                 close: Callable[[], None]) -> None:
        self.snapshot = snapshot
        self._activate = activate
        self._close = close

    def activate(self) -> None:
        self._activate()

    def close(self) -> None:
        self._close()


class _Subscriber:
    """One provider-side subscription (hashable identity for the subscriber set)."""

    __slots__ = ("listener", "buffer", "snapshot_sequences", "active", "draining", "closed")

    def __init__(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        self.listener = listener
        self.buffer: List[Dict[str, Any]] = []
        self.snapshot_sequences: Dict[str, int] = {}
        self.active = False
        self.draining = False
        self.closed = False


class RemoteServiceProvider:
    """Registry of remotely published services; one instance per remote consumer."""

    def __init__(self, entries: List[Dict[str, Any]]) -> None:
        definitions = []
        for entry in entries:
            if "service" in entry:
                definitions.append({"service": entry["service"], "mode": entry.get("mode", "singleton")})
            else:
                definitions.append({"service": {"id": entry["id"], "local": False},
                                    "mode": entry.get("mode", "singleton")})
        for definition in definitions:
            if definition["service"].get("local"):
                raise TypeError(f"Local service {definition['service']['id']} cannot be published remotely")
        ids = [definition["service"]["id"] for definition in definitions]
        if len(set(ids)) != len(ids):
            raise TypeError("Remote service catalogue contains duplicate IDs")
        self.catalogue = [{"serviceId": definition["service"]["id"], "mode": definition["mode"]}
                          for definition in definitions]
        self._registrations: Dict[str, Dict[str, Any]] = {}
        for definition in definitions:
            self._registrations[definition["service"]["id"]] = {
                "serviceId": definition["service"]["id"],
                "mode": definition["mode"],
                "singleton": None,
                "instances": {},
                "generations": {},
                "subscribers": set(),
            }
        self._disposed = False

    def provide(self, service_id: str, implementation: Any) -> None:
        registration = self._registration(service_id, "singleton")
        if registration["singleton"] is not None:
            raise RemoteServiceError("service_mode_mismatch", f"Remote service {service_id} already has a provider")
        registration["singleton"] = self._create_instance(registration, _classify(service_id, implementation), None)

    def spawn(self, service_id: str, key: str, implementation: Any) -> Callable[[], None]:
        if not key:
            raise TypeError("Remote service instance key must not be empty")
        registration = self._registration(service_id, "keyed")
        if key in registration["instances"]:
            raise RemoteServiceError("service_mode_mismatch",
                                     f"Remote service {service_id} already has a live instance with key {key}")
        generation = registration["generations"].get(key, 0) + 1
        registration["generations"][key] = generation
        address = {"key": key, "generation": generation}
        instance = self._create_instance(registration, _classify(service_id, implementation), address)
        registration["instances"][key] = instance
        self._emit(registration, {"type": "spawned", "instance": self._snapshot_instance(instance)})
        closed = False

        def close() -> None:
            nonlocal closed
            if closed or registration["instances"].get(key) is not instance:
                return
            closed = True
            del registration["instances"][key]
            self._emit(registration, {"type": "closed", "instance": address})

        return close

    async def invoke(self, call: Dict[str, Any], context: ServiceContext) -> Any:
        self._assert_active()
        registration = self._registrations.get(call["serviceId"])
        if registration is None:
            raise RemoteServiceError("service_not_found", f"Unknown remote service {call['serviceId']}")
        instance = self._resolve_instance(registration, call.get("instance"))
        member = instance["members"].get(call["member"])
        if member is None:
            raise RemoteServiceError("service_member_not_found",
                                     f"Unknown remote service member {call['serviceId']}.{call['member']}")
        if member["kind"] != "method":
            raise RemoteServiceError("service_member_mismatch",
                                     f"Remote service member {call['serviceId']}.{call['member']} is not a method")
        result = member["method"](*call["args"], context)
        if inspect.isawaitable(result):
            result = await result
        return result

    def subscribe(self, service_id: str, mode: str,
                  listener: Callable[[Dict[str, Any]], None]) -> ServiceSubscription:
        self._assert_active()
        registration = self._registration(service_id, mode)
        if registration["mode"] == "singleton" and registration["singleton"] is None:
            raise RemoteServiceError("service_not_found", f"Remote service {service_id} has no provider")
        subscriber = _Subscriber(listener)
        registration["subscribers"].add(subscriber)
        snapshot = self._snapshot(registration)
        _record_snapshot_sequences(subscriber.snapshot_sequences, snapshot["instances"])

        def activate() -> None:
            if subscriber.closed or subscriber.active:
                return
            subscriber.active = True
            _drain_subscriber(subscriber)

        def close() -> None:
            if subscriber.closed:
                return
            subscriber.closed = True
            subscriber.buffer.clear()
            registration["subscribers"].discard(subscriber)

        return ServiceSubscription(snapshot, activate, close)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for registration in self._registrations.values():
            if registration["singleton"] is not None:
                registration["singleton"] = None
                self._emit(registration, {"type": "unavailable"})
            for key in list(registration["instances"]):
                instance = registration["instances"].pop(key)
                self._emit(registration, {"type": "closed", "instance": instance["address"]})
            for subscriber in list(registration["subscribers"]):
                subscriber.closed = True
                subscriber.buffer.clear()
            registration["subscribers"].clear()
        self._registrations.clear()

    # ------------------------------------------------------------------ internal

    def _registration(self, service_id: str, mode: Optional[str]) -> Dict[str, Any]:
        registration = self._registrations.get(service_id)
        if registration is None:
            raise RemoteServiceError("service_not_found", f"Unknown remote service {service_id}")
        if mode is not None and registration["mode"] != mode:
            raise RemoteServiceError("service_mode_mismatch",
                                     f"Remote service {service_id} is {registration['mode']}, not {mode}")
        return registration

    def _create_instance(self, registration: Dict[str, Any], members: Dict[str, Dict[str, Any]],
                         address: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        instance: Dict[str, Any] = {"address": address, "members": members, "active": True}
        for name, member in members.items():
            if member["kind"] != "state":
                continue

            def listener(ops: List[Any], sequence: int, member_name: str = name) -> None:
                if not instance["active"]:
                    return
                update: Dict[str, Any] = {"type": "state", "member": member_name,
                                          "sequence": sequence, "ops": ops}
                if address is not None:
                    update["instance"] = dict(address)
                self._emit(registration, update)

            member["state"].subscribe_source(listener)
        return instance

    def _resolve_instance(self, registration: Dict[str, Any],
                          address: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if registration["mode"] == "singleton":
            if address is not None:
                raise RemoteServiceError("service_mode_mismatch",
                                         f"Remote service {registration['serviceId']} is singleton")
            if registration["singleton"] is None:
                raise RemoteServiceError("service_not_found",
                                         f"Remote service {registration['serviceId']} has no provider")
            return registration["singleton"]
        if address is None:
            raise RemoteServiceError("service_mode_mismatch",
                                     f"Remote service {registration['serviceId']} is keyed")
        instance = registration["instances"].get(address["key"])
        if instance is None:
            raise RemoteServiceError("service_instance_not_found",
                                     f"Remote service {registration['serviceId']} has no instance {address['key']}")
        if instance["address"]["generation"] != address["generation"]:
            raise RemoteServiceError("service_stale_instance",
                                     f"Remote service {registration['serviceId']} instance {address['key']} is stale")
        return instance

    def _snapshot(self, registration: Dict[str, Any]) -> Dict[str, Any]:
        if registration["mode"] == "singleton":
            instances = [self._snapshot_instance(registration["singleton"])] \
                if registration["singleton"] is not None else []
        else:
            instances = [self._snapshot_instance(registration["instances"][key])
                         for key in sorted(registration["instances"])]
        return {"serviceId": registration["serviceId"], "mode": registration["mode"], "instances": instances}

    def _snapshot_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        members = []
        for name, member in instance["members"].items():
            if member["kind"] == "method":
                members.append({"name": name, "kind": "method"})
            else:
                state = member["state"].snapshot()
                members.append({"name": name, "kind": "state", "sequence": state["sequence"],
                                "ops": [["r", state["value"]]]})
        snapshot: Dict[str, Any] = {"members": members}
        if instance["address"] is not None:
            snapshot["instance"] = dict(instance["address"])
        return snapshot

    def _emit(self, registration: Dict[str, Any], update: Dict[str, Any]) -> None:
        if not registration["subscribers"]:
            return
        errors: List[Any] = []
        subscribers = list(registration["subscribers"])
        # Queue for everyone before invoking user code, including reentrant publications.
        for subscriber in subscribers:
            if subscriber.closed or _update_covered_by_snapshot(subscriber.snapshot_sequences, update):
                continue
            if len(subscriber.buffer) == SUBSCRIBER_BUFFER_LIMIT:
                snapshot = self._snapshot(registration)
                subscriber.buffer.clear()
                subscriber.snapshot_sequences.clear()
                _record_snapshot_sequences(subscriber.snapshot_sequences, snapshot["instances"])
                subscriber.buffer.append({"type": "reset", "snapshot": snapshot})
            else:
                subscriber.buffer.append(update)
        for subscriber in subscribers:
            errors.extend(_drain_subscriber(subscriber))
        if errors:
            raise errors[0]

    def _assert_active(self) -> None:
        if self._disposed:
            raise RuntimeError("Remote service provider is disposed")


def _drain_subscriber(subscriber: "_Subscriber") -> List[Any]:
    if not subscriber.active or subscriber.closed or subscriber.draining:
        return []
    errors: List[Any] = []
    subscriber.draining = True
    try:
        while not subscriber.closed:
            if not subscriber.buffer:
                break
            update = subscriber.buffer.pop(0)
            try:
                subscriber.listener(update)
            except Exception as error:  # noqa: BLE001 - collected, then re-raised
                errors.append(error)
    finally:
        subscriber.draining = False
    return errors


def _record_snapshot_sequences(sequences: Dict[str, int], instances: List[Dict[str, Any]]) -> None:
    for instance in instances:
        for member in instance["members"]:
            if member.get("kind") == "state":
                sequences[_state_member_key(instance.get("instance"), member["name"])] = member["sequence"]


def _update_covered_by_snapshot(sequences: Dict[str, int], update: Dict[str, Any]) -> bool:
    kind = update.get("type")
    if kind == "state":
        key = _state_member_key(update.get("instance"), update["member"])
        sequence = sequences.get(key)
        if sequence is None:
            return False
        if update["sequence"] <= sequence:
            return True
        del sequences[key]
        return False
    if kind == "reset":
        sequences.clear()
        _record_snapshot_sequences(sequences, update["snapshot"]["instances"])
        return False
    if kind == "replaced":
        sequences.clear()
        _record_snapshot_sequences(sequences, [update["snapshot"]])
        return False
    if kind == "spawned":
        _record_snapshot_sequences(sequences, [update["instance"]])
        return False
    if kind == "unavailable":
        sequences.clear()
        return False
    if kind == "closed":
        address = update["instance"]
        prefix = json.dumps([address["key"], address["generation"]])[:-1] + ","
        for key in list(sequences):
            if key.startswith(prefix):
                del sequences[key]
        return False
    return False


def _state_member_key(instance: Optional[Dict[str, Any]], member: str) -> str:
    if instance is None:
        return json.dumps([member])
    return json.dumps([instance["key"], instance["generation"], member])


def _classify(service_id: str, implementation: Any) -> Dict[str, Dict[str, Any]]:
    """Classify public members: callables are methods, ``ReplicatedState`` values are states."""
    members: Dict[str, Dict[str, Any]] = {}
    for name in sorted(dir(implementation)):
        if name.startswith("_"):
            continue
        value = getattr(implementation, name)
        if callable(value):
            members[name] = {"kind": "method", "method": value}
            continue
        if isinstance(value, ReplicatedState):
            members[name] = {"kind": "state", "state": value}
            continue
        raise TypeError(f"Remote service member {service_id}.{name} is not remotely exposable")
    if not members:
        raise TypeError(f"Remote service {service_id} has no members")
    return members


class RemoteServiceEndpoint:
    """Serves one consumer: control calls plus business calls, publishing to ``publish``."""

    def __init__(self, provider: RemoteServiceProvider) -> None:
        self._provider = provider
        self._subscriptions: Dict[str, ServiceSubscription] = {}
        self._disposed = False

    async def invoke_service(self, call: Dict[str, Any], publish: Callable[[str, Dict[str, Any]], Any],
                             cancelled: Optional[asyncio.Event] = None) -> Any:
        if self._disposed:
            raise RuntimeError("Remote service endpoint is disposed")
        control = decode_control_call(call)
        if control is not None and control["type"] == "catalogue":
            return self._provider.catalogue
        if control is not None and control["type"] == "subscribe":
            subscription_id = control["subscriptionId"]
            if subscription_id in self._subscriptions:
                raise ValueError("Service subscription ID is already active")
            subscription = self._provider.subscribe(
                control["serviceId"], control["mode"],
                lambda update: self._forward(publish, subscription_id, update))
            self._subscriptions[subscription_id] = subscription
            subscription.activate()
            return subscription.snapshot
        if control is not None and control["type"] == "unsubscribe":
            subscription = self._subscriptions.pop(control["subscriptionId"], None)
            if subscription is None:
                raise RuntimeError("Service subscription was not found")
            subscription.close()
            return None
        return await self._provider.invoke(call, ServiceContext(cancelled))

    def _forward(self, publish: Callable[[str, Dict[str, Any]], Any], subscription_id: str,
                 update: Dict[str, Any]) -> None:
        task = asyncio.create_task(self._publish(publish, subscription_id, update))
        # Fire-and-forget like the TS endpoint; failures must not break the provider drain.
        task.add_done_callback(_swallow_task_error)

    async def _publish(self, publish: Callable[[str, Dict[str, Any]], Any], subscription_id: str,
                       update: Dict[str, Any]) -> None:
        try:
            await publish(subscription_id, update)
        except Exception:  # noqa: BLE001 - publish failures never break the provider
            pass

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for subscription in self._subscriptions.values():
            subscription.close()
        self._subscriptions.clear()


def _swallow_task_error(task: "asyncio.Task[Any]") -> None:
    if not task.cancelled() and task.exception() is not None:
        pass  # Publish failures are the transport's concern; the endpoint ignores them.
