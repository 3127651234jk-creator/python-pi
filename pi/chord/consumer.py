"""Chord consumer binding (compact port of ``chord/src/services/consumer.ts`` plus the
subscription half of ``packages/client/src/client.ts``).

A binding materializes a service contract over a wire transport: ``use`` returns
a singleton facade whose members call remote methods and hold state replicas;
``observe`` tracks keyed instances through ``spawned``/``closed``. Every replica
hydrates from a base batch, enforces ``sequence == previous + 1`` (a gap clears
the replica and raises), and revalidates strict JSON after each revision.

Deviations from TS: the trailing ``Context`` argument of remote methods is not
modeled (Python requests carry a timeout instead), ``rebind`` is not ported,
and update listeners are plain callbacks. Binding methods require a running
event loop (subscription starts are scheduled eagerly).
"""

from __future__ import annotations

import asyncio
import copy as _copy
from typing import Any, Callable, Dict, List, Optional

from ..delta import RevisionValidator, apply_ops, is_base
from .codec import StateDecoder
from .wire import (
    RemoteServiceError,
    create_subscribe_call,
    create_unsubscribe_call,
    parse_wire_provider_update,
    parse_wire_subscription_snapshot,
)

ErrorReporter = Callable[[Exception], None]


def _ignore_error(error: Exception) -> None:
    pass


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _report_task_errors(task: "asyncio.Task[Any]", report_error: ErrorReporter) -> None:
    def check(done: "asyncio.Task[Any]") -> None:
        if not done.cancelled() and done.exception() is not None:
            report_error(done.exception())

    task.add_done_callback(check)


def _same_address(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.get("key") == right.get("key") and left.get("generation") == right.get("generation")


class Replica:
    """Cold read-only replica of one remote state member."""

    def __init__(self, report_error: ErrorReporter = _ignore_error) -> None:
        self._value: Any = None
        self._sequence: Optional[int] = None
        self._listeners: List[Callable[[Any], None]] = []
        self._validator = RevisionValidator()
        self._report_error = report_error

    @property
    def value(self) -> Any:
        return self._value

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        if self._sequence is not None:
            listener(self._value)

        def remove() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return remove

    def hydrate(self, sequence: int, ops: List[Any]) -> None:
        if not is_base(ops):
            self.clear()
            raise ValueError("Replicated state snapshot is not a base operation batch")
        try:
            value = self._apply(None, ops)
        except Exception:
            self.clear()
            raise
        self._sequence = sequence
        self._value = value
        self._deliver()

    def update(self, sequence: int, ops: List[Any]) -> None:
        if self._sequence is None:
            raise ValueError("Replicated state received an update before hydration")
        if sequence != self._sequence + 1:
            self.clear()
            raise ValueError("Replicated state update sequence has a gap")
        try:
            value = self._apply(self._value, ops)
        except Exception:
            self.clear()
            raise
        self._sequence = sequence
        self._value = value
        self._deliver()

    def clear(self) -> None:
        self._value = None
        self._sequence = None

    def _apply(self, target: Any, ops: List[Any]) -> Any:
        # Fresh value per revision: Python applies onto plain mutable containers,
        # so a copy gives the immutable-revision behavior of TS applyImmutable.
        return self._validator.validate(apply_ops(_copy.deepcopy(target), ops))

    def _deliver(self) -> None:
        if self._sequence is None:
            return
        for listener in list(self._listeners):
            try:
                listener(self._value)
            except Exception as error:  # noqa: BLE001 - listener failures go to the reporter
                self._report_error(error)


class ServiceSubscriptionHandle:
    """A decoded remote service subscription with snapshot-then-update delivery."""

    def __init__(self, subscription_id: str, snapshot: Dict[str, Any], start: Callable[[], None],
                 close: Callable[[], Any]) -> None:
        self.id = subscription_id
        self.snapshot = snapshot
        self._start = start
        self._close = close

    def start(self) -> None:
        self._start()

    async def close(self) -> None:
        await _maybe_await(self._close())


class ClientServiceTransport:
    """Adapts a routed :class:`pi.client.Client` into the chord transport.

    One transport owns the client's shared update queue: a background router
    dispatches ``service_update`` envelopes to subscriptions by id. Updates that
    arrive before the subscribe response are decoded after the snapshot, and
    buffered ones are delivered once ``start()`` runs.
    """

    def __init__(self, client: Any, get_target: Callable[[], Optional[Dict[str, Any]]],
                 on_error: ErrorReporter = _ignore_error) -> None:
        self.client = client
        self._get_target = get_target
        self._report_error = on_error
        self._routes: Dict[str, Callable[[Any], None]] = {}
        self._router_task: Optional[asyncio.Task] = None
        self._sequence = 0

    def _target(self) -> Dict[str, Any]:
        target = self._get_target()
        if target is None:
            raise RuntimeError("Remote service target is unavailable")
        return target

    def _ensure_router(self) -> None:
        if self._router_task is None or self._router_task.done():
            self._router_task = asyncio.create_task(self._route())

    async def _route(self) -> None:
        while True:
            message = await self.client.updates.get()
            sink = self._routes.get(message["subscriptionId"])
            if sink is not None:
                sink(message["update"])

    async def invoke(self, call: Dict[str, Any]) -> Any:
        return await self.client.request(self._target(), call)

    async def subscribe(self, service_id: str, mode: str,
                        listener: Callable[[Dict[str, Any]], Any]) -> ServiceSubscriptionHandle:
        self._sequence += 1
        subscription_id = f"service-{self._sequence}"
        decoder = StateDecoder()
        state: Dict[str, Any] = {"hydrated": False, "wire": [], "ready": False, "queued": []}
        delivery: Dict[str, Any] = {"tail": None}

        def deliver(update: Dict[str, Any]) -> None:
            try:
                result = listener(update)
            except Exception as error:  # noqa: BLE001 - listener failures go to the reporter
                self._report_error(error)
                return
            if asyncio.iscoroutine(result):
                previous = delivery["tail"]

                async def run() -> None:
                    if previous is not None:
                        try:
                            await previous
                        except asyncio.CancelledError:
                            pass
                    try:
                        await result
                    except Exception as error:  # noqa: BLE001
                        self._report_error(error)

                delivery["tail"] = asyncio.create_task(run())

        def on_wire(wire_update: Any) -> None:
            if not state["hydrated"]:
                state["wire"].append(wire_update)
                return
            try:
                update = decoder.decode_update(parse_wire_provider_update(wire_update))
            except Exception as error:  # noqa: BLE001 - a broken delta stream kills the subscription
                self._report_error(error)
                self._routes.pop(subscription_id, None)
                return
            if state["ready"]:
                deliver(update)
            else:
                state["queued"].append(update)

        self._routes[subscription_id] = on_wire
        self._ensure_router()
        try:
            snapshot_wire = await self.client.request(
                self._target(), create_subscribe_call(subscription_id, service_id, mode))
            snapshot = decoder.decode_snapshot(parse_wire_subscription_snapshot(snapshot_wire))
            state["hydrated"] = True
            for wire_update in state["wire"]:
                state["queued"].append(decoder.decode_update(parse_wire_provider_update(wire_update)))
            state["wire"].clear()
        except BaseException:
            self._routes.pop(subscription_id, None)
            raise

        def start() -> None:
            if state["ready"]:
                return
            state["ready"] = True
            for update in state["queued"]:
                deliver(update)
            state["queued"].clear()

        async def close() -> None:
            self._routes.pop(subscription_id, None)
            try:
                await self.client.request(self._target(), create_unsubscribe_call(subscription_id))
            except (ConnectionError, RuntimeError, OSError):
                pass
            if delivery["tail"] is not None:
                try:
                    await delivery["tail"]
                except asyncio.CancelledError:
                    pass

        return ServiceSubscriptionHandle(subscription_id, snapshot, start, close)

    async def dispose(self) -> None:
        if self._router_task is not None:
            self._router_task.cancel()
            try:
                await self._router_task
            except asyncio.CancelledError:
                pass
            self._router_task = None


class MemberHandle:
    """One remote member of a facade: callable for methods, ``value`` for states."""

    def __init__(self, service_id: str, name: str, facade: "ServiceFacade") -> None:
        self._service_id = service_id
        self._name = name
        self._facade = facade
        self._kind: Optional[str] = None
        self._expected: Optional[str] = None
        self._replica = Replica(facade._report_error)

    def _set_kind(self, kind: str) -> None:
        if self._kind is not None and self._kind != kind:
            raise ValueError(f"Remote service member {self._service_id}.{self._name} changed kind")
        self._kind = kind
        if self._expected is not None and self._expected != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._service_id}.{self._name} is {kind}, not {self._expected}")

    def _expect(self, kind: str) -> None:
        if self._expected is not None and self._expected != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._service_id}.{self._name} was used as two different kinds")
        self._expected = kind
        if self._kind is not None and self._kind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._service_id}.{self._name} is {self._kind}, not {kind}")

    def hydrate(self, sequence: int, ops: List[Any]) -> None:
        self._set_kind("state")
        self._replica.hydrate(sequence, ops)

    def update(self, sequence: int, ops: List[Any]) -> None:
        self._set_kind("state")
        self._replica.update(sequence, ops)

    def clear(self) -> None:
        self._replica.clear()

    @property
    def value(self) -> Any:
        self._facade.assert_access()
        self._expect("state")
        return self._replica.value

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        self._expect("state")
        return self._replica.subscribe(listener)

    async def __call__(self, *args: Any) -> Any:
        self._facade.assert_access()
        self._expect("method")
        if not self._facade.is_active():
            raise RemoteServiceError("service_stale_instance",
                                     f"Remote service {self._service_id} binding is closed")
        call: Dict[str, Any] = {"serviceId": self._facade.service_id, "member": self._name, "args": list(args)}
        if self._facade.address is not None:
            call["instance"] = dict(self._facade.address)
        return await self._facade.transport.invoke(call)


class ServiceFacade:
    """Member views over one remote instance; states hydrate from snapshots."""

    def __init__(self, service_id: str, address: Optional[Dict[str, Any]], transport: Any,
                 is_active: Callable[[], bool], assert_access: Callable[[], None],
                 report_error: ErrorReporter = _ignore_error) -> None:
        self.service_id = service_id
        self.address = address
        self.transport = transport
        self.is_active = is_active
        self.assert_access = assert_access
        self._report_error = report_error
        self._slots: Dict[str, MemberHandle] = {}
        self._descriptions: Dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._slot(name)

    def install(self, instance: Dict[str, Any]) -> None:
        if not _same_address(instance.get("instance"), self.address):
            raise ValueError("Remote service snapshot has the wrong address")
        members: Dict[str, Dict[str, Any]] = {}
        for member in instance["members"]:
            name = member["name"]
            if not name or name in members:
                raise ValueError("Remote service has invalid member descriptions")
            members[name] = member
        for name in self._descriptions:
            if name not in members:
                raise RemoteServiceError("service_member_not_found",
                                         f"Unknown remote service member {self.service_id}.{name}")
        self._descriptions = {name: member["kind"] for name, member in members.items()}
        for name, member in members.items():
            if member["kind"] == "state":
                slot = self._slots.get(name) or self._slot(name)
                slot.hydrate(member["sequence"], member["ops"])
            elif name in self._slots:
                self._slots[name]._set_kind("method")

    def update_member(self, member: str, sequence: int, ops: List[Any]) -> None:
        if self._descriptions.get(member) != "state":
            raise ValueError(f"Remote service update targets non-state member {self.service_id}.{member}")
        self._slot(member).update(sequence, ops)

    def clear(self) -> None:
        for slot in self._slots.values():
            slot.clear()

    def _slot(self, member: str) -> MemberHandle:
        slot = self._slots.get(member)
        if slot is not None:
            return slot
        slot = MemberHandle(self.service_id, member, self)
        if member in self._descriptions:
            slot._set_kind(self._descriptions[member])
        self._slots[member] = slot
        return slot


class _KeyedObserver:
    """One keyed observation callback (hashable identity for the observer set)."""

    __slots__ = ("active", "handler")

    def __init__(self, handler: Callable[[Dict[str, Any], Optional["ServiceFacade"]], None]) -> None:
        self.active = True
        self.handler = handler


class RemoteServiceBinding:
    """Consumer-side chord contract binding over a chord transport."""

    def __init__(self, services: List[str], transport: Any,
                 on_error: ErrorReporter = _ignore_error) -> None:
        self._transport = transport
        self._allowlist = set(services)
        if len(self._allowlist) != len(services):
            raise TypeError("Remote service binding has duplicate service IDs")
        self._report_error = on_error
        self._modes: Dict[str, str] = {}
        self._singletons: Dict[str, Dict[str, Any]] = {}
        self._keyed: Dict[str, Dict[str, Any]] = {}
        self._disposed = False

    def use(self, service_id: str) -> ServiceFacade:
        """Return (and start subscribing to) the singleton facade for one allowlisted service."""
        self._assert_available(service_id, "singleton")
        binding = self._singletons.get(service_id)
        if binding is not None:
            return binding["facade"]
        binding: Dict[str, Any] = {"facade": None, "active": True, "subscription": None,
                                   "starting": None, "revision": 0}
        binding["facade"] = ServiceFacade(
            service_id, None, self._transport,
            lambda: binding["active"] and not self._disposed,
            self._assert_handle_access, self._report_error)
        self._singletons[service_id] = binding
        self._start_singleton(service_id, binding)
        return binding["facade"]

    def observe(self, service_id: str,
                handler: Callable[[Dict[str, Any], Optional[ServiceFacade]], None]) -> Callable[[], None]:
        """Track keyed instances: ``handler(address, facade)`` per spawn, ``handler(address, None)`` per close."""
        self._assert_available(service_id, "keyed")
        binding = self._keyed.get(service_id)
        if binding is None:
            binding = {"handlers": set(), "instances": {}, "subscription": None,
                       "starting": None, "closed": False}
            self._keyed[service_id] = binding
        observer = _KeyedObserver(handler)
        binding["handlers"].add(observer)
        if binding["starting"] is None and not binding["closed"]:
            binding["starting"] = asyncio.create_task(self._start_keyed(service_id, binding))
            _report_task_errors(binding["starting"], self._report_error)

        def stop() -> None:
            observer.active = False
            binding["handlers"].discard(observer)
            if not binding["handlers"] and self._keyed.get(service_id) is binding:
                binding["closed"] = True
                del self._keyed[service_id]
                _report_task_errors(asyncio.create_task(self._close_keyed(binding)), self._report_error)

        return stop

    async def ready(self) -> None:
        """Wait until every subscription started by ``use``/``observe`` has settled."""
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        seen: Any = None
        while True:
            starting = [binding["starting"] for binding in self._singletons.values()
                        if binding["starting"] is not None]
            starting += [binding["starting"] for binding in self._keyed.values()
                         if binding.get("starting") is not None]
            tasks = tuple(starting)
            if tasks == seen:
                return
            seen = tasks
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        closes: List[Any] = []
        for binding in self._singletons.values():
            binding["active"] = False
            binding["facade"].clear()
            if binding["subscription"] is not None:
                closes.append(_maybe_await(binding["subscription"].close()))
        for binding in self._keyed.values():
            binding["closed"] = True
            if binding["subscription"] is not None:
                closes.append(_maybe_await(binding["subscription"].close()))
        self._singletons.clear()
        self._keyed.clear()
        await asyncio.gather(*closes, return_exceptions=True)

    # ------------------------------------------------------------------ internal

    def _start_singleton(self, service_id: str, binding: Dict[str, Any]) -> None:
        binding["revision"] += 1
        revision = binding["revision"]
        task = asyncio.create_task(self._run_singleton(service_id, binding, revision))
        binding["starting"] = task
        _report_task_errors(task, self._report_error)

    async def _run_singleton(self, service_id: str, binding: Dict[str, Any], revision: int) -> None:
        subscription = await self._transport.subscribe(
            service_id, "singleton",
            lambda update: self._on_singleton_update(service_id, binding, revision, update))
        if not binding["active"] or self._disposed or binding["revision"] != revision:
            await subscription.close()
            return
        binding["subscription"] = subscription
        snapshot = subscription.snapshot
        if snapshot["mode"] != "singleton" or snapshot["serviceId"] != service_id \
                or len(snapshot["instances"]) != 1:
            raise ValueError(f"Remote service {service_id} returned an invalid singleton snapshot")
        binding["facade"].install(snapshot["instances"][0])
        subscription.start()

    def _on_singleton_update(self, service_id: str, binding: Dict[str, Any], revision: int,
                             update: Dict[str, Any]) -> None:
        if not binding["active"] or binding["revision"] != revision or self._disposed:
            return
        try:
            kind = update.get("type")
            if kind == "reset":
                snapshot = _validate_reset(update["snapshot"], service_id, "singleton")
                if snapshot["instances"]:
                    binding["facade"].install(snapshot["instances"][0])
                else:
                    binding["facade"].clear()
            elif kind == "unavailable":
                binding["facade"].clear()
            elif kind == "replaced":
                if update["snapshot"].get("instance") is not None:
                    raise ValueError("Singleton replacement has an instance address")
                binding["facade"].install(update["snapshot"])
            elif kind == "state" and update.get("instance") is None:
                binding["facade"].update_member(update["member"], update["sequence"], update["ops"])
        except Exception as error:  # noqa: BLE001 - update failures go to the reporter
            self._report_error(error)

    async def _start_keyed(self, service_id: str, binding: Dict[str, Any]) -> None:
        subscription = await self._transport.subscribe(
            service_id, "keyed", lambda update: self._on_keyed_update(service_id, binding, update))
        if binding["closed"] or self._disposed:
            await subscription.close()
            return
        binding["subscription"] = subscription
        snapshot = subscription.snapshot
        if snapshot["mode"] != "keyed" or snapshot["serviceId"] != service_id:
            raise ValueError(f"Remote service {service_id} returned the wrong keyed snapshot")
        for instance in snapshot["instances"]:
            self._spawn_keyed(service_id, binding, instance)
        subscription.start()

    def _spawn_keyed(self, service_id: str, binding: Dict[str, Any], instance: Dict[str, Any]) -> None:
        address = instance.get("instance")
        if address is None:
            raise ValueError("Keyed service instance snapshot has no address")
        entry: Dict[str, Any] = {"address": dict(address), "active": True, "facade": None}
        entry["facade"] = ServiceFacade(
            service_id, dict(address), self._transport,
            lambda: entry["active"] and not binding["closed"] and not self._disposed,
            self._assert_handle_access, self._report_error)
        entry["facade"].install(instance)
        binding["instances"][address["key"]] = entry
        for observer in list(binding["handlers"]):
            if observer.active:
                try:
                    observer.handler(dict(address), entry["facade"])
                except Exception as error:  # noqa: BLE001
                    self._report_error(error)

    def _remove_keyed(self, binding: Dict[str, Any], entry: Dict[str, Any]) -> None:
        entry["active"] = False
        entry["facade"].clear()
        key = entry["address"]["key"]
        if binding["instances"].get(key) is entry:
            del binding["instances"][key]
        for observer in list(binding["handlers"]):
            if observer.active:
                try:
                    observer.handler(dict(entry["address"]), None)
                except Exception as error:  # noqa: BLE001
                    self._report_error(error)

    def _on_keyed_update(self, service_id: str, binding: Dict[str, Any], update: Dict[str, Any]) -> None:
        if binding["closed"] or self._disposed:
            return
        try:
            kind = update.get("type")
            if kind == "reset":
                snapshot = _validate_reset(update["snapshot"], service_id, "keyed")
                snapshots = {instance["instance"]["key"]: instance for instance in snapshot["instances"]}
                for key, entry in list(binding["instances"].items()):
                    fresh = snapshots.get(key)
                    if fresh is None or fresh["instance"]["generation"] != entry["address"]["generation"]:
                        self._remove_keyed(binding, entry)
                for key, instance in snapshots.items():
                    existing = binding["instances"].get(key)
                    if existing is None:
                        self._spawn_keyed(service_id, binding, instance)
                    else:
                        existing["facade"].install(instance)
            elif kind in ("unavailable", "replaced"):
                raise ValueError("Keyed service received a singleton lifecycle update")
            elif kind == "spawned":
                self._spawn_keyed(service_id, binding, update["instance"])
            elif kind == "closed":
                entry = binding["instances"].get(update["instance"]["key"])
                if entry is not None and entry["address"]["generation"] == update["instance"]["generation"]:
                    self._remove_keyed(binding, entry)
            elif kind == "state":
                if update.get("instance") is None:
                    raise ValueError("Keyed state update has no instance address")
                entry = binding["instances"].get(update["instance"]["key"])
                if entry is not None and entry["address"]["generation"] == update["instance"]["generation"]:
                    entry["facade"].update_member(update["member"], update["sequence"], update["ops"])
        except Exception as error:  # noqa: BLE001
            self._report_error(error)

    async def _close_keyed(self, binding: Dict[str, Any]) -> None:
        binding["closed"] = True
        if binding["subscription"] is not None:
            await binding["subscription"].close()

    def _assert_available(self, service_id: str, mode: str) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        if service_id not in self._allowlist:
            raise RemoteServiceError("service_not_allowed", f"Remote service {service_id} is not allowlisted")
        existing = self._modes.get(service_id)
        if existing is not None and existing != mode:
            raise RemoteServiceError("service_mode_mismatch",
                                     f"Remote service {service_id} is already used as {existing}")
        self._modes[service_id] = mode

    def _assert_handle_access(self) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")


def _validate_reset(snapshot: Dict[str, Any], service_id: str, mode: str) -> Dict[str, Any]:
    if snapshot["serviceId"] != service_id or snapshot["mode"] != mode \
            or (mode == "singleton" and len(snapshot["instances"]) > 1):
        raise ValueError("Remote service reset has the wrong service or mode")
    keys = set()
    for instance in snapshot["instances"]:
        if (mode == "singleton") != (instance.get("instance") is None):
            raise ValueError("Remote service reset has an invalid instance address")
        if instance.get("instance") is not None:
            key = instance["instance"]["key"]
            if key in keys:
                raise ValueError("Remote service reset repeats an instance key")
            keys.add(key)
        for member in instance["members"]:
            if member.get("kind") == "state" and (len(member["ops"]) != 1 or member["ops"][0][0] != "r"):
                raise ValueError("Remote service reset must contain full root replacements")
    return snapshot
