"""Chord layer unit tests: delta wire codec, wire vocabulary, provider endpoint,
and consumer binding. Vectors mirror the TS chord test suite; byte-level
cross-checks against Node live in ``test_chord_interop.py``."""

import asyncio

import pytest

from pi.delta import (
    PathError,
    UnsafePathError,
    WireDecoder,
    WireEncoder,
    apply_ops,
    assert_valid_op,
    assert_valid_wire_op,
)
from pi.chord import (
    RemoteServiceError,
    ServiceContext,
    StateDecoder,
    StateEncoder,
    create_catalogue_call,
    create_subscribe_call,
    create_unsubscribe_call,
    decode_control_call,
    parse_provider_update,
    parse_service_call,
    parse_subscription_snapshot,
    parse_wire_provider_update,
    parse_wire_subscription_snapshot,
)
from pi.chord.provider import RemoteServiceEndpoint, RemoteServiceProvider, ReplicatedState
from pi.chord.consumer import ClientServiceTransport, RemoteServiceBinding, Replica


# ---------------------------------------------------------------------------
# Delta wire codec
# ---------------------------------------------------------------------------


def test_wire_codec_interns_path_on_second_use():
    encoder = WireEncoder()
    assert encoder.encode([["a", ["a", "deep"], "1"]]) == [["a", ["a", "deep"], "1"]]
    assert encoder.encode([["a", ["a", "deep"], "2"]]) == [["#", 0, ["a", "deep"]], ["a", 0, "2"]]


def test_wire_codec_short_form_for_adjacent_path():
    encoder = WireEncoder()
    assert encoder.encode([["s", ["value"], 1], ["s", ["value"], 2]]) == [["s", ["value"], 1], ["s", 2]]


def test_wire_codec_base_resets_dictionary():
    encoder = WireEncoder()
    assert encoder.encode([["r", {"value": 3}]]) == [["r", {"value": 3}]]
    # Inline again after the base: a recovery point must be self-contained.
    assert encoder.encode([["s", ["value"], 4]]) == [["s", ["value"], 4]]


def test_wire_codec_matches_ts_vectors():
    encoder = WireEncoder()
    # The second update of a member goes over the wire interned (service-wire.test.ts).
    assert encoder.encode([["s", ["revision"], 1]]) == [["s", ["revision"], 1]]
    assert encoder.encode([["s", ["revision"], 2]]) == [["#", 0, ["revision"]], ["s", 0, 2]]


def test_wire_decoder_round_trip_and_batch_scoping():
    encoder = WireEncoder()
    decoder = WireDecoder()
    batch1 = encoder.encode([["s", ["a", "b"], "hello"], ["a", ["a", "b"], " world"]])
    batch2 = encoder.encode([["t", ["a", "b"], 6]])
    decoded = decoder.decode(batch1) + decoder.decode(batch2)
    assert decoded == [["s", ["a", "b"], "hello"], ["a", ["a", "b"], " world"], ["t", ["a", "b"], 6]]
    value = {"a": {"b": ""}}
    assert apply_ops(value, decoded) == {"a": {"b": "world"}}


def test_wire_decoder_keeps_ids_across_batches_and_resets_on_base():
    encoder = WireEncoder()
    decoder = WireDecoder()
    decoder.decode(encoder.encode([["s", ["v"], 1]]))
    decoder.decode(encoder.encode([["s", ["v"], 2]]))  # interned here
    decoder.decode(encoder.encode([["r", {"v": 0}]]))
    # Dictionary cleared by the base: the same path is inline again.
    assert decoder.decode(encoder.encode([["s", ["v"], 3]]))[0] == ["s", ["v"], 3]


def test_wire_decoder_rejects_unresolved_short_form_and_id():
    with pytest.raises(PathError):
        WireDecoder().decode([["a", "x"]])
    with pytest.raises(PathError):
        WireDecoder().decode([["s", 5, 1]])


def test_wire_decoder_rejects_unsafe_interned_path():
    with pytest.raises(UnsafePathError):
        WireDecoder().decode([["#", 0, ["__proto__"]], ["s", 0, True]])


def test_wire_ops_reject_unknown_verb():
    with pytest.raises(TypeError, match="unknown op verb"):
        assert_valid_wire_op([["?", 0]])
    with pytest.raises(TypeError, match="unknown op verb"):
        assert_valid_op([["?", 0]])


def test_wire_decoder_short_forms_for_every_verb():
    decoded = WireDecoder().decode([
        ["s", ["xs"], 1],
        ["s", 2],                       # short: same path
        ["a", "text"],                  # short append
        ["t", 3],                       # short tail-slice
        ["p", 1, 0, [4]],               # short splice
        ["m", [1, 0]],                  # short permutation
        ["d"],                          # short delete
    ])
    assert decoded == [
        ["s", ["xs"], 1], ["s", ["xs"], 2], ["a", ["xs"], "text"], ["t", ["xs"], 3],
        ["p", ["xs"], 1, 0, [4]], ["m", ["xs"], [1, 0]], ["d", ["xs"]],
    ]


def test_replica_sequences_and_gap_clears_value():
    replica = Replica()
    replica.hydrate(0, [["r", {"count": 1}]])
    assert replica.value == {"count": 1}
    replica.update(1, [["s", ["count"], 2]])
    assert replica.value == {"count": 2}
    with pytest.raises(ValueError, match="gap"):
        replica.update(3, [["s", ["count"], 4]])
    assert replica.value is None
    with pytest.raises(ValueError, match="hydration"):
        replica.update(4, [["s", ["count"], 5]])


def test_replica_rejects_non_base_hydration_and_rejects_non_json():
    replica = Replica()
    with pytest.raises(ValueError, match="base"):
        replica.hydrate(0, [["s", ["count"], 1]])
    replica.hydrate(0, [["r", {"count": 1}]])
    with pytest.raises(TypeError, match="strict JSON"):
        replica.update(1, [["s", ["count"], float("nan")]])


# ---------------------------------------------------------------------------
# Wire vocabulary
# ---------------------------------------------------------------------------


def test_control_calls_round_trip():
    assert decode_control_call(create_catalogue_call()) == {"type": "catalogue"}
    assert decode_control_call(create_subscribe_call("sub-1", "pi.example", "singleton")) == {
        "type": "subscribe", "subscriptionId": "sub-1", "serviceId": "pi.example", "mode": "singleton"}
    assert decode_control_call(create_unsubscribe_call("sub-1")) == {
        "type": "unsubscribe", "subscriptionId": "sub-1"}
    assert decode_control_call({"serviceId": "pi.example", "member": "echo", "args": []}) is None
    # A subscribe carrying an instance address is not a control call.
    assert decode_control_call({**create_subscribe_call("s", "x", "singleton"),
                                "instance": {"key": "k", "generation": 1}}) is None
    assert decode_control_call({"serviceId": "$chord.service", "member": "subscribe",
                                "args": ["s", "x", "both"]}) is None


def test_parse_service_call_strict_keys():
    call = parse_service_call({"serviceId": "pi.example", "member": "echo", "args": [1]})
    assert call["member"] == "echo"
    with pytest.raises(TypeError):
        parse_service_call({"serviceId": "pi.example", "member": "echo", "args": [], "extra": True})
    with pytest.raises(TypeError):
        parse_service_call({"serviceId": "pi.example", "member": "echo"})
    with pytest.raises(TypeError):
        parse_service_call({"serviceId": "pi.example", "member": "echo", "args": [],
                            "instance": {"key": "k", "generation": 0}})


def test_parse_snapshot_and_update_shapes():
    snapshot = {"serviceId": "pi.example", "mode": "singleton", "instances": [
        {"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"v": 1}]]}]}]}
    parse_subscription_snapshot(snapshot)
    parse_wire_subscription_snapshot(snapshot)
    with pytest.raises(TypeError):
        snapshot["instances"][0]["members"][0]["sequence"] = 0
        parse_provider_update({"type": "state", "member": "state", "sequence": 0, "ops": []})
    with pytest.raises(TypeError):
        parse_provider_update({"type": "state", "member": "state", "sequence": 1, "ops": [],
                               "extra": 1})
    with pytest.raises(TypeError):
        parse_provider_update({"type": "reset", "snapshot": {**snapshot, "instances": [
            {"members": [{"name": "state", "kind": "state", "sequence": 5, "ops": [["s", ["v"], 1]]}]}]}})
    with pytest.raises(TypeError, match="Invalid service provider update"):
        parse_provider_update({"type": "exploded"})
    parse_provider_update({"type": "state", "instance": {"key": "k", "generation": 2},
                           "member": "state", "sequence": 1, "ops": [["s", ["v"], 1]]})
    with pytest.raises(TypeError):
        parse_provider_update({"type": "closed", "instance": {"key": "k", "generation": 0}})


def test_state_codec_registry_lifecycle():
    snapshot = {"serviceId": "pi.example", "mode": "singleton", "instances": [
        {"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"v": 1}]]}]}]}
    encoder = StateEncoder()
    wire_snapshot = encoder.encode_snapshot(snapshot)
    assert wire_snapshot["instances"][0]["members"][0]["ops"] == [["r", {"v": 1}]]
    # A fresh codec after the snapshot: first use inline, adjacent repeat shortened.
    wire = encoder.encode_update({"type": "state", "member": "state", "sequence": 1,
                                  "ops": [["s", ["v"], 2], ["s", ["v"], 3]]})
    assert wire["ops"] == [["s", ["v"], 2], ["s", 3]]
    # Second update batch: the path is interned across batches.
    wire = encoder.encode_update({"type": "state", "member": "state", "sequence": 2,
                                  "ops": [["s", ["v"], 4]]})
    assert wire["ops"] == [["#", 0, ["v"]], ["s", 0, 4]]
    decoder = StateDecoder()
    decoder.decode_snapshot(wire_snapshot)
    decoded = decoder.decode_update(parse_wire_provider_update(wire))
    assert decoded["ops"] == [["s", ["v"], 4]]
    # Unknown members fail loudly.
    with pytest.raises(ValueError, match="Unknown service state"):
        StateDecoder().decode_update({"type": "state", "member": "nope", "sequence": 1, "ops": []})


# ---------------------------------------------------------------------------
# Provider endpoint
# ---------------------------------------------------------------------------


class ExampleService:
    def __init__(self) -> None:
        self.state = ReplicatedState({"count": 0})

    async def echo(self, *args, context=None):
        # Provider invoke appends the context as the trailing positional argument.
        return list(args[:-1])


def build_provider():
    provider = RemoteServiceProvider([{"id": "pi.example"}, {"id": "pi.dialogs", "mode": "keyed"}])
    example = ExampleService()
    provider.provide("pi.example", example)
    return provider, example


def test_provider_catalogue_and_subscribe_snapshot():
    async def check():
        provider, _example = build_provider()
        endpoint = RemoteServiceEndpoint(provider)
        catalogue = await endpoint.invoke_service(create_catalogue_call(), None, None)
        assert catalogue == [{"serviceId": "pi.example", "mode": "singleton"},
                             {"serviceId": "pi.dialogs", "mode": "keyed"}]
        published = []

        async def publish(subscription_id, update):
            published.append((subscription_id, update))

        snapshot = await endpoint.invoke_service(
            create_subscribe_call("sub", "pi.example", "singleton"), publish, None)
        assert snapshot == {"serviceId": "pi.example", "mode": "singleton", "instances": [
            {"members": [{"name": "echo", "kind": "method"},
                         {"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"count": 0}]]}]}]}
        assert await endpoint.invoke_service(
            {"serviceId": "pi.example", "member": "echo", "args": ["a", 2]}, publish, None) == ["a", 2]
        with pytest.raises(ValueError, match="already active"):
            await endpoint.invoke_service(
                create_subscribe_call("sub", "pi.example", "singleton"), publish, None)
        with pytest.raises(RuntimeError, match="was not found"):
            await endpoint.invoke_service(create_unsubscribe_call("other"), publish, None)
        await endpoint.invoke_service(create_unsubscribe_call("sub"), publish, None)
        endpoint.dispose()
        provider.dispose()
    asyncio.run(check())


def test_provider_publishes_state_updates_with_sequences():
    async def check():
        provider, example = build_provider()
        updates = []
        subscription = provider.subscribe("pi.example", "singleton", updates.append)
        subscription.activate()
        example.state.replace({"count": 1})
        example.state.publish({"count": 1, "extra": True}, [["s", ["extra"], True]])
        assert [update["type"] for update in updates] == ["state", "state"]
        assert updates[0]["sequence"] == 1 and updates[0]["ops"] == [["r", {"count": 1}]]
        assert updates[1]["sequence"] == 2 and updates[1]["ops"] == [["s", ["extra"], True]]
        # Snapshot-suppression: a subscriber holding sequence 2 does not get it again.
        late = []
        late_subscription = provider.subscribe("pi.example", "singleton", late.append)
        late_subscription.activate()
        example.state.replace({"count": 1, "extra": True})
        assert late == []  # deep-equal replace publishes nothing
        example.state.replace({"count": 2, "extra": True})
        assert late[0]["sequence"] == 3
        late_subscription.close()
        subscription.close()
        provider.dispose()
    asyncio.run(check())


def test_provider_buffer_overflow_collapses_to_reset():
    async def check():
        provider, example = build_provider()
        updates = []
        subscription = provider.subscribe("pi.example", "singleton", updates.append)
        for count in range(1, 102):  # 101 publishes; the initial value is already {"count": 0}
            example.state.replace({"count": count})
        subscription.activate()
        assert len(updates) == 1
        assert updates[0]["type"] == "reset"
        snapshot = updates[0]["snapshot"]
        member = snapshot["instances"][0]["members"][1]
        assert member["ops"] == [["r", {"count": 101}]]
        subscription.close()
        provider.dispose()
    asyncio.run(check())


def test_provider_error_codes_and_keyed_lifecycle():
    async def check():
        provider, _example = build_provider()
        with pytest.raises(RemoteServiceError) as not_found:
            await provider.invoke({"serviceId": "pi.nope", "member": "echo", "args": []}, ServiceContext())
        assert not_found.value.code == "service_not_found"
        with pytest.raises(RemoteServiceError) as member_missing:
            await provider.invoke({"serviceId": "pi.example", "member": "nope", "args": []}, ServiceContext())
        assert member_missing.value.code == "service_member_not_found"
        with pytest.raises(RemoteServiceError) as mismatch:
            await provider.invoke({"serviceId": "pi.example", "member": "state", "args": []}, ServiceContext())
        assert mismatch.value.code == "service_member_mismatch"
        with pytest.raises(RemoteServiceError) as wrong_mode:
            await provider.invoke({"serviceId": "pi.example", "member": "echo", "args": [],
                                   "instance": {"key": "k", "generation": 1}}, ServiceContext())
        assert wrong_mode.value.code == "service_mode_mismatch"

        updates = []
        subscription = provider.subscribe("pi.dialogs", "keyed", updates.append)
        subscription.activate()
        close = provider.spawn("pi.dialogs", "inv-1", {"request": ReplicatedState({"value": 0})})
        with pytest.raises(RemoteServiceError, match="already has a live instance"):
            provider.spawn("pi.dialogs", "inv-1", {"request": ReplicatedState({"value": 0})})
        spawned = updates[0]
        assert spawned["type"] == "spawned"
        assert spawned["instance"]["instance"] == {"key": "inv-1", "generation": 1}
        with pytest.raises(RemoteServiceError) as stale:
            await provider.invoke({"serviceId": "pi.dialogs", "instance": {"key": "inv-1", "generation": 2},
                                   "member": "nope", "args": []}, ServiceContext())
        assert stale.value.code == "service_stale_instance"
        close()
        assert updates[-1] == {"type": "closed", "instance": {"key": "inv-1", "generation": 1}}
        subscription.close()
        provider.dispose()
    asyncio.run(check())


# ---------------------------------------------------------------------------
# Consumer binding (loopback over a provider)
# ---------------------------------------------------------------------------


class LoopbackTransport:
    """In-process RemoteServiceTransport over a provider (mirrors TS loopback.ts)."""

    def __init__(self, provider: RemoteServiceProvider) -> None:
        self.provider = provider

    async def invoke(self, call):
        return await self.provider.invoke(call, ServiceContext())

    async def subscribe(self, service_id, mode, listener):
        subscription = self.provider.subscribe(service_id, mode, lambda update: listener(update))
        return type("Handle", (), {"snapshot": subscription.snapshot,
                                   "start": subscription.activate,
                                   "close": subscription.close})()


def test_binding_use_facade_and_state_updates():
    async def check():
        provider, example = build_provider()
        binding = RemoteServiceBinding(["pi.example"], LoopbackTransport(provider))
        facade = binding.use("pi.example")
        await binding.ready()
        assert await facade.echo("a", 1) == ["a", 1]
        assert facade.state.value == {"count": 0}
        observed = []
        unsubscribe = facade.state.subscribe(observed.append)
        example.state.replace({"count": 1})
        assert facade.state.value == {"count": 1}
        # Subscribe delivers the current value first, then each revision.
        assert observed == [{"count": 0}, {"count": 1}]
        unsubscribe()
        example.state.replace({"count": 2})
        assert observed == [{"count": 0}, {"count": 1}]  # unsubscribed: no further delivery
        with pytest.raises(RemoteServiceError, match="not allowlisted"):
            RemoteServiceBinding(["other"], LoopbackTransport(provider)).use("pi.example")
        await binding.close()
        with pytest.raises(RuntimeError, match="disposed"):
            _ = facade.state.value
        with pytest.raises(RuntimeError, match="disposed"):
            await facade.echo()
        provider.dispose()
    asyncio.run(check())


def test_binding_reports_sequence_gaps_and_recovers_on_reset():
    async def check():
        provider, example = build_provider()
        errors = []
        binding = RemoteServiceBinding(["pi.example"], LoopbackTransport(provider),
                                       on_error=errors.append)
        facade = binding.use("pi.example")
        await binding.ready()
        updates = []
        subscription = provider.subscribe("pi.example", "singleton", updates.append)
        subscription.activate()
        example.state.publish({"count": 2}, [["s", ["count"], 2]])
        example.state.publish({"count": 5}, [["s", ["count"], 5]])
        assert facade.state.value == {"count": 5}
        # A gapped update on the binding's own stream clears the replica and reports.
        binding_state = binding._singletons["pi.example"]
        binding._on_singleton_update("pi.example", binding_state, 1,
                                     {"type": "state", "member": "state", "sequence": 1,
                                      "ops": [["s", ["count"], 9]]})
        assert facade.state.value is None and len(errors) == 1
        assert str(errors[0]).startswith("Replicated state update sequence has a gap")
        # Recovery via a full reset: the synthetic update never touched the state,
        # so the snapshot still carries the last published value.
        reset = {"type": "reset", "snapshot": provider._snapshot(
            provider._registrations["pi.example"])}
        binding._on_singleton_update("pi.example", binding._singletons["pi.example"], 1, reset)
        assert facade.state.value == {"count": 5}
        subscription.close()
        await binding.close()
        provider.dispose()
    asyncio.run(check())


def test_binding_observe_keyed_lifecycle():
    async def check():
        provider = RemoteServiceProvider([{"id": "pi.dialogs", "mode": "keyed"}])
        events = []
        binding = RemoteServiceBinding(["pi.dialogs"], LoopbackTransport(provider))
        stop = binding.observe("pi.dialogs", lambda address, facade: events.append((address, facade)))
        await binding.ready()
        close = provider.spawn("pi.dialogs", "d-1", {"request": ReplicatedState({"value": 0})})
        await asyncio.sleep(0)
        assert len(events) == 1 and events[0][0] == {"key": "d-1", "generation": 1}
        close()
        await asyncio.sleep(0)
        assert events[-1][0] == {"key": "d-1", "generation": 1} and events[-1][1] is None
        stop()
        await binding.close()
        provider.dispose()
    asyncio.run(check())
