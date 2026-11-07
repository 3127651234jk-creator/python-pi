"""Chord service layer (compact port of ``chord/src/services`` over ``pi.delta``).

Contracts bind over the routed transport: providers answer ``$chord.service``
control calls and publish replicated-state updates; consumers subscribe through
:class:`ClientServiceTransport` and hold facades with live state replicas. The
routed server (:mod:`pi.server`) wraps subscription traffic with per-connection
state encoders from :mod:`pi.chord.codec`, matching the TS ``packages/server``.
"""

from .codec import StateDecoder, StateEncoder
from .consumer import (
    ClientServiceTransport,
    MemberHandle,
    RemoteServiceBinding,
    Replica,
    ServiceFacade,
    ServiceSubscriptionHandle,
)
from .provider import (
    RemoteServiceEndpoint,
    RemoteServiceProvider,
    ReplicatedState,
    ServiceContext,
    ServiceSubscription,
)
from .wire import (
    REMOTE_SERVICE_ERROR_CODES,
    SERVICE_CONTROL_ID,
    RemoteServiceError,
    create_catalogue_call,
    create_subscribe_call,
    create_unsubscribe_call,
    decode_control_call,
    parse_provider_update,
    parse_service_call,
    parse_service_catalogue,
    parse_subscription_snapshot,
    parse_wire_provider_update,
    parse_wire_subscription_snapshot,
)

__all__ = [
    "REMOTE_SERVICE_ERROR_CODES",
    "SERVICE_CONTROL_ID",
    "ClientServiceTransport",
    "MemberHandle",
    "RemoteServiceBinding",
    "RemoteServiceEndpoint",
    "RemoteServiceError",
    "RemoteServiceProvider",
    "Replica",
    "ReplicatedState",
    "ServiceContext",
    "ServiceFacade",
    "ServiceSubscription",
    "ServiceSubscriptionHandle",
    "StateDecoder",
    "StateEncoder",
    "create_catalogue_call",
    "create_subscribe_call",
    "create_unsubscribe_call",
    "decode_control_call",
    "parse_provider_update",
    "parse_service_call",
    "parse_service_catalogue",
    "parse_subscription_snapshot",
    "parse_wire_provider_update",
    "parse_wire_subscription_snapshot",
]
