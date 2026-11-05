# Routed protocol (experimental)

`pi.protocol`, `pi.server`, and `pi.client` implement the Pi v8 routed envelope
transport separately from the existing coding-agent JSONL RPC.

Features:

- Strict JSON envelopes, UUID v4 server identities, CBOR, four-byte big-endian
  frame lengths, incremental decoding and frame/depth/container limits.
- Async TCP and Unix socket connections; handshake timeout and identity checks.
- Concurrent correlated requests, cooperative cancellation, opaque service
  updates and out-of-band attachment changes.
- Connection-scoped server endpoints, shared Session handles and separate
  presentation attachments. Full Session routes reject stale attachments.
- Disconnect waits for admitted calls before releasing endpoints. Reconnect is
  explicit; calls are never replayed automatically.

The application supplies an async host:

```python
from pi.server import Server
from pi.client import Client

# host.attach_client(presentation) -> server endpoint
# host.resolve_session(session_id) -> application metadata
# host.open_session(metadata) -> session handle
# handle.attach_client() -> session endpoint
# handle.close() -> release shared worker/storage ownership
# endpoint.invoke_service(call, publish, cancelled) -> strict JSON result
# endpoint.release() -> remove subscriptions and release presentation resources

server = Server(host)
address = await server.start_tcp()  # localhost, dynamically allocated port
client = await Client(server.server_id).connect_tcp(*address[:2])
try:
    result = await client.request(
        {"serverId": server.server_id},
        {"serviceId": "example", "member": "read", "args": []},
    )
finally:
    await client.close()
    await server.close()
```

Management endpoints call `presentation.attach_session(id)` or
`presentation.detach_session()`; these publish the new route before the
management response. Session requests use `client.attachment`. Endpoints use
`await publish(subscription_id, update)` to send opaque updates, received through
`client.updates`. Applications own subscription creation, snapshots, hydration,
unsubscribe and update codecs. A cancelled request sets the endpoint's
`asyncio.Event`; it does not forcibly interrupt admitted work.

Use `start_unix(path)` / `connect_unix(path)` where Python supports Unix sockets.
The caller owns socket path permissions and cleanup. TCP transport provides no
authentication or encryption; the default binds only to loopback.

Remaining differences from TS: no Chord contract bindings or Delta decoder,
Unix discovery, generic callback transport factory, session-removal management,
worker termination monitoring, durable coding-agent service host, or CLI server
entrypoint. Thus this is a routed transport foundation, not full application
equivalence with the TypeScript stack. Unix behavior has not been exercised on
this Windows host. Tests cover real loopback TCP and bidirectional TS CBOR
encoding; full TypeScript Client/Server interoperability is not yet verified.
