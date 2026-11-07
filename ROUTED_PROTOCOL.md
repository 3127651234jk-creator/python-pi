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

## Durable coding-agent service host (`pi.durable.server`)

`pi.durable.server` binds the durable runtime to the routed transport, closing
the TS `experimental` server stack in compact form:

- `python -m pi server` (or installed `pi server`) runs a foreground server:
  session catalog, worker manager, and routed `Server` on TCP loopback by
  default (`--host`/`--port`), or a Unix socket with `--unix`. It prints
  `Server: <id>` and `Address: ...`, runs until interrupted, then closes.
- Sessions are directories under `--session-dir` holding `meta.json` and
  worker-owned `session.sqlite`. `pi.session-management` exposes `create`,
  `attach`, `detach`, and `remove` (release attachments, close the worker,
  delete the directory); `pi.session-directory` exposes a `state` snapshot.
- Each session is served by one worker subprocess (`python -m pi
  durable-worker`, `pi/durable/worker.py`) that owns the storage exclusively
  and serves `pi.agent-controller` over its stdio (`Client.connect_streams` +
  `Server.start_streams`). Its members: `prompt`, `abort`, `transcript`,
  `tasks`, `compact`, `shutdown`. A commit-driven loop drives the scheduler.
- The manager monitors `process.wait()`: an unexpected worker exit resolves
  the handle's `terminated` future with an error, and the server invalidates
  the session — every attachment receives `attachment: null`, and the next
  attach opens a fresh worker that replays from durable state.
- Worker model flags (`--provider`, `--model`, `--api-key`, `--base-url`,
  `--concurrency`) are forwarded from the server command line.

Deviations from the TS stack: no coordinator (workers connect over stdio), no
server profile/lockfile, no plugin packages, no relay/auth, no idle worker
retirement (workers live until `shutdown` or the parent's death), and TCP
loopback is the default transport.

## Chord service layer (`pi.chord`)

`pi.chord` ports the TS `@earendil-works/chord` service vocabulary onto the
routed transport:

- `pi.delta` is the chord delta package: `apply_ops` replay (moved from
  `pi.durable.delta`, which re-exports it for storage) plus the wire codec.
  `WireEncoder` interns a path on its second use (`["#", id, path]`), short
  forms reuse the previous op's path within a batch, the id dictionary spans
  batches and an `r` clears it; `WireDecoder` expands that wire form;
  `RevisionValidator` revalidates each applied revision as strict JSON.
- `pi.chord.wire` validates service calls, the reserved control service
  `$chord.service` (`catalogue`/`subscribe`/`unsubscribe`), subscription
  snapshots and provider updates, with the eight TS `RemoteServiceError` codes.
- `pi.chord.provider` hosts services for one remote consumer: methods plus
  `ReplicatedState` members, the snapshot/activate/close subscription
  lifecycle, snapshot-sequence suppression, and a 100-update buffer that
  collapses into a full `reset`. Deviation: no tracker/diff layer — states
  publish explicit ops or root replacements.
- `pi.chord.consumer` binds contracts client-side: `RemoteServiceBinding.use`
  facades (methods callable, states as sequence-checked replicas; a gap or
  duplicate clears the replica and reports) and `observe` for keyed instances;
  `ClientServiceTransport` routes the client's shared update queue and decodes
  per subscription through `pi.chord.codec`.
- `pi.server` wraps chord subscriptions with per-connection state encoders:
  the subscribe response precedes buffered updates, updates are encoded in
  publication order, duplicate subscription ids fail with `invalid_request`,
  and error codes pass through (`service_not_found`, ...; unknown errors map
  to `internal_error`).

The durable host serves `pi.session-management` (`create`/`attach`/`detach`/
`remove`) and `pi.session-directory` (replicated state
`{revision, sessions}`) as chord contracts with TS `SessionSummary` shapes;
session creates and removals publish directory revisions to subscribers.

TypeScript interoperability runs against the real TS stack (loaded from the
packages source tree without a build via `tests/interop/register.mjs`):
`tests/test_chord_interop.py` drives the Python client against a real TS
`Server` over TCP and a real TS `Client` against the Python durable server,
with byte-level chord delta codec cross-checks in both directions.

Remaining differences from TS: no Unix discovery, no full generic callback
transport factory, and the session-scoped worker services
(`pi.agent-controller` and friends) remain plain calls without chord states.
Service shapes beyond the two server-scoped durable contracts follow the
Python port's own model. Unix behavior has not been exercised on this Windows
host. Tests cover real loopback TCP, chord subscriptions in both directions
against the TypeScript stack, removal and worker-termination semantics, and
worker subprocesses with the faux provider.
