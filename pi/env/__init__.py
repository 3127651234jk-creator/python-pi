"""Remote execution environments over SSH (port of ``@earendil-works/pi-env``).

An agent's tools (running commands, reading and writing files, watching files)
can run on another machine while the Durable worker, storage, and credentials
stay local. Three parts:

- ``daemon``: the remote side, speaking the frame protocol of
  ``packages/env/docs/protocol.md`` on its stdin/stdout (in TS a Rust binary;
  here a Python module started as ``python -m pi.env.daemon serve --token <hex>``).
- ``RemoteExecutionEnv``: a Durable ``ExecutionEnv`` implementation that talks
  to the daemon through a ``Connection``. Results match running
  ``LocalExecutionEnv`` on that machine directly; only error messages may differ.
- SSH bootstrap: ``connect_ssh`` detects the remote system, deploys the daemon
  (verified by its SHA-256), and connects; ``ssh_connection`` defers all of
  that to the first operation. Building the environment itself needs no
  network; failures (no network, host key not trusted) surface as errors of
  the operation, and the next operation tries again.
"""

from .connection import (
    Connection,
    ConnectionOptions,
    Reply,
    RemoteError,
    RemoteInfo,
    is_connection_lost,
)
from .remote_env import (
    RemoteBinaryReader,
    RemoteDirReader,
    RemoteExecutionEnv,
    RemotePaths,
    RemoteTextLineReader,
    to_file_error,
)
from .watch import RemoteWatchOptions, RemoteWatcher
from .ssh import (
    HostKeyChangedError,
    HostKeyUnknownError,
    RemotePlatform,
    SshConnectOptions,
    SshError,
    SshTarget,
    accept_host_key,
    connect_ssh,
    daemon_path,
    deploy_daemon,
    detect_platform,
    forget_host_key,
    launch_command,
    packaged_daemon,
    scan_host_key,
    ssh_arguments,
    ssh_connection,
)

__all__ = [
    "Connection",
    "ConnectionOptions",
    "HostKeyChangedError",
    "HostKeyUnknownError",
    "RemoteBinaryReader",
    "RemoteDirReader",
    "RemoteError",
    "RemoteExecutionEnv",
    "RemoteInfo",
    "RemotePaths",
    "RemotePlatform",
    "RemoteTextLineReader",
    "RemoteWatchOptions",
    "RemoteWatcher",
    "Reply",
    "SshConnectOptions",
    "SshError",
    "SshTarget",
    "accept_host_key",
    "connect_ssh",
    "daemon_path",
    "deploy_daemon",
    "detect_platform",
    "forget_host_key",
    "is_connection_lost",
    "launch_command",
    "packaged_daemon",
    "scan_host_key",
    "ssh_arguments",
    "ssh_connection",
    "to_file_error",
]
