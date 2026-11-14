"""SSH bootstrap for the pi-env daemon (port of ``packages/env/src/ssh.ts``).

``connect_ssh`` detects the remote system, verifies or deploys the daemon
(named and checked by its SHA-256), and returns a ``Connection`` that starts
it over ``ssh``; ``ssh_connection`` defers all of that to the first request.
Host key management works on an application-owned known-hosts file under a
fixed alias: ``scan_host_key`` shows fingerprints, ``accept_host_key`` trusts
them, ``forget_host_key`` removes them; a changed key is never accepted
automatically.

``ssh`` runs with ``BatchMode``: strict host key checks against the
application's own file, no forwarding of any kind, no shared connections, no
commands from the user's configuration, no locale forwarding.

Porting deviations from the TS implementation:
- every ``ssh`` invocation is an ``asyncio`` subprocess; timeouts cancel it;
- daemon launches use the same spellings, but the binary the Python package
  ``ships`` is a placeholder path (``pi/env/bin/<platform>-<arch>``): the
  Python port has no prebuilt daemon binaries, so deployments pass
  ``binary=`` pointing at a ``python -m pi.env.daemon`` launcher or upload a
  built daemon.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .connection import Connection, ConnectionOptions

# How long a detection, check or upload over ``ssh`` may take before it is abandoned.
SSH_TIMEOUT_MS = 60_000
UPLOAD_TIMEOUT_MS = 300_000


@dataclass
class RemotePlatform:
    """A remote system the package ships a daemon for."""

    platform: str  # "linux" | "android" | "darwin" | "windows"
    arch: str  # "x64" | "arm64"
    # The remote home directory, in the remote system's own spelling.
    home: str
    # Windows: the shell ``sshd`` runs commands with (``DefaultShell``).
    shell: Optional[str] = None  # "cmd" | "powershell"
    # Problems worth telling the owner about, such as Termux without ``termux-exec``.
    warnings: List[str] = dataclass_field(default_factory=list)


@dataclass
class SshTarget:
    """How to reach a machine with the system ``ssh``."""

    # Host name or ``~/.ssh/config`` alias.
    host: str
    known_hosts_file: str
    # Host keys this application trusts; ``accept_host_key`` adds to it.
    # The name keys are stored under, independent of aliases, ports and jump hosts, e.g. ``pi-env-<env name>``.
    host_key_alias: str
    user: Optional[str] = None
    port: Optional[int] = None
    identity_file: Optional[str] = None
    # The ``ssh`` program; default ``ssh``.
    ssh: Optional[str] = None
    # An ``ssh`` configuration file instead of ``~/.ssh/config`` (``-F``).
    config_file: Optional[str] = None


class HostKeyUnknownError(Exception):
    """The remote host's key is not in the known-hosts file; ``scan_host_key`` shows it so the owner can accept it."""


class HostKeyChangedError(Exception):
    """The remote host's key differs from the one in the known-hosts file; it is never accepted automatically."""


class SshError(Exception):
    """An ``ssh`` invocation that failed for another reason, with its exit code and diagnostics."""

    def __init__(self, message: str, exit_code: Optional[int], stderr: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


def package_root() -> str:
    """The installed package's root, where a ``bin/`` directory of daemons would live."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _check_field(name: str, value: str) -> None:
    """No leading ``-`` (it would be read as an ``ssh`` option) and no whitespace or control characters."""
    if value == "" or value.startswith("-") or re.search(r"[\s\x00-\x1f\x7f]", value):
        raise ValueError(f"Invalid {name}: {value!r}")


def _config_path(name: str, path: str) -> str:
    """A path for an ``ssh`` option that expands ``%`` tokens: quoted, with ``%`` literal."""
    if path == "" or re.search(r'["\x00-\x1f\x7f]', path):
        raise ValueError(f"Invalid {name}: {path!r}")
    return '"' + path.replace("%", "%%") + '"'


def ssh_arguments(
    target: SshTarget, strict_host_keys: bool = True, known_hosts_file: Optional[str] = None
) -> List[str]:
    """
    Arguments for ``ssh`` up to the host: no prompts, no forwarding of any kind, no shared
    connections, no commands from the configuration, no locale forwarding (the remote uses its
    own), and host keys checked strictly against the application's own file under a fixed alias.
    """
    _check_field("host", target.host)
    _check_field("host key alias", target.host_key_alias)
    if target.user is not None:
        _check_field("user", target.user)
    if target.port is not None and (not isinstance(target.port, int) or not (1 <= target.port <= 65535)):
        raise ValueError(f"Invalid port: {target.port}")
    if known_hosts_file is None:
        known_hosts_file = target.known_hosts_file

    def option(setting: str) -> List[str]:
        return ["-o", setting]

    arguments: List[str] = []
    if target.config_file is not None:
        arguments += ["-F", target.config_file]
    arguments += [
        "-T",
        "-a",
        "-x",
        *option("BatchMode=yes"),
        *option("ClearAllForwardings=yes"),
        *option("ForwardAgent=no"),
        *option("ForwardX11=no"),
        *option("ControlMaster=no"),
        *option("ControlPath=none"),
        *option("RemoteCommand=none"),
        *option("PermitLocalCommand=no"),
        *option("SendEnv=-*"),
        *option("ServerAliveInterval=15"),
        *option(f"StrictHostKeyChecking={'yes' if strict_host_keys else 'accept-new'}"),
        *option(f"UserKnownHostsFile={_config_path('known hosts file', known_hosts_file)}"),
        *option("GlobalKnownHostsFile=none"),
        *option("HashKnownHosts=no"),
        *option(f"HostKeyAlias={target.host_key_alias}"),
    ]
    if target.user is not None:
        arguments += ["-l", target.user]
    if target.port is not None:
        arguments += ["-p", str(target.port)]
    if target.identity_file is not None:
        arguments += [
            *option(f"IdentityFile={_config_path('identity file', target.identity_file)}"),
            *option("IdentitiesOnly=yes"),
        ]
    arguments += ["--", target.host]
    return arguments


async def run_ssh(
    target: SshTarget,
    command: str,
    stdin: Optional[bytes] = None,
    args: Optional[Sequence[str]] = None,
    timeout_ms: Optional[int] = None,
) -> str:
    """Run one remote command over ``ssh``, optionally feeding stdin; resolves with stdout, rejects on failure."""
    if args is None:
        args = ssh_arguments(target)
    timeout_s = (timeout_ms if timeout_ms is not None else SSH_TIMEOUT_MS) / 1000
    try:
        process = await asyncio.create_subprocess_exec(
            target.ssh or "ssh",
            *args,
            command,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise SshError(str(error), None, "") from None

    async def feed() -> None:
        if stdin is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin)
                await process.stdin.drain()
            except (OSError, RuntimeError):
                pass
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except (OSError, RuntimeError):
                        pass

    async def collect() -> Tuple[bytes, bytes]:
        # Read both pipes while the feed writes stdin: ``communicate()`` would close stdin itself
        # and truncate an upload, and undrained pipes could block ``ssh`` mid-upload.
        readers = asyncio.ensure_future(
            asyncio.gather(process.stdout.read(), process.stderr.read())
        )
        try:
            await process.wait()
            stdout, stderr = await readers
            return stdout, stderr
        finally:
            if not readers.done():
                readers.cancel()
                try:
                    await readers
                except BaseException:
                    pass

    async def terminate() -> None:
        if process.returncode is None:
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        try:
            await process.wait()
        except (OSError, RuntimeError):
            pass

    feeding = asyncio.ensure_future(feed()) if stdin is not None else None
    try:
        stdout, stderr = await asyncio.wait_for(collect(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await terminate()
        diagnostics = ""
        if process.stderr is not None:
            try:
                diagnostics = (await asyncio.wait_for(process.stderr.read(), timeout=1)).decode("utf-8", "replace")
            except (asyncio.TimeoutError, OSError, RuntimeError):
                diagnostics = ""
        raise SshError(
            f"ssh {target.host} did not finish within {int(timeout_s)} s: {diagnostics.strip()}", None, diagnostics
        ) from None
    except BaseException:
        await terminate()
        raise
    finally:
        if feeding is not None:
            feeding.cancel()
            try:
                await feeding
            except BaseException:
                pass

    output = stdout.decode("utf-8", "replace")
    diagnostics = stderr.decode("utf-8", "replace")
    code = process.returncode
    if code == 0:
        return output
    if "REMOTE HOST IDENTIFICATION HAS CHANGED" in diagnostics:
        raise HostKeyChangedError(f"The host key of {target.host} changed; remove the old key to continue")
    if re.search(r"Host key verification failed|No .* host key is known", diagnostics):
        raise HostKeyUnknownError(f"The host key of {target.host} is not trusted yet")
    raise SshError(
        f"ssh {target.host} failed with exit code {code}: {diagnostics.strip()}", code, diagnostics
    )


async def scan_host_key(target: SshTarget) -> Dict[str, List[str]]:
    """
    Connect once with ``accept-new`` against a temporary known-hosts file to capture the key the
    host presents, through the same route (``~/.ssh/config``, jump hosts) as real connections.
    Returns its known-hosts lines and fingerprints. Showing a fingerprint is not authentication:
    compare it with one obtained out of band before accepting it.
    """
    directory = tempfile.mkdtemp(prefix="pi-env-hostkey-")
    try:
        scanned = os.path.join(directory, "known_hosts")
        # Authentication may fail without the key; the key is recorded before authentication. A
        # changed host key is not swallowed: the owner must see it.
        try:
            await run_ssh(target, "exit 0", args=ssh_arguments(target, False, scanned))
        except (SshError, HostKeyUnknownError):
            pass
        lines: List[str] = []
        if os.path.exists(scanned):
            with open(scanned, "r", encoding="utf-8", errors="replace") as handle:
                lines = [line for line in handle.read().split("\n") if line.strip() != ""]
        if not lines:
            raise RuntimeError(f"No host key received from {target.host}")
        process = await asyncio.create_subprocess_exec(
            "ssh-keygen",
            "-lf",
            scanned,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await process.communicate()
        fingerprints = [line for line in stdout.decode("utf-8", "replace").split("\n") if line.strip() != ""]
        return {"lines": lines, "fingerprints": fingerprints}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _parse_host_key_line(alias: str, line: str) -> Tuple[str, str]:
    """A known-hosts line for ``alias``: ``alias type key [comment]``, no markers, patterns or hashed names."""
    parts = line.strip().split()
    if len(parts) < 3:
        raise RuntimeError(f"Not a host key line for {alias}: {line!r}")
    host, key_type, key = parts[0], parts[1], parts[2]
    if host != alias or not re.fullmatch(r"(ssh-[a-z0-9-]+|ecdsa-sha2-[a-z0-9-]+|sk-[a-z0-9@.-]+)", key_type) or not re.fullmatch(
        r"[A-Za-z0-9+/]+={0,2}", key
    ):
        raise RuntimeError(f"Not a host key line for {alias}: {line!r}")
    return key_type, key


# Known-hosts files being changed, so changes to one file happen one at a time.
_known_hosts_locks: Dict[str, asyncio.Lock] = {}


def _known_hosts_lock(file: str) -> asyncio.Lock:
    if file not in _known_hosts_locks:
        _known_hosts_locks[file] = asyncio.Lock()
    return _known_hosts_locks[file]


async def _change_known_hosts(file: str, change: Callable[[List[str]], List[str]]) -> None:
    """Rewrite the known-hosts file one change at a time, atomically."""
    async with _known_hosts_lock(file):
        os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
        existing: List[str] = []
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8", errors="replace") as handle:
                existing = [line for line in handle.read().split("\n") if line != ""]
        lines = change(existing)
        temporary = f"{file}.{secrets.token_hex(6)}.tmp"
        content = "" if not lines else "\n".join(lines) + "\n"
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(content)
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
        try:
            os.replace(temporary, file)
        except OSError:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise


async def accept_host_key(target: SshTarget, lines: Sequence[str]) -> None:
    """
    Trust host-key lines from ``scan_host_key`` by adding them to the target's known-hosts file.
    Only plain lines for the target's alias are accepted. A key that differs from a trusted key of
    the same type is refused with ``HostKeyChangedError``; ``forget_host_key`` must remove the old
    keys first.
    """
    alias = target.host_key_alias
    accepted = [((line.strip(),) + _parse_host_key_line(alias, line)) for line in lines]

    def change(existing: List[str]) -> List[str]:
        trusted: Dict[str, str] = {}
        for line in existing:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == alias:
                trusted[parts[1]] = parts[2]
        added: List[str] = []
        for line, key_type, key in accepted:
            known = trusted.get(key_type)
            if known == key:
                continue
            if known is not None:
                raise HostKeyChangedError(
                    f"The {key_type} host key of {target.host} changed; forget the old key first"
                )
            trusted[key_type] = key
            added.append(line)
        return [*existing, *added]

    await _change_known_hosts(target.known_hosts_file, change)


async def forget_host_key(target: SshTarget) -> None:
    """Stop trusting every key stored for the target's alias, e.g. after the owner confirmed a changed host key."""

    def change(existing: List[str]) -> List[str]:
        return [line for line in existing if line.strip().split()[0:1] != [target.host_key_alias]]

    await _change_known_hosts(target.known_hosts_file, change)


# POSIX detection, run by the remote login shell; a Windows host answers through PowerShell instead.
# ``printf "%s\n"``: the value carries one backslash and an ``n``, which the remote printf turns into
# a newline. (Two backslashes would print them literally.)
POSIX_PROBE = "sh -c 'echo PI-ENV-PROBE; uname -s; uname -m; uname -o 2>/dev/null || echo -; printf \"%s\\n\" \"$HOME\" \"${TMPDIR:--}\" \"${LD_PRELOAD:--}\"'"


def powershell(script: str) -> str:
    return (
        "powershell -NoProfile -NonInteractive -EncodedCommand "
        + base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    )


WINDOWS_PROBE = powershell(
    "'PI-ENV-PROBE'; 'Windows'; [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString(); '-'; $HOME; '-'; '-'"
)


def normalize_arch(machine: str) -> str:
    lower = machine.lower()
    if lower in ("x86_64", "amd64", "x64"):
        return "x64"
    if lower in ("aarch64", "arm64"):
        return "arm64"
    raise RuntimeError(f"Unsupported remote architecture: {machine}")


def parse_probe(target: SshTarget, output: str) -> Dict[str, str]:
    """Login shells may print a banner first."""
    lines = output.splitlines()
    if "PI-ENV-PROBE" not in lines:
        raise RuntimeError(f"Unexpected answer from {target.host}: {output.strip()}")
    start = lines.index("PI-ENV-PROBE")
    values = lines[start + 1 :]
    while len(values) < 6:
        values.append("-")
    system, machine, os_, home, tmpdir, preload = values[:6]
    return {"system": system, "machine": machine, "os": os_, "home": home, "tmpdir": tmpdir, "preload": preload}


def termux_warnings(probe: Dict[str, str]) -> List[str]:
    """Termux works only with its own ``TMPDIR`` and with ``termux-exec``, which makes ``#!/usr/bin/env`` shebangs work."""
    warnings: List[str] = []
    if probe["tmpdir"] == "-":
        warnings.append("TMPDIR is not set; Termux's sshd normally sets it to $PREFIX/tmp.")
    if "termux-exec" not in probe["preload"]:
        warnings.append("termux-exec is not loaded (LD_PRELOAD); scripts with #!/usr/bin/env shebangs will fail.")
    warnings.append("Android may suspend Termux; run termux-wake-lock on the device to keep the connection alive.")
    return warnings


async def detect_platform(target: SshTarget) -> RemotePlatform:
    """Which system the target runs: ``uname`` through the login shell, or PowerShell on Windows."""
    try:
        probe = parse_probe(target, await run_ssh(target, POSIX_PROBE))
    except (RuntimeError, SshError) as error:
        # cmd.exe or PowerShell as the remote shell: no ``sh``, or one that gets the probe's quotes wrong.
        if isinstance(error, (HostKeyUnknownError, HostKeyChangedError)):
            raise
        probe = parse_probe(target, await run_ssh(target, WINDOWS_PROBE))
    # Git Bash as Windows' default SSH shell: ask PowerShell for Windows' own architecture and home spelling.
    if re.match(r"^(MINGW|MSYS|CYGWIN)", probe["system"]):
        probe = parse_probe(target, await run_ssh(target, WINDOWS_PROBE))
    arch = normalize_arch(probe["machine"])
    system, os_, home = probe["system"], probe["os"], probe["home"]
    if system == "Windows":
        # cmd.exe expands %OS%; PowerShell prints it as is.
        shell = "cmd" if "Windows_NT" in (await run_ssh(target, "echo %OS%")) else "powershell"
        return RemotePlatform(platform="windows", arch=arch, home=home, shell=shell, warnings=[])
    if system == "Darwin":
        return RemotePlatform(platform="darwin", arch=arch, home=home, warnings=[])
    if system == "Linux":
        android = os_ == "Android"
        return RemotePlatform(
            platform="android" if android else "linux",
            arch=arch,
            home=home,
            warnings=termux_warnings(probe) if android else [],
        )
    raise RuntimeError(f"Unsupported remote system: {system}")


def packaged_daemon(remote: RemotePlatform) -> str:
    """The daemon binary this package ships for a remote system."""
    name = f"pi-env-{remote.platform}-{remote.arch}"
    binary = os.path.join(package_root(), "env", "bin", name, "pi-env.exe" if remote.platform == "windows" else "pi-env")
    manifest_file = os.path.join(os.path.dirname(binary), "manifest.json")
    if not os.path.isfile(binary):
        raise RuntimeError(f"No bundled daemon for {remote.platform}-{remote.arch}; install a daemon distribution or pass binary=")
    with open(manifest_file, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest.get("platform"), manifest.get("arch"), manifest.get("protocol")) != (remote.platform, remote.arch, 1):
        raise RuntimeError("Bundled daemon manifest does not match the target platform/protocol")
    with open(binary, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if manifest.get("sha256") != digest:
        raise RuntimeError("Bundled daemon checksum mismatch")
    return binary


def quote_posix(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def quote_power_shell(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def daemon_path(remote: RemotePlatform, sha256: str) -> str:
    """Where a daemon with this content lives on the remote machine: named by its SHA-256, so versions never collide."""
    name = f"pi-env-{sha256[:32]}"
    if remote.platform == "windows":
        return f"{remote.home}\\.pi\\mobile\\tools\\{name}.exe"
    return f"{remote.home}/.pi/mobile/tools/{name}"


POSIX_HASH = (
    "hash() { if command -v sha256sum >/dev/null 2>&1; then sha256sum \"$1\" | cut -d' ' -f1; "
    "elif command -v shasum >/dev/null 2>&1; then shasum -a 256 \"$1\" | cut -d' ' -f1; "
    "elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 \"$1\" | sed 's/.*= //'; "
    "else echo none; fi; }"
)


async def check_daemon(target: SshTarget, remote: RemotePlatform, file: str, sha256: str) -> str:
    """Whether the remote file has this content: ``present``, ``missing``, or (POSIX without a hash tool) ``nohash``."""
    if remote.platform == "windows":
        script = (
            f"$f = {quote_power_shell(file)}; if ((Test-Path -LiteralPath $f) -and "
            f"((Get-FileHash -Algorithm SHA256 -LiteralPath $f).Hash.ToLower() -eq '{sha256}')) "
            "{ 'present' } else { 'missing' }"
        )
        output = await run_ssh(target, powershell(script))
        return "present" if output.strip().endswith("present") else "missing"
    check = (
        f"{POSIX_HASH}; f={quote_posix(file)}; "
        f'if [ -f "$f" ] && [ "$(hash "$f")" = {sha256} ]; then echo present; '
        'elif [ "$(hash /dev/null)" = none ]; then echo nohash; else echo missing; fi'
    )
    output = await run_ssh(target, f"sh -c {quote_posix(check)}")
    parts = output.strip().split("\n")
    return parts[-1] if parts else "missing"


async def deploy_daemon(target: SshTarget, remote: RemotePlatform, binary: Optional[str] = None) -> str:
    """
    Make sure the daemon is on the remote machine, verified by its SHA-256 before it ever runs, and
    remove daemons of other contents. The upload goes to a new temporary file and is renamed into
    place only once its hash matches. Returns the remote path of the binary.
    """
    if binary is None:
        binary = packaged_daemon(remote)
    if not os.path.exists(binary):
        raise RuntimeError(f"No pi-env daemon for {remote.platform}-{remote.arch} at {binary}")
    with open(binary, "rb") as handle:
        content = handle.read()
    sha256 = hashlib.sha256(content).hexdigest()
    file = daemon_path(remote, sha256)
    state = await check_daemon(target, remote, file, sha256)
    if state == "present":
        return file
    if state == "nohash":
        raise RuntimeError(f"{target.host} has no sha256sum, shasum or openssl to verify pi-env")
    if remote.platform == "windows":
        script = "; ".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$f = {quote_power_shell(file)}",
                "$d = Split-Path -Parent $f",
                "New-Item -ItemType Directory -Force -Path $d | Out-Null",
                "$t = Join-Path $d ('.pi-env-' + [guid]::NewGuid().ToString() + '.tmp')",
                # PowerShell reads redirected stdin itself, as text lines for ``$input``, so the
                # binary comes as base64 lines up to an end marker: Windows' sshd may never pass on
                # the end of stdin.
                "$text = New-Object System.Text.StringBuilder",
                "foreach ($line in $input) { if ($line -eq 'PI-ENV-END') { break }; [void]$text.Append($line) }",
                "$bytes = [Convert]::FromBase64String($text.ToString())",
                "$out = [IO.File]::Open($t, 'CreateNew', 'Write', 'None'); $out.Write($bytes, 0, $bytes.Length); $out.Close()",
                f"if ((Get-FileHash -Algorithm SHA256 -LiteralPath $t).Hash.ToLower() -ne '{sha256}') {{ Remove-Item -LiteralPath $t; throw 'pi-env upload is corrupt' }}",
                # A running daemon or a virus scanner can hold the old file for a moment.
                "for ($i = 0; ; $i++) { try { Move-Item -Force -LiteralPath $t -Destination $f; break } catch { if ($i -ge 20) { throw }; Start-Sleep -Milliseconds 250 } }",
                # Older daemons; one that is running stays until it exits.
                "Get-ChildItem -LiteralPath $d -Filter 'pi-env-*.exe' | Where-Object { $_.FullName -ne $f } | ForEach-Object { Remove-Item -LiteralPath $_.FullName -ErrorAction SilentlyContinue }",
                "'deployed'",
            ]
        )
        upload = powershell(script)
        lines = base64.b64encode(content).decode("ascii")
        body = "\n".join(lines[index : index + 76] for index in range(0, len(lines), 76)) + "\nPI-ENV-END\n"
        await run_ssh(target, upload, stdin=body.encode("ascii"), timeout_ms=UPLOAD_TIMEOUT_MS)
        return file
    upload_script = "\n".join(
        [
            "set -e",
            POSIX_HASH,
            f"f={quote_posix(file)}",
            'd=$(dirname "$f")',
            'mkdir -p "$d"',
            'chmod 700 "$d"',
            't=$(mktemp "$d/.pi-env.XXXXXX")',
            'cat > "$t"',
            f'if [ "$(hash "$t")" != {sha256} ]; then rm -f "$t"; echo "pi-env upload is corrupt" >&2; exit 1; fi',
            'chmod 700 "$t"',
            'mv -f "$t" "$f"',
            # Older daemons; running ones keep their file open until they exit.
            'for old in "$d"/pi-env-*; do [ "$old" = "$f" ] || rm -f "$old"; done',
            "echo deployed",
        ]
    )
    await run_ssh(target, f"sh -c {quote_posix(upload_script)}", stdin=content, timeout_ms=UPLOAD_TIMEOUT_MS)
    return file


@dataclass
class SshConnectOptions(SshTarget):
    """Options for ``ssh_connection`` and ``connect_ssh``: the target, plus the binary to deploy."""

    binary: Optional[str] = None
    # POSIX: start the daemon through the user's login shell (``$SHELL -l``), so commands see the
    # environment of ``~/.profile`` and similar files. Off by default: ``ssh`` runs commands without
    # a login shell.
    login_shell: bool = False
    on_log: Optional[Callable[[str], None]] = None


def launch_command(remote: RemotePlatform, file: str, login_shell: bool) -> str:
    """The remote command that starts the daemon at ``file``, before the ``serve`` arguments."""
    if remote.platform == "windows":
        # PowerShell runs a quoted path only with the call operator; cmd.exe keeps one pair of quotes around a program.
        if remote.shell == "powershell":
            return f"& {quote_power_shell(file)}"
        return f'"{file}"' if " " in file else file
    # ``$0`` is the daemon, ``"$@"`` its arguments; ``$SHELL`` is the login shell's own name for itself.
    if login_shell:
        return "exec \"$SHELL\" -lc 'exec \"$0\" \"$@\"' " + quote_posix(file)
    return quote_posix(file)


class SshConnectionHandle:
    """What an SSH connection learned about its remote system, and the ``Connection`` that starts daemons over it."""

    def __init__(self, connection: Connection, state: Dict[str, Optional[RemotePlatform]]) -> None:
        self.connection = connection
        self._state = state

    def remote(self) -> Optional[RemotePlatform]:
        """The detected system, once known."""
        return self._state.get("remote")


def ssh_connection_for(options: SshConnectOptions, state: Dict[str, Optional[RemotePlatform]]) -> Connection:
    """A ``Connection`` whose every start detects the remote system (once), verifies or deploys the daemon, then starts it."""

    async def command() -> Sequence[str]:
        if state.get("remote") is None:
            remote = await detect_platform(options)
            for warning in remote.warnings:
                if options.on_log is not None:
                    options.on_log(warning + "\n")
            state["remote"] = remote
        remote = state["remote"]
        assert remote is not None
        # A deployment that just verified the binary counts for the first start.
        if state.get("verified") is not None:
            file = state["verified"]
            state["verified"] = None
        else:
            file = await deploy_daemon(options, remote, options.binary)
        return [
            options.ssh or "ssh",
            *ssh_arguments(options),
            launch_command(remote, file, options.login_shell),
        ]

    return Connection(ConnectionOptions(command=command, on_log=options.on_log))


def ssh_connection(options: SshConnectOptions) -> SshConnectionHandle:
    """
    A ``Connection`` to the target that does nothing until its first request. Each start detects the
    remote system (the first time), verifies the daemon and deploys it if it is missing or changed,
    then starts it over ``ssh``. A failure (no network, an untrusted or changed host key) fails the
    requests waiting for that start with code ``spawn_error`` and the ``ssh`` diagnostics as
    message; the next request tries again.
    """
    state: Dict[str, Optional[RemotePlatform]] = {}
    return SshConnectionHandle(ssh_connection_for(options, state), state)


async def connect_ssh(options: SshConnectOptions) -> Tuple[Connection, RemotePlatform]:
    """
    Detect the remote system and deploy the daemon now, then return a ``Connection`` that starts it
    over ``ssh``, like ``ssh_connection``. Failures reject here: host keys must already be trusted
    (``scan_host_key``, ``accept_host_key``), otherwise this raises ``HostKeyUnknownError``.
    """
    remote = await detect_platform(options)
    verified = await deploy_daemon(options, remote, options.binary)
    for warning in remote.warnings:
        if options.on_log is not None:
            options.on_log(warning + "\n")
    state: Dict[str, Optional[RemotePlatform]] = {"remote": remote, "verified": verified}
    return ssh_connection_for(options, state), remote
