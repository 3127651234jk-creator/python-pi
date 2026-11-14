"""SSH bootstrap pieces that can be tested without a network: argument building, host-key
management, probe parsing, and launch commands. A full round trip runs only when
``PI_ENV_SSH_HOST`` is set (like ``packages/env``'s ``test/ssh-external.test.ts``).
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.env import (
    HostKeyChangedError,
    RemotePlatform,
    SshConnectOptions,
    SshTarget,
    accept_host_key,
    forget_host_key,
    launch_command,
    scan_host_key,
    ssh_arguments,
)
from pi.env.ssh import normalize_arch, parse_probe, termux_warnings


def target(tmp_path, alias="pi-env-test"):
    return SshTarget(
        host="test-host",
        known_hosts_file=str(tmp_path / "known_hosts"),
        host_key_alias=alias,
    )


RSA_LINE = "pi-env-test ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDTest0123456789"
ED_LINE = "pi-env-test ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample"


def test_ssh_arguments():
    arguments = ssh_arguments(
        SshTarget(host="h", known_hosts_file="C:/k h", host_key_alias="alias", user="u", port=2222)
    )
    assert arguments[0:2] == ["-T", "-a"]
    assert "-o" in arguments and "BatchMode=yes" in arguments
    assert "ClearAllForwardings=yes" in arguments
    assert "ControlMaster=no" in arguments
    assert "SendEnv=-*" in arguments
    assert "StrictHostKeyChecking=yes" in arguments
    # The known-hosts file is quoted; ``%`` is doubled for ssh's token expansion.
    assert 'UserKnownHostsFile="C:/k h"' in arguments
    assert "HostKeyAlias=alias" in arguments
    assert arguments[-2:] == ["--", "h"]
    assert "-l" in arguments and arguments[arguments.index("-l") + 1] == "u"
    assert "-p" in arguments and arguments[arguments.index("-p") + 1] == "2222"
    # accept-new mode for scanning
    relaxed = ssh_arguments(
        SshTarget(host="h", known_hosts_file="C:/k h", host_key_alias="alias"), strict_host_keys=False
    )
    assert "StrictHostKeyChecking=accept-new" in relaxed


def test_ssh_arguments_rejects_options():
    with pytest.raises(ValueError):
        ssh_arguments(SshTarget(host="-oProxyCommand=x", known_hosts_file="k", host_key_alias="a"))
    with pytest.raises(ValueError):
        ssh_arguments(SshTarget(host="h", known_hosts_file="k", host_key_alias="a", port=0))
    with pytest.raises(ValueError):
        ssh_arguments(SshTarget(host="h", known_hosts_file="k", host_key_alias="a", port=70000))
    with pytest.raises(ValueError):
        ssh_arguments(SshTarget(host="h", known_hosts_file="k", host_key_alias="-bad"))


def test_accept_and_forget_host_keys(tmp_path):
    async def main():
        target_ = target(tmp_path)
        await accept_host_key(target_, [RSA_LINE, ED_LINE])
        with open(target_.known_hosts_file, encoding="utf-8") as handle:
            assert handle.read().splitlines() == [RSA_LINE, ED_LINE]
        # The same keys again: no change.
        await accept_host_key(target_, [RSA_LINE])
        with open(target_.known_hosts_file, encoding="utf-8") as handle:
            assert handle.read().splitlines() == [RSA_LINE, ED_LINE]
        # A different key of a trusted type is refused until the old key is forgotten.
        changed = RSA_LINE.replace("Test0123456789", "Different0987654321")
        with pytest.raises(HostKeyChangedError):
            await accept_host_key(target_, [changed])
        await forget_host_key(target_)
        with open(target_.known_hosts_file, encoding="utf-8") as handle:
            assert handle.read() == ""
        await accept_host_key(target_, [changed])
        with open(target_.known_hosts_file, encoding="utf-8") as handle:
            assert handle.read().splitlines() == [changed]

    asyncio.run(main())


def test_host_key_lines_are_validated(tmp_path):
    async def main():
        target_ = target(tmp_path)
        for bad in [
            "@revoked " + RSA_LINE,  # markers are not plain lines
            "other-host ssh-rsa AAAAB3NzaC1yc2EAAA",  # another alias
            "pi-env-test not-a-type AAAAB3NzaC1yc2EAAA",  # not a key type
            "pi-env-test ssh-rsa not!base64",  # not a key
            "pi-env-test ssh-rsa",  # no key
        ]:
            with pytest.raises(Exception):
                await accept_host_key(target_, [bad])

    asyncio.run(main())


def test_probe_parsing():
    probe = parse_probe(
        SshTarget(host="h", known_hosts_file="k", host_key_alias="a"),
        "banner line\nPI-ENV-PROBE\nLinux\nx86_64\nLinux\n/home/u\n/tmp\ntermux-exec:...-x\n",
    )
    assert probe["system"] == "Linux"
    assert probe["home"] == "/home/u"
    assert probe["preload"].startswith("termux-exec")
    with pytest.raises(RuntimeError):
        parse_probe(SshTarget(host="h", known_hosts_file="k", host_key_alias="a"), "no marker here")
    assert normalize_arch("x86_64") == "x64"
    assert normalize_arch("AMD64") == "x64"
    assert normalize_arch("aarch64") == "arm64"
    with pytest.raises(RuntimeError):
        normalize_arch("mips")
    warnings = termux_warnings({"tmpdir": "-", "preload": "nothing"})
    assert len(warnings) == 3
    assert termux_warnings({"tmpdir": "/data/tmp", "preload": "termux-exec"})[-1].startswith("Android")


def test_posix_probe_prints_one_field_per_line():
    """The probe's ``printf`` format needs exactly one backslash before ``n``; two print it literally."""
    from pi.env.ssh import POSIX_PROBE

    assert 'printf "%s\\n"' in POSIX_PROBE
    assert 'printf "%s\\\\n"' not in POSIX_PROBE


def test_launch_command():
    remote = RemotePlatform(platform="linux", arch="x64", home="/home/u")
    assert launch_command(remote, "/home/u/.pi/mobile/tools/pi-env", False) == "'/home/u/.pi/mobile/tools/pi-env'"
    assert launch_command(remote, "/bin/pi-env", True) == "exec \"$SHELL\" -lc 'exec \"$0\" \"$@\"' '/bin/pi-env'"
    windows = RemotePlatform(platform="windows", arch="x64", home="C:\\Users\\u", shell="cmd")
    assert launch_command(windows, "C:\\tools\\pi-env.exe", False) == "C:\\tools\\pi-env.exe"
    assert launch_command(windows, "C:\\my tools\\pi-env.exe", False) == '"C:\\my tools\\pi-env.exe"'
    windows.shell = "powershell"
    assert launch_command(windows, "C:\\tools\\pi-env.exe", False) == "& 'C:\\tools\\pi-env.exe'"


def test_run_ssh_feeds_stdin_while_reading_stderr(tmp_path):
    """An upload must not be truncated by ``communicate()``, and a chatty process must not deadlock."""
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        "\n".join(
            [
                "import sys",
                "# More than a pipe buffer of diagnostics before reading stdin.",
                "sys.stderr.buffer.write(b'e' * (2 * 1024 * 1024))",
                "sys.stderr.buffer.flush()",
                "data = sys.stdin.buffer.read()",
                "sys.stdout.write(str(len(data)))",
            ]
        ),
        encoding="utf-8",
    )
    from pi.env.ssh import run_ssh

    async def main():
        target = SshTarget(
            host="h", known_hosts_file=str(tmp_path / "kh"), host_key_alias="a", ssh=sys.executable
        )
        output = await asyncio.wait_for(
            run_ssh(target, "ignored", stdin=b"x" * 300_000, args=[str(script)]), timeout=20
        )
        assert output == "300000"

    asyncio.run(main())


@pytest.mark.skipif("PI_ENV_SSH_HOST" not in os.environ, reason="set PI_ENV_SSH_HOST to run against a real SSH server")
def test_ssh_external(tmp_path):
    host = os.environ["PI_ENV_SSH_HOST"]

    async def main():
        from pi.durable.env import Context, get_or_throw
        from pi.env import RemoteExecutionEnv, connect_ssh

        options = SshConnectOptions(
            host=host,
            known_hosts_file=str(tmp_path / "known_hosts"),
            host_key_alias="pi-env-test",
        )
        connection, remote = await connect_ssh(options)
        try:
            info = await connection.info()
            assert info.protocol == 1
            environment = RemoteExecutionEnv(
                connection=connection, id="ssh-test", cwd=remote.home, shell_env={"PI_ENV_TEST": "1"}
            )
            get_or_throw(await environment.write_file("pi-env-test.txt", "hello from ssh", Context()))
            assert get_or_throw(await environment.read_text_file("pi-env-test.txt", Context())) == "hello from ssh"
            await environment.cleanup(Context())
        finally:
            connection.close()
            await connection.wait_closed()

    asyncio.run(main())
