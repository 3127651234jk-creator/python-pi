import asyncio
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import pytest

from pi.env.connection import Connection
from pi.env.ssh import RemotePlatform, packaged_daemon


def test_packaged_daemon_checks_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr("pi.env.ssh.package_root", lambda: str(tmp_path))
    directory = tmp_path / "env/bin/pi-env-linux-x64"
    directory.mkdir(parents=True)
    binary = directory / "pi-env"
    binary.write_bytes(b"daemon")
    manifest = {"platform": "linux", "arch": "x64", "protocol": 1,
                "sha256": hashlib.sha256(b"daemon").hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    remote = RemotePlatform("linux", "x64", "/home/test")
    assert packaged_daemon(remote) == str(binary)
    binary.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        packaged_daemon(remote)


def test_frozen_daemon_protocol():
    system = {"win32": "windows", "linux": "linux", "darwin": "darwin"}.get(sys.platform)
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    remote = RemotePlatform(system, arch, str(Path.home()))
    directory = Path(__file__).resolve().parents[1] / "pi/env/bin" / f"pi-env-{system}-{arch}"
    if not directory.exists():
        pytest.skip("Build the native daemon with scripts/build_daemon.py first")
    binary = packaged_daemon(remote)
    async def scenario():
        connection = Connection([binary])
        try:
            info = await asyncio.wait_for(connection.info(), timeout=30)
            assert info is not None
            reply = await connection.request("lstat", {"path": binary})
            assert reply.json
        finally:
            connection.close()
            await asyncio.wait_for(connection.wait_closed(), timeout=10)
    asyncio.run(scenario())


def test_built_wheel_contains_daemon():
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("Build a distribution wheel first")
    with zipfile.ZipFile(max(wheels, key=lambda path: path.stat().st_mtime)) as wheel:
        manifests = [name for name in wheel.namelist() if name.startswith("pi/env/bin/") and name.endswith("/manifest.json")]
        assert manifests
        for name in manifests:
            manifest = json.loads(wheel.read(name))
            binary = name.rsplit("/", 1)[0] + "/" + manifest["file"]
            assert hashlib.sha256(wheel.read(binary)).hexdigest() == manifest["sha256"]
