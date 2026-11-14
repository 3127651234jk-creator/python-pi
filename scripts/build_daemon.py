"""Freeze the daemon on its target OS, then write its SHA-256 manifest."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    system = {"win32": "windows", "linux": "linux", "darwin": "darwin"}.get(sys.platform)
    machine = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if system is None or arch is None:
        parser.error(f"Unsupported build host: {sys.platform}-{machine}")
    name = f"pi-env-{system}-{arch}"
    output = (args.output or root / "pi" / "env" / "bin").resolve()
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pi-env-freeze-") as temporary:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile",
                        "--console", "--name", "pi-env", "--paths", str(root),
                        "--distpath", str(destination), "--workpath", str(Path(temporary) / "work"),
                        "--specpath", temporary, str(root / "scripts" / "daemon_entry.py")], check=True, cwd=root)
    binary = destination / ("pi-env.exe" if system == "windows" else "pi-env")
    subprocess.run([str(binary), "--version"], check=True, timeout=60)
    manifest = {"platform": system, "arch": arch, "protocol": 1,
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "file": binary.name}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(binary)


if __name__ == "__main__":
    main()
