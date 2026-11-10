"""Package management for extensions, skills, prompts and themes.

Installs from npm tarballs or git repositories into the agent directory's
default discovery locations. Lifecycle scripts are never executed: npm
packages are extracted as plain archives, git packages are cloned/updated with
git only. Installed files and metadata are recorded in ``packages.json`` so
``remove``/``update`` know exactly what was placed on disk.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import tarfile
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional

import httpx

from .config import Configuration, read_json, write_json

KINDS = ("extensions", "skills", "prompts", "themes")
REGISTRY_URL = "https://registry.npmjs.org"


@dataclass
class InstalledPackage:
    name: str
    source: str
    version: str
    files: Dict[str, List[str]] = field(default_factory=dict)  # kind -> installed paths
    installed_at: float = 0.0

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "version": self.version,
                "files": self.files, "installed_at": self.installed_at}

    @classmethod
    def from_json(cls, data: dict) -> "InstalledPackage":
        return cls(name=data.get("name", ""), source=data.get("source", ""), version=data.get("version", ""),
                   files=data.get("files", {}), installed_at=data.get("installed_at", 0.0))


def parse_source(source: str) -> dict:
    """Parse "npm:<pkg>[@version]", "git:<url>[#ref]" or a plain git URL."""
    if source.startswith("npm:"):
        spec = source[4:]
        match = re.fullmatch(r"(@[\w.-]+/[\w.-]+|[\w.-]+)(?:@([\w.^~*-]+))?", spec) if spec else None
        if not match:
            raise ValueError("Invalid npm package spec")
        return {"type": "npm", "name": match.group(1), "version": match.group(2)}
    url = source[4:] if source.startswith("git:") else source
    if "://" not in url or urlsplit_scheme(url) not in ("https", "http", "ssh", "git"):
        raise ValueError("Package source must be npm:<pkg> or a git URL")
    base, _, ref = url.partition("#")
    return {"type": "git", "url": base, "ref": ref or None}


def urlsplit_scheme(url: str) -> str:
    return urllib.parse.urlsplit(url).scheme.lower()


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def safe_extract_tar(data: bytes, target: Path) -> None:
    """Extract an npm tarball, refusing absolute paths and traversal."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        members = archive.getmembers()
        root = None
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (member.name.startswith("/") or ".." in parts
                    or "\\" in member.name or ":" in member.name):
                raise ValueError(f"Unsafe tarball path: {member.name}")
            if not (target / Path(*parts)).resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Unsafe tarball path: {member.name}")
            if root is None and parts:
                root = parts[0]
        if root is None:
            return
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.parts and relative.parts[0] == root:
                relative = PurePosixPath(*relative.parts[1:])
            if not relative.parts:
                continue
            destination = target / Path(*relative.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.issym() or member.islnk():
                raise ValueError(f"Refusing linked tarball entry: {member.name}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is not None:
                    destination.write_bytes(extracted.read())


def read_manifest(root: Path) -> Dict[str, List[str]]:
    """The "pi" manifest from package.json, falling back to convention dirs."""
    manifest: Dict[str, List[str]] = {}
    package_json = root / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
        declared = data.get("pi") if isinstance(data, dict) else None
        if isinstance(declared, dict):
            for kind in KINDS:
                value = declared.get(kind)
                if isinstance(value, str):
                    manifest[kind] = [value]
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    manifest[kind] = value
    if not manifest:
        for kind in KINDS:
            if (root / kind).is_dir():
                manifest[kind] = [kind]
    return manifest


class PackageManager:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.store_path = config.agent_dir / "packages.json"

    # -- store --------------------------------------------------------------

    def _read(self) -> Dict[str, dict]:
        return read_json(self.store_path)

    def _write(self, data: dict) -> None:
        write_json(self.store_path, data)

    def list(self) -> List[InstalledPackage]:
        return [InstalledPackage.from_json(value) for value in self._read().values()]

    def get(self, name: str) -> Optional[InstalledPackage]:
        record = self._read().get(name)
        return InstalledPackage.from_json(record) if record else None

    # -- install / update / remove -------------------------------------------

    async def install(self, source: str, notify: Callable[[str], None] = print) -> InstalledPackage:
        parsed = parse_source(source)
        if parsed["type"] == "npm":
            name, version, root = await self._install_npm(parsed, notify)
        else:
            name, version, root = await self._install_git(parsed, notify)
        existing = self._read().get(name)
        if existing:
            self._remove_files(InstalledPackage.from_json(existing))
        package = InstalledPackage(name=name, source=source, version=version, installed_at=time.time())
        package.files = self._place_files(name, root, notify)
        if not any(package.files.values()):
            raise ValueError(f"Package has no pi extensions/skills/prompts/themes: {source}")
        store = self._read()
        store[name] = package.to_json()
        self._write(store)
        self._persist_setting(source)
        return package

    async def update(self, name: str, notify: Callable[[str], None] = print) -> InstalledPackage:
        package = self.get(name)
        if package is None:
            raise ValueError(f"Package not installed: {name}")
        return await self.install(package.source, notify)

    def remove(self, name: str) -> None:
        package = self.get(name)
        if package is None:
            raise ValueError(f"Package not installed: {name}")
        self._remove_files(package)
        store = self._read()
        store.pop(name, None)
        self._write(store)

    # -- sources --------------------------------------------------------------

    async def _install_npm(self, parsed: dict, notify) -> tuple:
        name, version = parsed["name"], parsed["version"]
        async with httpx.AsyncClient(timeout=60, headers={"user-agent": "pi-python"}) as client:
            response = await client.get(f"{REGISTRY_URL}/{name}")
            if response.status_code >= 400:
                raise ValueError(f"npm package not found: {name} (HTTP {response.status_code})")
            metadata = response.json()
            versions = metadata.get("versions", {})
            if not isinstance(versions, dict) or not versions:
                raise ValueError(f"npm package has no versions: {name}")
            if version is None:
                version = (metadata.get("dist-tags") or {}).get("latest") or sorted(versions)[-1]
            if version not in versions:
                raise ValueError(f"npm version not found: {name}@{version}")
            tarball = versions[version].get("dist", {}).get("tarball")
            if not isinstance(tarball, str) or not tarball:
                raise ValueError(f"npm version has no tarball: {name}@{version}")
            notify(f"downloading {name}@{version}")
            archive = await client.get(tarball)
            if archive.status_code >= 400:
                raise ValueError(f"npm tarball download failed (HTTP {archive.status_code})")
        target = self.config.agent_dir / "npm" / re.sub(r"[^\w.-]", "_", name)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        safe_extract_tar(archive.content, target)
        return name, version, target

    async def _install_git(self, parsed: dict, notify) -> tuple:
        url, ref = parsed["url"], parsed["ref"]
        parsed_url = urllib.parse.urlsplit(url)
        relative = (parsed_url.path or "").removesuffix(".git").strip("/")
        hostname = parsed_url.hostname
        if (not relative or not hostname or "\\" in relative or ":" in relative
                or ".." in PurePosixPath(relative).parts
                or "\\" in hostname or ":" in hostname or hostname in (".", "..")):
            raise ValueError(f"Cannot derive a git target directory from: {url}")
        cache_root = (self.config.agent_dir / "git").resolve()
        target = cache_root / hostname / relative
        if not target.resolve().is_relative_to(cache_root):
            raise ValueError(f"Cannot derive a git target directory from: {url}")
        if target.exists() and not (target / ".git").exists():
            shutil.rmtree(target)
        if not target.exists():
            target.mkdir(parents=True)
            command = ["git", "clone", "--depth", "1", url, str(target)]
            if ref:
                command = ["git", "clone", "--depth", "1", "--branch", ref, url, str(target)]
            await self._run_git(command, notify)
        else:
            await self._run_git(["git", "-C", str(target), "fetch", "--depth", "1", "origin", ref or "HEAD"], notify)
            await self._run_git(["git", "-C", str(target), "reset", "--hard", "FETCH_HEAD"], notify)
        version = (await self._run_git(["git", "-C", str(target), "rev-parse", "HEAD"], notify)).strip()
        return relative.rsplit("/", 1)[-1], version, target

    async def _run_git(self, command: List[str], notify) -> str:
        process = await asyncio.create_subprocess_exec(*command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise ValueError("git failed: " + stderr.decode("utf-8", "replace").strip()[:500])
        return stdout.decode("utf-8", "replace")

    # -- placement -------------------------------------------------------------

    def _place_files(self, name: str, root: Path, notify) -> Dict[str, List[str]]:
        manifest = read_manifest(root)
        placed: Dict[str, List[str]] = {}
        for kind in KINDS:
            for relative in manifest.get(kind, []):
                origin = (root / relative).resolve()
                if not origin.exists() or not origin.is_relative_to(root.resolve()):
                    notify(f"skipping missing manifest path: {relative}")
                    continue
                if kind == "prompts":
                    # Prompt discovery scans only flat template files; packages
                    # install theirs flattened under "<package>--" prefixed names.
                    files = ([origin] if origin.is_file()
                             else sorted(origin.glob("*.md")))
                    if not files:
                        continue
                    destination = self.config.agent_dir / "prompts"
                    destination.mkdir(parents=True, exist_ok=True)
                    for source_file in files:
                        target_file = destination / f"{_safe_name(name)}--{source_file.name}"
                        shutil.copy2(source_file, target_file)
                        placed.setdefault("prompts", []).append(str(target_file))
                    continue
                destination = self.config.agent_dir / kind / name
                if kind not in placed:
                    if destination.exists():
                        shutil.rmtree(destination)
                    destination.mkdir(parents=True)
                if origin.is_dir():
                    shutil.copytree(origin, destination / origin.name)
                    placed.setdefault(kind, []).append(str(destination / origin.name))
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin, destination / origin.name)
                    placed.setdefault(kind, []).append(str(destination / origin.name))
        return placed

    def _remove_files(self, package: InstalledPackage) -> None:
        safe = _safe_name(package.name)
        for kind, paths in package.files.items():
            if kind not in KINDS:
                continue
            if kind == "prompts":
                base = self.config.agent_dir / "prompts"
                for path in paths:
                    target = Path(path)
                    try:
                        if target.parent == base and target.name.startswith(safe + "--") and target.exists():
                            target.unlink()
                    except (OSError, ValueError):
                        continue
                continue
            for path in paths:
                target = Path(path)
                root = self.config.agent_dir / kind / package.name
                try:
                    if target.is_relative_to(root) and target.exists():
                        shutil.rmtree(target.parent if target.is_file() else target, ignore_errors=True)
                except (OSError, ValueError):
                    continue
            base = self.config.agent_dir / kind / package.name
            if base.exists() and not any(base.iterdir()):
                base.rmdir()

    def _persist_setting(self, source: str) -> None:
        packages = self.config.settings.get("packages", [])
        if isinstance(packages, list) and source not in packages:
            self.config.save("packages", packages + [source])
