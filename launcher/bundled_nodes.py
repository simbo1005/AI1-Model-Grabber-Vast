"""Install the two original KRAST nodes from a tiny, pinned public image layer.

The node code is retrieved at runtime, not republished in this repository. No
container engine or multi-gigabyte image pull is involved.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any
from uuid import uuid4

import httpx

REPOSITORY = "velmcomfy/studio-legacy-4dd5158486"
IMAGE = REPOSITORY + "@sha256:e33fdbabe0430f217b680d6b947f41af1eb915c5795b89bf533e0a70f99f24ef"
LAYER = "sha256:9c480306a3532d1bf4d10c72c4ffb976e303b3c35cc0e681b9d0cd705c5ee11c"
MAX_LAYER_BYTES = 1024 * 1024
FILES = {
    "ComfyUI-SamplerV3": {
        "__init__.py": "6761759c2a9ea45e241f26f2e4d7871c72e5c19d17c9f51619fd02f83a599910",
        "nodes.py": "2b0964520a8e8d390df25b76fb09c01baa86355dfd2f82da71830cc6b3cc75c7",
    },
    "mrteez_prompt_source": {
        "__init__.py": "126d6141ffe40e7251f4a3e1a01e8085e825b639c0c36fa1ef57e5b9c6619fc2",
    },
}
_layer_cache: bytes | None = None


def validate_bundle_spec(node: dict[str, Any]) -> str:
    name = str(node.get("name", ""))
    if (name not in FILES or node.get("source") != "image_bundle"
            or node.get("source_image") != IMAGE or node.get("source_layer") != LAYER):
        raise RuntimeError("Unsupported or unpinned bundled custom node.")
    return name


async def fetch_layer() -> bytes:
    global _layer_cache
    if _layer_cache is not None:
        return _layer_cache
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            response = await client.get("https://auth.docker.io/token", params={
                "service": "registry.docker.io", "scope": f"repository:{REPOSITORY}:pull",
            })
            response.raise_for_status()
            token = response.json()["token"]
            buffer = bytearray()
            async with client.stream("GET", f"https://registry-1.docker.io/v2/{REPOSITORY}/blobs/{LAYER}",
                                     headers={"Authorization": f"Bearer {token}"}) as stream:
                stream.raise_for_status()
                async for chunk in stream.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > MAX_LAYER_BYTES:
                        raise RuntimeError("Bundled node layer exceeds the small-file download limit.")
        archive = bytes(buffer)
    except (httpx.HTTPError, KeyError, ValueError):
        raise RuntimeError("Could not fetch the pinned KRAST custom-node bundle. Retry setup.") from None
    if "sha256:" + hashlib.sha256(archive).hexdigest() != LAYER:
        raise RuntimeError("Bundled custom-node layer checksum mismatch.")
    _layer_cache = archive
    return archive


def extract_files(archive: bytes, name: str) -> dict[str, bytes]:
    expected = FILES[name]
    prefix = PurePosixPath("opt/app/custom_nodes") / name
    result = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe path in bundled custom-node archive.")
            if not path.is_relative_to(prefix):
                continue
            relative = str(path.relative_to(prefix))
            if relative == "." and member.isdir():
                continue
            if not member.isfile() or relative not in expected or relative in result:
                raise RuntimeError("Unexpected file or link in bundled custom-node archive.")
            if member.size > MAX_LAYER_BYTES:
                raise RuntimeError("Bundled node source exceeds the small-file limit.")
            content = bundle.extractfile(member).read(MAX_LAYER_BYTES + 1)
            if hashlib.sha256(content).hexdigest() != expected[relative]:
                raise RuntimeError("Bundled custom-node source checksum mismatch.")
            result[relative] = content
    if result.keys() != expected.keys():
        raise RuntimeError("Bundled custom-node archive is incomplete.")
    return result


async def prepare_bundled_node(node: dict[str, Any], root: Path) -> Path:
    name = validate_bundle_spec(node)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.is_symlink() or not destination.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Unsafe bundled custom-node destination.")
    if destination.exists():
        matches = destination.is_dir() and all(
            not (destination / path).is_symlink() and (destination / path).is_file()
            and hashlib.sha256((destination / path).read_bytes()).hexdigest() == digest
            for path, digest in FILES[name].items()
        )
        if matches:
            return destination
        raise RuntimeError(f"Existing {name} differs from the pinned original; preserve or move it before setup.")
    content = extract_files(await fetch_layer(), name)
    staging = root / (".dsnn-" + uuid4().hex + ".part")
    staging.mkdir()
    try:
        for path, value in content.items():
            (staging / path).write_bytes(value)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
