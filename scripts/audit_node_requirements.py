#!/usr/bin/env python3
"""Fail image publishing when a pinned node adds an unreviewed source dependency."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "catalog" / "workflows.json").read_text(encoding="utf-8"))
# These are intentionally not part of the wheel cache. dlib is built separately;
# the others need a deliberate compatibility review before being pre-installed.
ALLOWED_SOURCE_MARKERS = tuple(
    line.partition("#")[0].strip().lower()
    for line in (ROOT / "docker" / "known-source-builds.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)


def raw_requirements(repo: str, ref: str) -> str:
    match = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+)\.git", repo)
    if not match:
        raise RuntimeError(f"Unsupported custom-node repository: {repo}")
    owner, name = match.groups()
    url = f"https://raw.githubusercontent.com/{owner}/{name}/{ref}/requirements.txt"
    try:
        with urlopen(url, timeout=20) as response:  # nosec B310: validated GitHub URL
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        if exc.code == 404:
            return ""
        raise RuntimeError(f"Could not read {name} requirements (HTTP {exc.code})") from exc


def main() -> int:
    nodes = {
        (node["name"], node["repo"], node["ref"])
        for workflow in CATALOG["workflows"]
        for node in workflow.get("custom_nodes", [])
    }
    failures: list[str] = []
    for name, repo, ref in sorted(nodes):
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            failures.append(f"{name}: ref must be a pinned commit SHA")
            continue
        for line in raw_requirements(repo, ref).splitlines():
            candidate = line.strip().lower()
            if not candidate or candidate.startswith("#"):
                continue
            is_source = "git+" in candidate or candidate.startswith("dlib")
            if is_source and not any(marker in candidate for marker in ALLOWED_SOURCE_MARKERS):
                failures.append(f"{name}: unreviewed source dependency: {line.strip()}")
    if failures:
        print("Custom-node dependency audit failed:", *failures, sep="\n  - ")
        return 1
    print(f"Custom-node dependency audit passed ({len(nodes)} pinned packs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
