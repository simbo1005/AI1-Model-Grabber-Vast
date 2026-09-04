#!/usr/bin/env python3
"""Create the public Docker Hub repository used by the publish workflow."""

from __future__ import annotations

import os
import sys

import httpx


NAMESPACE = "sdcioba"
REPOSITORY = "comfyui-workflow-launcher-vast"
API_ROOT = "https://hub.docker.com/v2"


def main() -> int:
    secret = os.getenv("DOCKERHUB_TOKEN", "").strip()
    if not secret:
        print("DOCKERHUB_TOKEN is not set.", file=sys.stderr)
        return 1

    with httpx.Client(timeout=30.0) as client:
        auth = client.post(
            f"{API_ROOT}/auth/token",
            json={"identifier": NAMESPACE, "secret": secret},
        )
        auth.raise_for_status()
        access_token = auth.json().get("access_token", "")
        if not access_token:
            raise RuntimeError("Docker Hub authentication did not return an access token.")

        headers = {"Authorization": f"Bearer {access_token}"}
        repository_url = (
            f"{API_ROOT}/namespaces/{NAMESPACE}/repositories/{REPOSITORY}"
        )
        existing = client.get(repository_url, headers=headers)
        if existing.status_code == 200:
            if existing.json().get("is_private"):
                raise RuntimeError(
                    f"{NAMESPACE}/{REPOSITORY} exists but is private; make it public "
                    "in Docker Hub before publishing."
                )
            print(f"Docker Hub repository {NAMESPACE}/{REPOSITORY} already exists.")
            return 0
        if existing.status_code != 404:
            existing.raise_for_status()

        created = client.post(
            f"{API_ROOT}/namespaces/{NAMESPACE}/repositories",
            headers=headers,
            json={
                "name": REPOSITORY,
                "namespace": NAMESPACE,
                "description": "Vast.ai-native ComfyUI workflow and model downloader",
                "registry": "docker.io",
                "is_private": False,
            },
        )
        created.raise_for_status()
        if created.json().get("is_private"):
            raise RuntimeError("Docker Hub created the repository as private.")
        print(f"Created public Docker Hub repository {NAMESPACE}/{REPOSITORY}.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
