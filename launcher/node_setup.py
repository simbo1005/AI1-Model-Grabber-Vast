"""Install pinned node repositories and verify the running ComfyUI registry."""
from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

RunProcess = Callable[..., Awaitable[tuple[int, str]]]


def repair_seedvr_optional_imports(repository: Path) -> bool:
    """Apply SeedVR2 PR #641 without replacing Torch or optional backends.

    SeedVR2 2.5.24 creates phantom modules even when Flash Attention or
    bitsandbytes is absent. Their truthy specs break xformers/peft detection.
    Keep the existing fallback for installed-but-broken libraries unchanged.
    """
    source = repository / "src" / "optimization" / "compatibility.py"
    original = source.read_text(encoding="utf-8")
    updated = original
    for package, comment in (
        ("flash_attn", "DLL broken or not installed - create stub with proper __spec__"),
        ("bitsandbytes", "Installation broken, not present, or version detection failed - create stub"),
    ):
        guard = f"        if importlib.machinery.PathFinder().find_spec('{package}') is None:\n            return\n"
        if guard in updated:
            continue
        before = f"        # {comment}\n        stub = types.ModuleType('{package}')"
        after = (
            "        # Do not advertise an absent optional backend (SeedVR2 PR #641).\n"
            + guard + f"        stub = types.ModuleType('{package}')"
        )
        if updated.count(before) != 1:
            raise RuntimeError(
                f"SeedVR2 compatibility code for {package} changed; "
                "the reviewed optional-backend fix could not be applied safely."
            )
        updated = updated.replace(before, after, 1)
    if updated == original:
        return False
    backup = source.with_suffix(".py.dsnn-backup")
    if not backup.exists():
        shutil.copy2(source, backup)
    source.write_text(updated, encoding="utf-8")
    return True


SYSTEM_TOOLS = {
    "libimage-exiftool-perl": ("exiftool", "-ver"),
    "ffmpeg": ("ffmpeg", "-version"),
}
_PROTECTED_PACKAGE = re.compile(
    r"^(?:torch|torchvision|torchaudio|torchcodec|numpy)==[^\s]+$", re.IGNORECASE
)


async def ensure_system_tools(
    packages: list[str], run: RunProcess, notify: Callable[[str], None]
) -> None:
    unknown = set(packages) - SYSTEM_TOOLS.keys()
    if unknown:
        raise RuntimeError("Unsupported system packages: " + ", ".join(sorted(unknown)))
    missing = [name for name in packages if not shutil.which(SYSTEM_TOOLS[name][0])]
    if missing:
        if not shutil.which("apt-get"):
            raise RuntimeError("Missing system tools; apt-get is unavailable: " + ", ".join(missing))
        notify("Installing ExifTool and required system tools…")
        for command in (
            ("apt-get", "-o", "Acquire::Retries=3", "update"),
            ("apt-get", "install", "-y", "--no-install-recommends", *missing),
        ):
            code, output = await run(*command)
            if code:
                raise RuntimeError("System package installation failed: " + output[-1500:])
    for package in packages:
        executable, flag = SYSTEM_TOOLS[package]
        code, output = await run(executable, flag)
        if code:
            raise RuntimeError(f"{executable} could not run after installation: {output[-500:]}")


async def install_requirements(
    requirements: Path,
    python: Path,
    run: RunProcess,
    retry_with_isolation: Callable[[str], bool],
    *,
    extra_index_url: str = "",
) -> tuple[int, str]:
    """Constrain installed CUDA packages so a node cannot silently replace them."""
    if extra_index_url and extra_index_url != "https://pypi.nvidia.com/":
        raise RuntimeError("Unsupported Python package index: " + extra_index_url)
    code, frozen = await run(python, "-m", "pip", "freeze", "--disable-pip-version-check")
    if code:
        raise RuntimeError("Could not inspect ComfyUI's Python packages: " + frozen[-1000:])
    protected = [line.strip() for line in frozen.splitlines()
                 if _PROTECTED_PACKAGE.fullmatch(line.strip())]
    with tempfile.TemporaryDirectory(prefix="dsnn-node-requirements-") as directory:
        constraints = Path(directory) / "constraints.txt"
        constraints.write_text("\n".join(protected) + "\n", encoding="utf-8")
        command = [
            python, "-m", "pip", "install", "--disable-pip-version-check",
            "--timeout", "30", "--retries", "3", "--no-build-isolation",
            "--constraint", constraints, "-r", requirements,
        ]
        if extra_index_url:
            command.extend(["--extra-index-url", extra_index_url])
        code, output = await run(*command)
        if code and retry_with_isolation(output):
            command.remove("--no-build-isolation")
            code, output = await run(*command)
        return code, output


async def prepare_repository(
    node: dict[str, Any], root: Path, run: RunProcess
) -> Path:
    """Only expose a clone to ComfyUI after its pinned checkout is complete."""
    name = str(node.get("name", "")).strip()
    repo = str(node.get("repo", "")).strip()
    ref = str(node.get("ref", "")).strip()
    if name.startswith(".") or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise RuntimeError(f"Unsafe custom node name: {name!r}")
    if not re.fullmatch(r"https://github\.com/[^/\s]+/[^/\s]+", repo):
        raise RuntimeError(f"Custom node {name} must use a GitHub HTTPS repository.")
    if ref and not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise RuntimeError(f"Custom node {name} must use a pinned commit SHA.")
    root.mkdir(parents=True, exist_ok=True)
    destination = (root / name).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise RuntimeError(f"Unsafe custom node destination: {name}")
    staging = root / (".dsnn-" + uuid4().hex + ".part")
    checkout = destination if destination.exists() else staging
    try:
        if checkout == staging:
            code, output = await run(
                "git", "clone", "--filter=blob:none", "--depth=1", repo, checkout
            )
            if code:
                raise RuntimeError(f"Could not clone {name}: {output[-1500:]}")
        else:
            code, origin = await run("git", "-C", checkout, "remote", "get-url", "origin")
            expected = repo.rstrip("/").removesuffix(".git").lower()
            actual = origin.strip().rstrip("/").removesuffix(".git").lower()
            if code or actual != expected:
                raise RuntimeError(f"The existing {name} folder is not the expected Git repository.")
        if ref:
            code, _ = await run("git", "-C", checkout, "cat-file", "-e", ref + "^{commit}")
            if code:
                code, output = await run(
                    "git", "-C", checkout, "fetch", "--no-tags", "--filter=blob:none",
                    "--depth=1", "origin", ref,
                )
                if code:
                    raise RuntimeError(f"Could not fetch the pinned version for {name}: {output[-1500:]}")
            code, output = await run("git", "-C", checkout, "checkout", "--detach", ref)
            if code:
                raise RuntimeError(f"Could not select the pinned version for {name}: {output[-1500:]}")
        if not (checkout / "__init__.py").is_file():
            raise RuntimeError(f"{name} has no __init__.py; its node package is incomplete.")
        if checkout == staging:
            staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def node_registration_checks(
    nodes: list[dict[str, Any]], registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Check package origins and explicit workflow-critical node IDs."""
    def normalized(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    checks = []
    for node in nodes:
        name = node["name"]
        modules = {normalized(name), *(normalized(x) for x in node.get("module_aliases", []))}
        registered = [
            node_type for node_type, info in registry.items()
            if isinstance(info, dict)
            and normalized(str(info.get("python_module", "")).rsplit(".", 1)[-1]) in modules
        ]
        required = node.get("required_node_types", [])
        missing = [node_type for node_type in required
                   if not isinstance(registry.get(node_type), dict)]
        valid = not missing and (bool(registered) or bool(required))
        if valid:
            message = "Loaded in ComfyUI."
        elif missing:
            message = "Missing node types: " + ", ".join(missing)
        else:
            message = "ComfyUI did not register this package; check its startup import error."
        checks.append({
            "name": name,
            "status": "verified" if valid else "missing",
            "node_count": len(registered) or (len(required) - len(missing)),
            "message": message,
        })
    return checks

