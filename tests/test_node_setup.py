import asyncio
from pathlib import Path

import httpx
import pytest

from launcher import app as launcher_app
from launcher import node_setup


def test_every_pipeline_includes_bond_and_exiftool():
    root = Path(launcher_app.__file__).resolve().parents[1]
    assert "libimage-exiftool-perl" in (root / "Dockerfile").read_text()
    catalog = launcher_app.load_catalog()
    for workflow in catalog["workflows"]:
        assert "libimage-exiftool-perl" in workflow["system_packages"]
        bond = next(n for n in workflow["custom_nodes"] if n["name"] == "Bond-Node-Suite")
        assert bond["repo"] == "https://github.com/bondngn-studios/Bond-Node-Suite.git"
        assert len(bond["ref"]) == 40
        assert len(bond["required_node_types"]) == 26
        assert "BondReadMetadata" in bond["required_node_types"]
    seed = next(n for w in catalog["workflows"] for n in w["custom_nodes"]
                if n["name"] == "ComfyUI-SeedVR2_VideoUpscaler")
    assert set(seed["required_node_types"]) == {
        "SeedVR2VideoUpscaler", "SeedVR2LoadDiTModel",
        "SeedVR2LoadVAEModel", "SeedVR2TorchCompileSettings",
    }


def test_repository_clone_is_atomic_and_failed_clone_leaves_no_package(tmp_path):
    root = tmp_path / "custom_nodes"
    node = {"name": "Test-Pack", "repo": "https://github.com/example/Test-Pack.git", "ref": "a" * 40}
    calls = []

    async def run(*command):
        calls.append(tuple(map(str, command)))
        if "clone" in command:
            assert not (root / node["name"]).exists()
            destination = Path(command[-1])
            destination.mkdir()
            (destination / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}")
        return 0, ""

    destination = asyncio.run(node_setup.prepare_repository(node, root, run))
    assert destination == root / node["name"]
    assert (destination / "__init__.py").is_file()
    assert not list(root.glob(".dsnn-*"))
    assert any("checkout" in c and node["ref"] in c for c in calls)

    failed = {"name": "Broken-Pack", "repo": "https://github.com/example/Broken-Pack.git"}

    async def broken_clone(*command):
        if "clone" in command:
            Path(command[-1]).mkdir()
            return 1, "connection reset"
        return 0, ""

    with pytest.raises(RuntimeError, match="Could not clone"):
        asyncio.run(node_setup.prepare_repository(failed, root, broken_clone))
    assert not (root / failed["name"]).exists()
    assert not list(root.glob(".dsnn-*"))


def test_incomplete_existing_node_is_not_accepted_or_deleted(tmp_path):
    destination = tmp_path / "Test-Pack"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("user data")

    async def run(*command):
        return 0, "https://github.com/example/Test-Pack.git"

    with pytest.raises(RuntimeError, match="incomplete"):
        asyncio.run(node_setup.prepare_repository(
            {"name": "Test-Pack", "repo": "https://github.com/example/Test-Pack.git"},
            tmp_path, run,
        ))
    assert marker.read_text() == "user data"


def test_node_requirements_constrain_existing_cuda_and_numpy(tmp_path):
    python = Path("/venv/comfy/bin/python")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("peft\ntorch\nnumpy\n")
    calls = []
    constraint_path = None

    async def run(*command):
        nonlocal constraint_path
        assert command[0] == python
        calls.append(tuple(map(str, command)))
        if "freeze" in command:
            return 0, "torch==2.10.0+cu128\ntorchvision==0.25.0+cu128\nnumpy==2.3.0\npeft==0.17.0\n"
        constraint_path = Path(command[command.index("--constraint") + 1])
        constraints = constraint_path.read_text()
        assert "torch==2.10.0+cu128" in constraints
        assert "torchvision==0.25.0+cu128" in constraints
        assert "numpy==2.3.0" in constraints
        assert "peft" not in constraints
        return 0, "installed"

    code, _ = asyncio.run(node_setup.install_requirements(
        requirements, python, run, lambda _: False,
    ))
    assert code == 0
    assert len(calls) == 2
    assert not constraint_path.exists()


def test_system_tool_install_uses_the_debian_exiftool_package(monkeypatch):
    monkeypatch.setattr(node_setup.shutil, "which",
                        lambda name: None if name == "exiftool" else "/usr/bin/" + name)
    commands = []

    async def run(*command):
        commands.append(command)
        return 0, "ready"

    asyncio.run(node_setup.ensure_system_tools(
        ["libimage-exiftool-perl", "ffmpeg"], run, lambda _: None,
    ))
    assert commands == [
        ("apt-get", "-o", "Acquire::Retries=3", "update"),
        ("apt-get", "install", "-y", "--no-install-recommends", "libimage-exiftool-perl"),
        ("exiftool", "-ver"),
        ("ffmpeg", "-version"),
    ]


def test_broken_system_tool_is_not_accepted(monkeypatch):
    monkeypatch.setattr(node_setup.shutil, "which", lambda name: "/usr/bin/" + name)

    async def run(*command):
        return 1, "broken executable"

    with pytest.raises(RuntimeError, match="exiftool could not run"):
        asyncio.run(node_setup.ensure_system_tools(["libimage-exiftool-perl"], run, lambda _: None))


def test_node_registry_detects_missing_packages_and_partial_seedvr():
    nodes = [
        {"name": "Example-Pack"},
        {"name": "Missing-Pack"},
        {"name": "SeedVR2", "required_node_types": ["SeedVR2VideoUpscaler", "SeedVR2LoadVAEModel"]},
    ]
    registry = {
        "ExampleNode": {"python_module": "custom_nodes.Example_Pack"},
        "SeedVR2VideoUpscaler": {"python_module": "custom_nodes.SeedVR2"},
    }
    checks = node_setup.node_registration_checks(nodes, registry)
    assert [c["status"] for c in checks] == ["verified", "missing", "missing"]
    assert "SeedVR2LoadVAEModel" in checks[2]["message"]


@pytest.mark.parametrize("registered,install_failed,expected", [
    (True, False, "complete"),
    (False, False, "error"),
    (True, True, "error"),
])
def test_workflow_readiness_depends_on_live_node_registration(
    registered, install_failed, expected, monkeypatch,
):
    controller = launcher_app.JobController()
    node = {"name": "Test-Pack"}
    registry = {"TestNode": {"python_module": "custom_nodes.Test-Pack"}} if registered else {
        "CoreNode": {"python_module": "nodes"}
    }
    events = []

    class Restart:
        async def start(self):
            events.append("restart")
        async def wait(self):
            events.append("ready")

    async def install(_workflow):
        events.append("install")
        if install_failed:
            controller.node_failures["Test-Pack"] = "Dependency install failed"

    def handle(request):
        events.append("registry")
        assert request.url.path == "/object_info"
        return httpx.Response(200, json=registry)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(controller, "_install_workflow", install)
    monkeypatch.setattr(launcher_app, "comfy_service_controller", Restart())
    monkeypatch.setattr(launcher_app.httpx, "AsyncClient", lambda **_: client)
    asyncio.run(controller._run({"id": "test", "custom_nodes": [node]}))
    assert events == ["install", "restart", "ready", "registry"]
    assert controller.state.status == expected
    assert controller.state.node_checks[0]["status"] == (
        "error" if install_failed else "verified" if registered else "missing"
    )
    if expected == "error":
        assert "Test-Pack" in controller.state.error


def test_failed_restart_cannot_mark_a_node_workflow_ready(monkeypatch):
    controller = launcher_app.JobController()

    async def install(_workflow):
        pass

    class Restart:
        async def start(self):
            pass
        async def wait(self):
            raise RuntimeError("ComfyUI failed to restart")

    monkeypatch.setattr(controller, "_install_workflow", install)
    monkeypatch.setattr(launcher_app, "comfy_service_controller", Restart())
    asyncio.run(controller._run({"id": "test", "custom_nodes": [{"name": "SeedVR2"}]}))
    assert controller.state.status == "error"
    assert "could not be verified" in controller.state.error


def test_transient_node_failure_is_retried(tmp_path, monkeypatch):
    controller = launcher_app.JobController()
    calls = []

    async def install(node):
        calls.append(node["name"])
        if len(calls) == 1:
            raise RuntimeError("connection reset")

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", tmp_path)
    monkeypatch.setattr(controller, "_install_custom_node", install)
    monkeypatch.setattr(launcher_app.asyncio, "sleep", no_sleep)
    asyncio.run(controller._install_custom_nodes([{"name": "Test-Pack"}]))
    assert calls == ["Test-Pack", "Test-Pack"]
    assert controller.node_failures == {}
    assert controller.state.warnings == []
    assert controller.state.node_checks[0]["status"] == "installed"


def test_workflow_waits_for_the_configured_python_environment(tmp_path, monkeypatch):
    root = tmp_path / "ComfyUI"
    root.mkdir()
    venv = tmp_path / "venv"
    controller = launcher_app.JobController()
    sleeps = []
    requests = []

    class Client:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
        async def get(self, url):
            requests.append(url)
            return httpx.Response(200)

    async def sleep(seconds):
        sleeps.append(seconds)
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").touch()

    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", root)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", venv)
    monkeypatch.setattr(launcher_app.httpx, "AsyncClient", lambda **_: Client())
    monkeypatch.setattr(launcher_app.asyncio, "sleep", sleep)
    asyncio.run(controller._wait_for_comfyui())
    assert sleeps == [1]
    assert len(requests) == 1

