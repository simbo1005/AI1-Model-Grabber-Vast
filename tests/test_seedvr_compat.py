"""Regression tests for SeedVR2 upstream issue #640 / proposed fix #641."""
import asyncio
import builtins
import importlib.machinery
import sys

import pytest

from launcher import app as launcher_app
from launcher.node_setup import repair_seedvr_optional_imports


def compatibility_source():
    source = "import importlib.machinery\nimport sys\nimport types\n\n"
    for package, comment in (
        ("flash_attn", "DLL broken or not installed - create stub with proper __spec__"),
        ("bitsandbytes", "Installation broken, not present, or version detection failed - create stub"),
    ):
        source += (
            f"def ensure_{package}_safe():\n"
            f"    try:\n        import {package}\n"
            f"    except (ImportError, OSError):\n"
            f"        # {comment}\n"
            f"        stub = types.ModuleType('{package}')\n"
            f"        stub.__spec__ = importlib.machinery.ModuleSpec('{package}', None)\n"
            f"        sys.modules['{package}'] = stub\n\n"
        )
    return source


def make_repository(tmp_path):
    source = tmp_path / "src" / "optimization" / "compatibility.py"
    source.parent.mkdir(parents=True)
    source.write_text(compatibility_source())
    return source


@pytest.mark.parametrize("installed", [False, True])
def test_absent_backends_are_not_faked_but_broken_installed_fallback_is_preserved(
    tmp_path, monkeypatch, installed
):
    source = make_repository(tmp_path)
    assert repair_seedvr_optional_imports(tmp_path)
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name in ("flash_attn", "bitsandbytes"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec",
                        lambda *args, **kwargs: object() if installed else None)
    namespace = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    for package in ("flash_attn", "bitsandbytes"):
        monkeypatch.delitem(sys.modules, package, raising=False)
        # Ensure monkeypatch restores the previous entry after executing the shim.
        monkeypatch.setitem(sys.modules, package, None)
        del sys.modules[package]
        namespace[f"ensure_{package}_safe"]()
        assert (package in sys.modules) == installed


def test_fix_is_idempotent_and_preserves_original_backup(tmp_path):
    source = make_repository(tmp_path)
    original = source.read_text()
    assert repair_seedvr_optional_imports(tmp_path)
    updated = source.read_text()
    assert not repair_seedvr_optional_imports(tmp_path)
    assert source.read_text() == updated
    assert source.with_suffix(".py.dsnn-backup").read_text() == original


def test_unknown_source_is_left_untouched(tmp_path):
    source = make_repository(tmp_path)
    source.write_text(source.read_text().replace(
        "stub = types.ModuleType('bitsandbytes')", "raise RuntimeError('changed upstream')"
    ))
    original = source.read_text()
    with pytest.raises(RuntimeError, match="could not be applied safely"):
        repair_seedvr_optional_imports(tmp_path)
    assert source.read_text() == original
    assert not source.with_suffix(".py.dsnn-backup").exists()


def test_workflow_install_applies_fix_before_restart(tmp_path, monkeypatch):
    source = make_repository(tmp_path)

    async def prepare(*args):
        return tmp_path

    monkeypatch.setattr(launcher_app, "prepare_repository", prepare)
    controller = launcher_app.JobController()
    asyncio.run(controller._install_custom_node({
        "name": "ComfyUI-SeedVR2_VideoUpscaler",
        "repo": "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git",
        "install_requirements": False,
    }))
    assert "PathFinder().find_spec('flash_attn')" in source.read_text()
    assert controller.state.restart_required


def test_manual_node_repair_also_applies_fix(tmp_path):
    source = make_repository(tmp_path)
    controller = launcher_app.CustomNodeController()
    item = launcher_app.CustomNodeState(
        id="seedvr", url="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
    )
    asyncio.run(controller._repair_requirements(tmp_path, item))
    assert "PathFinder().find_spec('flash_attn')" in source.read_text()

