"""Regression tests for Impact Pack workflow restore compatibility."""
import asyncio
from pathlib import Path

import pytest

from launcher import app as launcher_app
from launcher import impact_compat


def fixture_source():
    return """const is_legacy_front = () => false;
function first() { const stackTrace = new Error().stack;
    return !stackTrace.includes('loadGraphData');
}
function second() { const stackTrace = new Error().stack;
    return stackTrace.includes('loadGraphData');
}
function third() { const stackTrace = new Error().stack;
    return !stackTrace.includes('loadGraphData');
}
function reconnect(node, link_info) {
    node.connect(link_info.origin_slot, node.id, 'input1');
}
"""


def repository(root):
    source = root / "js" / "impact-pack.js"
    source.parent.mkdir(parents=True)
    source.write_text(fixture_source())
    return source


def test_repair_preserves_backup_and_covers_all_load_detection_sites(tmp_path):
    source = repository(tmp_path)
    original = source.read_text()
    assert impact_compat.repair_impact_frontend(tmp_path)
    result = source.read_text()
    assert result.count("const stackTrace = dsnnImpactStackTrace();") == 3
    assert result.count("(app.configuringGraph || stackTrace.includes('loadGraphData'))") == 3
    assert "node.connect(link_info.origin_slot, node, 'input1');" in result
    assert "node.id, 'input1'" not in result
    assert "finally" in result
    assert "Error.stackTraceLimit = previousLimit" in result
    assert source.with_suffix(".js.dsnn-backup").read_text() == original
    assert list(source.parent.glob("*.js")) == [source]
    assert not impact_compat.repair_impact_frontend(tmp_path)
    assert source.read_text() == result
    assert source.with_suffix(".js.dsnn-backup").read_text() == original


def test_unknown_script_is_not_modified(tmp_path):
    source = repository(tmp_path)
    source.write_text(source.read_text().replace("node.id, 'input1'", "someNewApi, 'input1'"))
    original = source.read_text()
    with pytest.raises(RuntimeError, match="could not be applied safely"):
        impact_compat.repair_impact_frontend(tmp_path)
    assert source.read_text() == original
    assert not source.with_suffix(".js.dsnn-backup").exists()


def test_incomplete_existing_repair_is_not_reported_as_success(tmp_path):
    source = repository(tmp_path)
    source.write_text(impact_compat._MARKER + "\n" + source.read_text())
    with pytest.raises(RuntimeError, match="incomplete"):
        impact_compat.repair_impact_frontend(tmp_path)


@pytest.mark.parametrize("folder", ["ComfyUI-Impact-Pack", "comfyui-impact-pack"])
def test_cli_finds_both_manager_and_catalog_folder_names(tmp_path, folder):
    root = tmp_path / "ComfyUI"
    pack = root / "custom_nodes" / folder
    repository(pack)
    assert impact_compat.find_repository(root) == pack
    assert impact_compat.find_repository(pack) == pack


def test_no_impact_package_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="Could not locate"):
        impact_compat.find_repository(tmp_path)


def test_workflow_applies_impact_fix_before_restart(tmp_path, monkeypatch):
    source = repository(tmp_path)

    async def prepare(*args):
        return tmp_path

    monkeypatch.setattr(launcher_app, "prepare_repository", prepare)
    controller = launcher_app.JobController()
    asyncio.run(controller._install_custom_node({
        "name": "ComfyUI-Impact-Pack",
        "repo": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "install_requirements": False,
    }))
    assert impact_compat._MARKER in source.read_text()
    assert controller.state.restart_required


def test_manual_node_repair_applies_impact_fix(tmp_path):
    source = repository(tmp_path)
    controller = launcher_app.CustomNodeController()
    item = launcher_app.CustomNodeState(
        id="impact", url="https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
    )
    asyncio.run(controller._repair_requirements(tmp_path, item))
    assert impact_compat._MARKER in source.read_text()

