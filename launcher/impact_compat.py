"""Targeted Impact Pack frontend compatibility repair (upstream issue #1244).

Standard-library only: this file can also repair an existing GPU instance.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


_MARKER = "// DSNN Impact frontend compatibility v1 (upstream issue #1244)"
_STACK_HELPER = """// DSNN Impact frontend compatibility v1 (upstream issue #1244)
function dsnnImpactStackTrace() {
    const previousLimit = Error.stackTraceLimit;
    try {
        Error.stackTraceLimit = Infinity;
        return new Error().stack || "";
    } finally {
        Error.stackTraceLimit = previousLimit;
    }
}

"""


def repair_impact_frontend(repository: Path) -> bool:
    """Preserve restored slots and use the node-object connect API.

    Do not delete switches, alter saved workflows, downgrade the frontend,
    or install dependencies. Keep the legacy self-connection no-op, but pass
    an actual node so string IDs cannot reach findInputSlot.
    """
    source = repository / "js" / "impact-pack.js"
    original = source.read_text(encoding="utf-8")
    if _MARKER in original:
        if (
            original.count("const stackTrace = dsnnImpactStackTrace();") == 3
            and original.count("(app.configuringGraph || stackTrace.includes('loadGraphData'))") == 3
            and "node.connect(link_info.origin_slot, node, 'input1');" in original
            and "node.connect(link_info.origin_slot, node.id, 'input1');" not in original
        ):
            return False
        raise RuntimeError("Impact Pack frontend repair is incomplete; original file was left unchanged.")

    edits = (
        ("const is_legacy_front = ", _STACK_HELPER + "const is_legacy_front = ", 1),
        ("const stackTrace = new Error().stack;", "const stackTrace = dsnnImpactStackTrace();", 3),
        (
            "stackTrace.includes('loadGraphData')",
            "(app.configuringGraph || stackTrace.includes('loadGraphData'))",
            3,
        ),
        (
            "node.connect(link_info.origin_slot, node.id, 'input1');",
            "node.connect(link_info.origin_slot, node, 'input1');",
            1,
        ),
    )
    updated = original
    for before, after, count in edits:
        if updated.count(before) != count:
            raise RuntimeError(
                "Impact Pack script changed; the reviewed frontend compatibility "
                "repair could not be applied safely. No file was changed."
            )
        updated = updated.replace(before, after)
    # This MUST NOT end with .js: ComfyUI serves every JS file in this directory.
    backup = source.with_suffix(".js.dsnn-backup")
    if not backup.exists():
        shutil.copy2(source, backup)
    source.write_text(updated, encoding="utf-8")
    return True


def find_repository(path: Path) -> Path:
    if (path / "js" / "impact-pack.js").is_file():
        return path
    custom_nodes = path / "custom_nodes"
    candidates = (
        [p for p in custom_nodes.iterdir()
         if p.name.lower() == "comfyui-impact-pack"
         and (p / "js" / "impact-pack.js").is_file()]
        if custom_nodes.is_dir() else []
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not locate exactly one Impact Pack. "
            "Pass its repository directory or the correct ComfyUI directory."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="ComfyUI or Impact Pack repository directory")
    args = parser.parse_args()
    try:
        repository = find_repository(args.path)
        changed = repair_impact_frontend(repository)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"Repair failed: {exc}\n")
    print("Impact Pack patched." if changed else "Impact Pack already patched.")
    print("Hard-refresh ComfyUI (Ctrl+Shift+R), then reopen the original workflow JSON.")


if __name__ == "__main__":
    main()

