import asyncio
import hashlib
import importlib
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient


launcher_app = importlib.import_module("launcher.app")


def test_vast_image_disables_serverless_worker_and_uses_compatible_shell() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    dockerfile = (repository_root / "Dockerfile").read_text(encoding="utf-8")
    launcher_script = (
        repository_root / "docker" / "vast" / "dsnn-launcher.sh"
    ).read_text(encoding="utf-8")

    assert "SERVERLESS=false" in dockerfile
    assert "SUPERVISOR_SKIP_PYWORKER=true" in dockerfile
    assert "set -Eeo pipefail" in launcher_script
    assert "set -Eeuo pipefail" not in launcher_script


def test_vast_public_service_urls(monkeypatch) -> None:
    monkeypatch.delenv("COMFYUI_PUBLIC_URL", raising=False)
    monkeypatch.delenv("JUPYTER_PUBLIC_URL", raising=False)
    monkeypatch.delenv("ENABLE_HTTPS", raising=False)
    monkeypatch.setenv("PUBLIC_IPADDR", "103.116.53.7")
    monkeypatch.setenv("VAST_TCP_PORT_8188", "34721")
    monkeypatch.setenv("VAST_TCP_PORT_8080", "34722")

    assert launcher_app.comfy_public_url() == "http://103.116.53.7:34721"
    assert launcher_app.jupyter_public_url() == "http://103.116.53.7:34722"


def test_explicit_service_urls_override_platform_detection(monkeypatch) -> None:
    monkeypatch.setenv("COMFYUI_PUBLIC_URL", "https://comfy.example.test/")
    monkeypatch.setenv("JUPYTER_PUBLIC_URL", "https://jupyter.example.test/")

    assert launcher_app.comfy_public_url() == "https://comfy.example.test"
    assert launcher_app.jupyter_public_url() == "https://jupyter.example.test"


def test_health_and_public_catalog() -> None:
    with TestClient(launcher_app.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.get("/api/catalog")
        assert response.status_code == 200
        workflows = response.json()["workflows"]
        assert len(workflows) == 5
        assert sum(not item.get("disabled", False) for item in workflows) == 5
        assert "files" not in workflows[0]
        assert "custom_nodes" not in workflows[0]
        assert "url" not in workflows[0]


def test_catalog_contains_installers_but_no_product_workflows() -> None:
    catalog = launcher_app.load_catalog()
    enabled = [item for item in catalog["workflows"] if not item.get("disabled")]

    assert [item["id"] for item in enabled] == [
        "krea-2-extended",
        "image-edit",
        "motion-control",
        "motion-control-god-edition",
        "minimax-h3",
    ]
    assert {
        "image-generation",
        "krea-2",
        "dataset-generator",
    }.isdisjoint(item["id"] for item in catalog["workflows"])
    assert all(item["files"] for item in enabled)
    assert all(item["custom_nodes"] for item in enabled)

    for installer in enabled:
        for file_spec in installer["files"]:
            destination = file_spec["destination"].lower()
            assert not destination.endswith(".json")
            assert "workflow" not in destination
            assert file_spec["size_bytes"] > 0
            assert len(file_spec["sha256"]) == 64
            assert file_spec["auth"] in {"none", "huggingface"}
            assert file_spec["parallel"] is True


def test_krea_2_extended_installer_matches_the_multiflow_registry() -> None:
    catalog = launcher_app.load_catalog()
    installer = next(
        item for item in catalog["workflows"] if item["id"] == "krea-2-extended"
    )

    assert installer["estimated_size"] == "Approx. 40.0 GB"
    assert sum(item["size_bytes"] for item in installer["files"]) == 40_044_495_122
    assert [item["destination"] for item in installer["files"]] == [
        "models/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "models/diffusion_models/krea2_raw_fp8_scaled.safetensors",
        "models/loras/krea2_turbo_lora_rank_64_bf16.safetensors",
        "models/loras/krea2_identity_edit_v1_2.safetensors",
        "models/loras/snofs_krea_v1_4.safetensors",
        "models/loras/krea2-bloomgirls-realism-step00004000.safetensors",
        "models/loras/ass_v2_krea2_loraholic.safetensors",
        "models/loras/breast_size_v2_krea2_loraholic.safetensors",
        "models/loras/famegrid_spicy.safetensors",
        "models/upscale_models/4xNMKDSuperscale_4xNMKDSuperscale.pt",
        "models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "models/vae/qwen_image_vae.safetensors",
        "models/vae/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors",
        "models/vae/Wan2_1_VAE_fp32.safetensors",
        "models/sams/sam_vit_b_01ec64.pth",
        "models/ultralytics/bbox/face_yolov8m.pt",
    ]
    assert [item["name"] for item in installer["custom_nodes"]] == [
        "comfyui-krea2edit",
        "ComfyUI_Comfyroll_CustomNodes",
        "ComfyUI-KJNodes",
        "ComfyUI-Impact-Subpack",
        "ComfyUI-Impact-Pack",
        "rgthree-comfy",
        "RES4LYF",
        "ComfyUI-VAE-Utils",
    ]
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", node["ref"])
        for node in installer["custom_nodes"]
    )


def test_motion_control_installer_matches_the_wan_manifest() -> None:
    catalog = launcher_app.load_catalog()
    installer = next(
        item for item in catalog["workflows"] if item["id"] == "motion-control"
    )

    assert installer["estimated_size"] == "Approx. 45.4 GB"
    assert sum(item["size_bytes"] for item in installer["files"]) == 45_373_632_603
    assert [item["destination"] for item in installer["files"]] == [
        "models/checkpoints/sam3.1_multiplex_fp16.safetensors",
        "models/clip_vision/clip_vision_h.safetensors",
        "models/diffusion_models/wan2.1_14B_SCAIL_2_fp16.safetensors",
        "models/vae/wan_2.1_vae.safetensors",
        "models/loras/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
        "models/loras/slop_twerk_LowNoise_merged3_7_v2.safetensors",
        "models/loras/slop_twerk_HighNoise_merged3_7_v2.safetensors",
        "models/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
        "models/text_encoders/umt5-xxl-encoder-fp8-e4m3fn-scaled.safetensors",
    ]
    assert [item["name"] for item in installer["custom_nodes"]] == [
        "ComfyUI-SAM3",
        "ComfyUI-VideoHelperSuite",
        "ComfyUI-Logic",
        "Nvidia_RTX_Nodes_ComfyUI",
        "ComfyUI-Easy-Use",
        "ComfyUI-Custom-Scripts",
        "ComfyUI-Impact-Pack",
    ]
    nvidia = next(
        item
        for item in installer["custom_nodes"]
        if item["name"] == "Nvidia_RTX_Nodes_ComfyUI"
    )
    assert nvidia["requirements_extra_index_url"] == "https://pypi.nvidia.com/"


def test_motion_control_god_edition_matches_the_workflow_export() -> None:
    catalog = launcher_app.load_catalog()
    installer = next(
        item
        for item in catalog["workflows"]
        if item["id"] == "motion-control-god-edition"
    )

    assert installer["estimated_size"] == "Approx. 61.3 GB"
    assert sum(item["size_bytes"] for item in installer["files"]) == 61_317_797_596
    assert installer["update_comfyui"] is True
    assert [item["destination"] for item in installer["files"]] == [
        "models/diffusion_models/wan2.2_animate_14B_bf16.safetensors",
        "models/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors",
        "models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
        "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
        "models/loras/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
        "models/loras/Wan2.2-Fun-A14B-InP-low-noise-MPS.safetensors",
        "models/vae/Wan2_1_VAE_bf16.safetensors",
        "models/text_encoders/umt5_xxl_fp16.safetensors",
        "models/clip_vision/clip_vision_h.safetensors",
        "models/detection/yolov10m.onnx",
        "models/detection/vitpose_h_wholebody_model.onnx",
        "models/detection/vitpose_h_wholebody_data.bin",
        "models/sams/sam2.1_hiera_base_plus.safetensors",
        "models/frame_interpolation/rife49.pth",
    ]
    assert [item["name"] for item in installer["custom_nodes"]] == [
        "ComfyUI-WanVideoWrapper",
        "ComfyUI-WanAnimatePreprocess",
        "ComfyUI-KJNodes",
        "ComfyUI-segment-anything-2",
        "ComfyUI-VideoHelperSuite",
        "ComfyUI-Frame-Interpolation",
        "ComfyUI-Custom-Scripts",
        "rgthree-comfy",
        "ComfyUI-Easy-Use",
        "ComfyMath",
        "comfyui-propost",
        "CRT-Nodes",
        "ComfyUI_Swwan",
    ]
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", node["ref"])
        for node in installer["custom_nodes"]
    )
    assert installer["model_links"] == [
        {
            "source": "models/frame_interpolation/rife49.pth",
            "destination": "custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth",
        }
    ]


def test_minimax_h3_installer_matches_the_catalog_manifest() -> None:
    catalog = launcher_app.load_catalog()
    installer = next(
        item for item in catalog["workflows"] if item["id"] == "minimax-h3"
    )

    assert installer["estimated_size"] == "Approx. 63.4 GB"
    assert installer["update_comfyui"] is True
    assert [item["destination"] for item in installer["files"]] == [
        "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "models/vae/minimax_h3_video_vae_fp16.safetensors",
        "models/vae/minimax_h3_audio_vae_fp32.safetensors",
    ]
    assert [item["name"] for item in installer["custom_nodes"]] == [
        "ComfyUI-KJNodes",
        "rgthree-comfy",
        "ComfyUI-VideoHelperSuite",
    ]
    assert all("/resolve/main/" in item["url"] for item in installer["files"])


def test_local_windows_installers_match_the_catalog() -> None:
    catalog = launcher_app.load_catalog()
    workflows = {item["id"]: item for item in catalog["workflows"]}
    installers = {
        "minimax_h3_model_installer.bat": "minimax-h3",
    }

    for filename, workflow_id in installers.items():
        script = (
            launcher_app.SOURCE_ROOT / "local-installers" / filename
        ).read_text(encoding="utf-8")
        workflow = workflows[workflow_id]
        downloads = [
            (url, destination.replace("\\", "/"), sha256)
            for url, destination, sha256 in re.findall(
                r'^call :download "([^"]+)" "([^"]+)" "([0-9a-f]{64})"',
                script,
                flags=re.MULTILINE,
            )
        ]
        nodes = re.findall(
            r'^call :install_node "([^"]+)" "([^"]+)" "([0-9a-f]{40})"',
            script,
            flags=re.MULTILINE,
        )

        assert downloads == [
            (item["url"], item["destination"], item["sha256"])
            for item in workflow["files"]
        ]
        assert nodes == [
            (item["name"], item["repo"], item["ref"])
            for item in workflow["custom_nodes"]
        ]
        assert "Get-FileHash -Algorithm SHA256" in script
        assert "checkout --detach" in script
        assert "pip install --disable-pip-version-check" in script


def test_unknown_workflow_cannot_start() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.post("/api/install/does-not-exist")
        assert response.status_code == 404


def test_huggingface_auth_can_use_baked_token_file(
    tmp_path,
    monkeypatch,
) -> None:
    token_file = tmp_path / "hf_token"
    token_file.write_text("hf_test_only", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN_FILE", str(token_file))

    url, headers = launcher_app.tokenized_request(
        {
            "name": "Gated test model",
            "url": "https://huggingface.co/example/model/resolve/main/model.safetensors",
            "auth": "huggingface",
        }
    )

    assert url.endswith("model.safetensors")
    assert headers["Authorization"] == "Bearer hf_test_only"


def test_wget_command_uses_resumable_native_downloads(tmp_path) -> None:
    command = launcher_app.wget_command(
        "https://huggingface.co/example/model/resolve/main/model.safetensors",
        {"Authorization": "Bearer hf_test_only"},
        destination_dir=tmp_path,
        resume=True,
    )

    assert command[:2] == ["wget", "--no-verbose"]
    assert "--continue" in command
    assert f"--directory-prefix={tmp_path}" in command
    assert "--header=Authorization: Bearer hf_test_only" in command
    assert command[-1].endswith("model.safetensors")


def test_huggingface_file_reference_parses_resolve_urls() -> None:
    assert launcher_app.huggingface_file_reference(
        "https://huggingface.co/Comfy-Org/example/resolve/main/models/model.safetensors"
    ) == (
        "Comfy-Org/example",
        "model",
        "main",
        "models/model.safetensors",
    )
    assert launcher_app.huggingface_file_reference(
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_b_01ec64.pth"
    ) == (
        "Gourieff/ReActor",
        "dataset",
        "main",
        "models/sams/sam_vit_b_01ec64.pth",
    )
    assert launcher_app.huggingface_file_reference(
        "https://example.com/owner/model/resolve/main/file"
    ) is None


def test_hf_download_command_uses_the_official_cli_without_a_token_argument(tmp_path) -> None:
    command = launcher_app.hf_download_command(
        ("Gourieff/ReActor", "dataset", "main", "models/sams/sam_vit_b_01ec64.pth"),
        destination_dir=tmp_path,
    )

    assert command[:3] == ["hf", "download", "Gourieff/ReActor"]
    assert "--repo-type" in command
    assert "dataset" in command
    assert "--token" not in command
    assert command[-1] == str(tmp_path)




def test_xet_progress_reads_the_active_incomplete_file(tmp_path) -> None:
    incomplete = tmp_path / ".cache" / "huggingface" / "download" / "model.incomplete"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_bytes(b"x" * 4096)

    assert launcher_app.xet_incomplete_bytes(tmp_path, 0) == 4096


def test_workflow_download_resets_file_metrics_before_the_next_model(tmp_path, monkeypatch) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.JobController()
    controller.state.file_downloaded_bytes = 8 * 1024**3
    controller.state.file_total_bytes = 8 * 1024**3

    async def fake_download(*_args, **_kwargs) -> int:
        return 0

    monkeypatch.setattr(controller, "_download_file_with_wget", fake_download)
    monkeypatch.setattr(launcher_app.shutil, "which", lambda _name: "wget")
    async def invoke() -> None:
        await controller._download_file(
            object(),
            {
                "name": "next-model.safetensors",
                "url": "https://example.test/next-model.safetensors",
                "destination": "models/checkpoints/next-model.safetensors",
                "size_bytes": 9 * 1024**3,
            },
            1,
            2,
            8 * 1024**3,
            17 * 1024**3,
            99,
        )

    asyncio.run(invoke())
    assert controller.state.file_downloaded_bytes == 0
    assert controller.state.file_total_bytes == 9 * 1024**3


def test_pip_build_isolation_retry_is_only_used_for_missing_backends() -> None:
    assert launcher_app.needs_build_isolation("ModuleNotFoundError: No module named 'cmake'")
    assert not launcher_app.needs_build_isolation("Read timed out while fetching a wheel")
    assert not launcher_app.needs_build_isolation("A package version conflict occurred")


def test_diagnostics_are_redacted_and_human_readable() -> None:
    controller = launcher_app.JobController()
    controller.record_diagnostic(
        "download",
        "https://token@example.test/private/model at /workspace/secret",
        time.monotonic() - 2,
    )
    previous = launcher_app.controller
    launcher_app.controller = controller
    try:
        with TestClient(launcher_app.app) as client:
            report = client.get("/api/diagnostics/report")
            assert report.status_code == 200
            assert "example.test" not in report.text
            assert "/workspace" not in report.text
            assert "download:" in report.text
    finally:
        launcher_app.controller = previous


def test_frontend_is_served() -> None:
    with TestClient(launcher_app.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "dsnn Model Grabber" in response.text
        assert "Custom models" in response.text
        assert "Download queue" in response.text
        assert "Custom nodes" in response.text
        assert "Install queue" in response.text
        assert 'id="job-warnings"' in response.text
        assert 'id="restart-button"' in response.text
        assert 'id="custom-node-restart-button"' in response.text

        logo = client.get("/logo.png")
        assert logo.status_code == 200
        assert logo.headers["content-type"] == "image/png"


def test_real_download_writes_and_verifies_file(tmp_path, monkeypatch) -> None:
    payload = b"dsnn-download-test-" * 32768
    expected_hash = hashlib.sha256(payload).hexdigest()

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(
        launcher_app,
        "CUSTOM_NODES_DIR",
        comfy_dir / "custom_nodes",
    )

    workflow = {
        "id": "real-download",
        "title": "Real Download",
        "files": [
            {
                "name": "test-model.safetensors",
                "url": f"http://127.0.0.1:{server.server_port}/model",
                "destination": "models/checkpoints/test-model.safetensors",
                "size_bytes": len(payload),
                "sha256": expected_hash,
                "auth": "none",
            }
        ],
        "custom_nodes": [],
    }
    controller = launcher_app.JobController()
    try:
        asyncio.run(controller._run(workflow))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    destination = comfy_dir / "models" / "checkpoints" / "test-model.safetensors"
    assert destination.read_bytes() == payload
    assert controller.state.status == "complete"
    assert controller.state.percent == 100


def test_custom_model_locations_include_defaults_and_existing_folders(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "models" / "sams").mkdir(parents=True)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    locations = launcher_app.available_model_locations()

    assert "checkpoints" in locations
    assert "diffusion_models" in locations
    assert "text_encoders" in locations
    assert "controlnet" in locations
    assert "sams" in locations


def test_custom_model_location_cannot_escape_models(tmp_path, monkeypatch) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    try:
        launcher_app.validate_model_location("sams/../../outside")
    except RuntimeError as exc:
        assert "safe" in str(exc).lower()
    else:
        raise AssertionError("Path traversal should be rejected.")


def test_custom_download_deletes_partial_and_starts_from_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"fresh-custom-model" * 65536
    received_range_headers: list[str | None] = []

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received_range_headers.append(self.headers.get("Range"))
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    destination_dir = comfy_dir / "models" / "sams"
    destination_dir.mkdir(parents=True)
    partial = destination_dir / "model.safetensors.part"
    partial.write_bytes(b"corrupt partial data")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)

    controller = launcher_app.CustomModelController()

    async def run_download() -> launcher_app.CustomModelState:
        item = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/model.safetensors",
            "sams",
        )
        if controller.worker_task:
            await controller.worker_task
        return controller.items[item["id"]]

    try:
        state = asyncio.run(run_download())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    destination = destination_dir / "model.safetensors"
    assert state.status == "complete"
    assert destination.read_bytes() == payload
    assert not partial.exists()
    assert received_range_headers == [None]


def test_custom_download_queue_is_strictly_sequential(tmp_path, monkeypatch) -> None:
    payload = b"queued-model" * 32768
    counter_lock = threading.Lock()
    active_requests = 0
    maximum_active_requests = 0

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal active_requests, maximum_active_requests
            with counter_lock:
                active_requests += 1
                maximum_active_requests = max(maximum_active_requests, active_requests)
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                midpoint = len(payload) // 2
                self.wfile.write(payload[:midpoint])
                self.wfile.flush()
                time.sleep(0.05)
                self.wfile.write(payload[midpoint:])
            finally:
                with counter_lock:
                    active_requests -= 1

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.CustomModelController()

    async def run_downloads() -> list[launcher_app.CustomModelState]:
        first = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/first.safetensors",
            "checkpoints",
        )
        second = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/second.safetensors",
            "loras",
        )
        if controller.worker_task:
            await controller.worker_task
        return [controller.items[first["id"]], controller.items[second["id"]]]

    try:
        states = asyncio.run(run_downloads())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert maximum_active_requests == 1
    assert [state.status for state in states] == ["complete", "complete"]
    assert (comfy_dir / "models" / "checkpoints" / "first.safetensors").exists()
    assert (comfy_dir / "models" / "loras" / "second.safetensors").exists()


def test_existing_custom_model_is_moved_to_downloaded_as_found(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"already-installed-model"

    class DownloadHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    comfy_dir = tmp_path / "ComfyUI"
    destination = comfy_dir / "models" / "vae" / "existing.safetensors"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.CustomModelController()

    async def run_download() -> launcher_app.CustomModelState:
        item = await controller.enqueue(
            f"http://127.0.0.1:{server.server_port}/existing.safetensors",
            "vae",
        )
        if controller.worker_task:
            await controller.worker_task
        return controller.items[item["id"]]

    try:
        state = asyncio.run(run_download())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    snapshot = controller.snapshot()
    assert state.status == "skipped"
    assert snapshot["queue"] == []
    assert snapshot["downloaded"][0]["status"] == "skipped"
    assert destination.read_bytes() == payload


def test_custom_node_url_must_be_a_github_repository() -> None:
    assert (
        launcher_app.validate_custom_node_url("https://github.com/example/ComfyUI-Test")
        == "https://github.com/example/ComfyUI-Test.git"
    )

    for invalid in (
        "https://example.com/example/ComfyUI-Test",
        "https://github.com/example/ComfyUI-Test/issues",
        "http://github.com/example/ComfyUI-Test",
    ):
        try:
            launcher_app.validate_custom_node_url(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Invalid custom node URL was accepted: {invalid}")


def test_custom_node_queue_is_strictly_sequential(monkeypatch) -> None:
    controller = launcher_app.CustomNodeController()
    active = 0
    maximum_active = 0
    order: list[str] = []

    async def fake_install(item) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(f"start:{item.name}")
        await asyncio.sleep(0.03)
        controller.update(item, status="complete", percent=100)
        order.append(f"end:{item.name}")
        active -= 1

    monkeypatch.setattr(controller, "_run_item", fake_install)

    async def run_installs() -> None:
        await controller.enqueue("https://github.com/example/Node-One")
        await controller.enqueue("https://github.com/example/Node-Two")
        await controller.enqueue("https://github.com/example/Node-Three")
        if controller.worker_task:
            await controller.worker_task

    asyncio.run(run_installs())

    assert maximum_active == 1
    assert order == [
        "start:Node-One",
        "end:Node-One",
        "start:Node-Two",
        "end:Node-Two",
        "start:Node-Three",
        "end:Node-Three",
    ]


def test_custom_node_clone_requirements_and_existing_detection(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "Example-ComfyUI-Node"
    source.mkdir()
    (source / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    (source / "requirements.txt").write_text("# no extra packages\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=dsnn Test",
            "-c",
            "user.email=test@dsnn.invalid",
            "commit",
            "-m",
            "Initial node",
        ],
        check=True,
        capture_output=True,
    )

    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes_dir = comfy_dir / "custom_nodes"
    comfy_dir.mkdir()
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes_dir)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")
    monkeypatch.setattr(launcher_app, "validate_custom_node_url", lambda url: url)

    controller = launcher_app.CustomNodeController()
    source_url = source.resolve().as_uri()

    async def install_twice() -> tuple[launcher_app.CustomNodeState, launcher_app.CustomNodeState]:
        first = await controller.enqueue(source_url)
        if controller.worker_task:
            await controller.worker_task
        second = await controller.enqueue(source_url)
        if controller.worker_task:
            await controller.worker_task
        return controller.items[first["id"]], controller.items[second["id"]]

    first_state, second_state = asyncio.run(install_twice())
    destination = custom_nodes_dir / "Example-ComfyUI-Node"

    assert first_state.status == "complete"
    assert first_state.restart_required is True
    assert second_state.status == "skipped"
    assert (destination / "__init__.py").exists()
    assert not list(custom_nodes_dir.glob(".dsnn-*.part"))


def test_workflow_fetches_a_missing_pinned_custom_node_commit(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    custom_nodes_dir = comfy_dir / "custom_nodes"
    destination = custom_nodes_dir / "ComfyUI-KJNodes"
    destination.mkdir(parents=True)
    (destination / "requirements.txt").write_text(
        "nvidia-vfx\n", encoding="utf-8"
    )
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes_dir)

    controller = launcher_app.JobController()
    commands: list[tuple[str, ...]] = []
    repo = "https://github.com/kijai/ComfyUI-KJNodes.git"
    ref = "1289b52fbb6d64a339a4047b9ea74cf7758ccf1e"

    async def fake_process(*command) -> tuple[int, str]:
        normalized = tuple(str(part) for part in command)
        commands.append(normalized)
        if "remote" in normalized:
            return 0, repo + "\n"
        if "cat-file" in normalized:
            return 1, "missing"
        return 0, ""

    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(
        controller._install_custom_node(
            {
                "name": "ComfyUI-KJNodes",
                "repo": repo,
                "ref": ref,
                "requirements_extra_index_url": "https://pypi.nvidia.com/",
                "install_requirements": True,
            }
        )
    )

    assert any("fetch" in command and ref in command for command in commands)
    assert any("checkout" in command and ref in command for command in commands)
    pip_command = next(command for command in commands if "pip" in command)
    assert pip_command[pip_command.index("--extra-index-url") + 1] == (
        "https://pypi.nvidia.com/"
    )
    assert controller.state.restart_required is True


def test_workflow_skips_failed_custom_node_and_continues(
    tmp_path,
    monkeypatch,
) -> None:
    custom_nodes_dir = tmp_path / "ComfyUI" / "custom_nodes"
    monkeypatch.setattr(launcher_app, "CUSTOM_NODES_DIR", custom_nodes_dir)
    controller = launcher_app.JobController()
    attempted: list[str] = []

    async def fake_install(node) -> None:
        attempted.append(node["name"])
        if node["name"] == "Broken-Node":
            raise RuntimeError("simulated node failure")

    monkeypatch.setattr(controller, "_install_custom_node", fake_install)

    asyncio.run(
        controller._install_custom_nodes(
            [
                {"name": "Broken-Node"},
                {"name": "Working-Node"},
            ]
        )
    )

    assert attempted == ["Broken-Node", "Working-Node"]
    assert controller.state.percent == 99
    assert controller.state.warnings == [
        "Broken-Node: simulated node failure",
    ]


def test_workflow_skips_failed_model_and_finishes_with_warning(monkeypatch) -> None:
    controller = launcher_app.JobController()
    attempted: list[str] = []

    class FakeRestartService:
        async def start(self) -> dict:
            return {"status": "restarting"}

        async def wait(self) -> dict:
            return {"status": "ready"}

    async def ready() -> None:
        return None

    async def fake_download(
        _client,
        file_spec,
        _index,
        _file_count,
        _completed_bytes,
        _known_total,
        _download_ceiling,
    ) -> int:
        attempted.append(file_spec["name"])
        if file_spec["name"] == "Broken model":
            raise RuntimeError("simulated download failure")
        return 10

    monkeypatch.setattr(controller, "_wait_for_comfyui", ready)
    monkeypatch.setattr(controller, "_download_file", fake_download)
    monkeypatch.setattr(
        launcher_app,
        "comfy_service_controller",
        FakeRestartService(),
    )

    asyncio.run(
        controller._run(
            {
                "id": "continue-test",
                "title": "Continue Test",
                "files": [
                    {
                        "name": "Broken model",
                        "destination": "models/checkpoints/broken.safetensors",
                        "size_bytes": 10,
                    },
                    {
                        "name": "Working model",
                        "destination": "models/checkpoints/working.safetensors",
                        "size_bytes": 10,
                    },
                ],
                "custom_nodes": [],
            }
        )
    )

    assert attempted == ["Broken model", "Working model"]
    assert controller.state.status == "complete"
    assert controller.state.percent == 100
    assert controller.state.warnings == [
        "Broken model: simulated download failure",
    ]
    assert "1 skipped item" in controller.state.message


def test_comfyui_manager_restart_waits_until_comfyui_is_ready(monkeypatch) -> None:
    states = iter([503, 200])
    marked_ready: list[bool] = []

    class Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str):
            if url.endswith("/manager/version"):
                return Response(200)
            return Response(next(states))

        async def post(self, _url: str, **_kwargs):
            raise launcher_app.httpx.RemoteProtocolError("expected reboot disconnect")

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(launcher_app.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(launcher_app.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        launcher_app,
        "mark_comfy_restart_complete",
        lambda: marked_ready.append(True),
    )
    service = launcher_app.ComfyServiceController()

    async def restart() -> None:
        await service.start()
        await service.wait()

    asyncio.run(restart())

    assert service.state.status == "ready"
    assert service.state.error is None
    assert marked_ready == [True]


def test_every_real_workflow_automatically_restarts_comfyui(
    monkeypatch,
) -> None:
    controller = launcher_app.JobController()
    calls: list[str] = []

    class FakeRestartService:
        async def start(self) -> dict:
            calls.append("start")
            return {"status": "restarting"}

        async def wait(self) -> dict:
            calls.append("wait")
            return {"status": "ready"}

    async def fake_install(_workflow) -> None:
        return None

    monkeypatch.setattr(controller, "_install_workflow", fake_install)
    monkeypatch.setattr(
        launcher_app,
        "comfy_service_controller",
        FakeRestartService(),
    )

    asyncio.run(
        controller._run(
            {
                "id": "restart-test",
                "title": "Restart Test",
                "files": [],
                "custom_nodes": [],
            }
        )
    )

    assert calls == ["start", "wait"]
    assert controller.state.status == "complete"
    assert controller.state.restart_required is False
    assert controller.state.comfy_restarted is True


def test_demo_workflow_does_not_restart_comfyui(monkeypatch) -> None:
    controller = launcher_app.JobController()
    calls: list[str] = []

    class FakeRestartService:
        async def start(self) -> dict:
            calls.append("start")
            return {"status": "restarting"}

        async def wait(self) -> dict:
            calls.append("wait")
            return {"status": "ready"}

    async def fake_demo(_workflow) -> None:
        return None

    monkeypatch.setattr(controller, "_run_demo", fake_demo)
    monkeypatch.setattr(
        launcher_app,
        "comfy_service_controller",
        FakeRestartService(),
    )

    asyncio.run(
        controller._run(
            {
                "id": "foundation-test",
                "title": "Foundation Test",
                "demo": True,
            }
        )
    )

    assert calls == []
    assert controller.state.status == "complete"
    assert controller.state.comfy_restarted is False


def test_comfyui_update_uses_official_master_and_runtime_python(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / ".git").mkdir(parents=True)
    (comfy_dir / "requirements.txt").write_text("", encoding="utf-8")
    comfy_python = comfy_dir / ".venv-cu128" / "bin" / "python"
    comfy_python.parent.mkdir(parents=True)
    comfy_python.touch()

    controller = launcher_app.JobController()
    commands: list[tuple[str, ...]] = []

    async def fake_process(*command) -> tuple[int, str]:
        commands.append(tuple(str(part) for part in command))
        return 0, "ok"

    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(launcher_app, "COMFYUI_VENV", comfy_dir / ".venv-cu128")
    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(controller._update_comfyui())

    assert commands[:6] == [
        (
            "git",
            "-C",
            str(comfy_dir),
            "rev-parse",
            "HEAD",
        ),
        (
            "git",
            "ls-remote",
            launcher_app.COMFYUI_UPSTREAM,
            "refs/heads/master",
        ),
        (
            "git",
            "-C",
            str(comfy_dir),
            "remote",
            "set-url",
            "origin",
            "https://github.com/Comfy-Org/ComfyUI.git",
        ),
        (
            "git",
            "-C",
            str(comfy_dir),
            "fetch",
            "--prune",
            "origin",
            "master",
        ),
        (
            "git",
            "-C",
            str(comfy_dir),
            "reset",
            "--hard",
            "origin/master",
        ),
        (
            "git",
            "-C",
            str(comfy_dir),
            "rev-parse",
            "HEAD",
        ),
    ]
    assert commands[6:] == [
        (
            str(comfy_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--timeout",
            "30",
            "--retries",
            "3",
            "-r",
            str(comfy_dir / "requirements.txt"),
        ),
    ]


def test_model_link_places_a_custom_node_checkpoint_without_a_second_download(
    tmp_path,
    monkeypatch,
) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    source = comfy_dir / "models" / "frame_interpolation" / "rife49.pth"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"rife-model")
    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    controller = launcher_app.JobController()

    asyncio.run(
        controller._apply_model_links(
            [
                {
                    "source": "models/frame_interpolation/rife49.pth",
                    "destination": (
                        "custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth"
                    ),
                }
            ]
        )
    )

    destination = (
        comfy_dir
        / "custom_nodes"
        / "ComfyUI-Frame-Interpolation"
        / "ckpts"
        / "rife"
        / "rife49.pth"
    )
    assert destination.read_bytes() == b"rife-model"


def test_comfyui_update_skips_when_the_installed_commit_is_current(tmp_path, monkeypatch) -> None:
    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / ".git").mkdir(parents=True)
    current = "a" * 40
    controller = launcher_app.JobController()
    commands: list[tuple[str, ...]] = []

    async def fake_process(*command) -> tuple[int, str]:
        commands.append(tuple(str(part) for part in command))
        if "ls-remote" in command:
            return 0, f"{current}\trefs/heads/master\n"
        return 0, current + "\n"

    monkeypatch.setattr(launcher_app, "COMFYUI_DIR", comfy_dir)
    monkeypatch.setattr(controller, "_run_process", fake_process)

    asyncio.run(controller._update_comfyui())

    assert len(commands) == 2
    assert controller.state.message == "ComfyUI is already current — update skipped."
