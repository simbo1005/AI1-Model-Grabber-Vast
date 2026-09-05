# dsnn AI1 Model Grabber for Vast.ai

A Vast.ai-native version of the dsnn workflow launcher. It extends Vast.ai's
official ComfyUI image and integrates the launcher with the Instance Portal,
Supervisor, JupyterLab, ComfyUI, authentication, logs, and `/workspace` layout.

## Included services

- Vast.ai Instance Portal on external port `1111`.
- dsnn Workflow Downloader on external port `3000` (internal port `13000`).
- ComfyUI on external port `8188` (internal port `18188`).
- ComfyUI API Wrapper on external port `8288`.
- JupyterLab on external port `8080`.
- Syncthing on external port `8384`.
- Automatic detection of Vast's random public port mappings.
- Direct launcher buttons for both ComfyUI and JupyterLab.

The image keeps the fast Hugging Face CLI/Xet downloader, resumable downloads,
live progress, declarative workflow catalog, custom-model queue, custom-node
installer, dependency warming, and diagnostics from the original launcher.

## Vast.ai template

Use the following image:

```text
sdcioba/comfyui-workflow-launcher-vast:1.1
```

Select **Docker ENTRYPOINT**. Leave entrypoint arguments and the on-start script
empty; the image inherits Vast.ai's native boot process.

Docker options:

```text
-p 1111:1111 -p 3000:3000 -p 8080:8080 -p 8188:8188 -p 8288:8288 -p 8384:8384 -e SERVERLESS=false -e SUPERVISOR_SKIP_PYWORKER=true -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/workspace -e DATA_DIRECTORY=/workspace
```

Use at least 60 GB of disk for Krea 2. Use 100 GB for general use and at least
130 GB for the largest installer. No persistent volume is required.

After the instance starts, click **Open**. The Instance Portal provides links
to Workflow Downloader, ComfyUI, the API Wrapper, JupyterLab, and Syncthing.
The portal's default username is `vastai`; its password is the per-instance
Open Button token supplied by Vast.ai.

## Paths

```text
Workspace:       /workspace
ComfyUI:         /workspace/ComfyUI
Models:          /workspace/ComfyUI/models
Custom nodes:    /workspace/ComfyUI/custom_nodes
Python venv:     /venv/main
Launcher source: /opt/dsnn
```

Vast copies the baked `/opt/workspace-internal/ComfyUI` tree into
`/workspace/ComfyUI` on first boot. The launcher waits for the platform's
ComfyUI service on `127.0.0.1:18188`.

## Workflow catalog

The catalog contains six installers:

- Image Generation (approximately 19.9 GB)
- Krea 2 (approximately 18.4 GB)
- Dataset Generator (approximately 44.6 GB)
- Image Edit (approximately 17.8 GB)
- Motion Control (approximately 26.5 GB)
- MiniMax H3 (approximately 63.4 GB)

Each preset installs its models and custom nodes. An individual failure is
reported as a warning while remaining items continue. Product workflow JSON
files are deliberately not included.

## Credentials

For gated Hugging Face models, add `HF_TOKEN` to a **private** Vast.ai template.
For other authenticated downloads, use the corresponding runtime variables:

```text
HF_TOKEN=hf_...
CIVITAI_TOKEN=...
GITHUB_TOKEN=...
```

Do not commit credentials and do not bake a personal Hugging Face token into a
public image. Environment variables and files inside a rented container are
visible to whoever controls that instance.

## Automatic launcher updates

At startup the launcher checks:

```text
LAUNCHER_GITHUB_REPO=simbo1005/AI1-Model-Grabber-Vast
LAUNCHER_GITHUB_REF=main
LAUNCHER_AUTO_UPDATE=1
```

Updates to the Python launcher, static UI, or workflow catalog therefore reach
new instances without rebuilding the image. When GitHub is unavailable, the
version baked into the image is used.

## Service management

From the Jupyter terminal or SSH:

```bash
supervisorctl status
supervisorctl restart dsnn-launcher
supervisorctl restart comfyui
supervisorctl tail -f dsnn-launcher
supervisorctl tail -f comfyui
```

## Local tests

```bash
python -m pip install -r requirements-launcher.txt pytest
python -m pytest
python scripts/audit_node_requirements.py
```

## Publishing

The GitHub Actions workflow publishes both the requested tag and `latest` to:

```text
sdcioba/comfyui-workflow-launcher-vast
```

Add a GitHub Actions secret named `DOCKERHUB_TOKEN`, then run **Build and
publish Vast.ai Docker image** with tag `1.1`.

For a manual build:

```bash
docker build --platform linux/amd64 -t sdcioba/comfyui-workflow-launcher-vast:1.1 .
docker push sdcioba/comfyui-workflow-launcher-vast:1.1
```
