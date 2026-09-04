# syntax=docker/dockerfile:1.7

ARG VAST_COMFY_IMAGE=vastai/comfy:v0.34.0-cuda-12.9-py312
FROM ${VAST_COMFY_IMAGE}

ARG IMAGE_VERSION=dev

LABEL org.opencontainers.image.title="dsnn ComfyUI Workflow Launcher for Vast.ai" \
      org.opencontainers.image.description="Vast.ai ComfyUI with the dsnn Model Grabber and native Instance Portal integration" \
      org.opencontainers.image.source="https://github.com/simbo1005/AI1-Model-Grabber-Vast" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LAUNCHER_PORT=13000 \
    LAUNCHER_AUTO_UPDATE=1 \
    LAUNCHER_GITHUB_REPO=simbo1005/AI1-Model-Grabber-Vast \
    LAUNCHER_GITHUB_REF=main \
    COMFYUI_DIR=/workspace/ComfyUI \
    COMFYUI_VENV=/venv/main \
    COMFYUI_LOCAL_URL=http://127.0.0.1:18188 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_TOKEN_FILE=/opt/dsnn/secrets/hf_token \
    WORKSPACE=/workspace \
    DATA_DIRECTORY=/workspace \
    JUPYTER_DIR=/workspace \
    PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:3000:13000:/:Workflow Downloader|localhost:8188:18188:/:ComfyUI|localhost:8288:18288:/docs:API Wrapper|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing" \
    OPEN_BUTTON_PORT=1111 \
    OPEN_BUTTON_TOKEN=1

WORKDIR /opt/dsnn

RUN if ! command -v wget >/dev/null 2>&1; then \
      apt-get update \
      && apt-get install -y --no-install-recommends wget \
      && rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements-launcher.txt /tmp/requirements-launcher.txt
RUN uv pip install \
      --python /venv/main/bin/python \
      --no-cache \
      -r /tmp/requirements-launcher.txt \
    && rm /tmp/requirements-launcher.txt

# Warm dependencies shared by the pinned custom nodes without replacing the
# CUDA-enabled packages supplied by the Vast.ai base image.
COPY docker/custom-node-requirements.txt /tmp/custom-node-requirements.txt
RUN set -eu; \
    /venv/main/bin/python -m pip freeze \
      | grep -iE '^(torch|torchvision|torchaudio|torchcodec|numpy|transformers|pillow|opencv-[a-z-]+)==' \
      > /tmp/dsnn-constraints.txt; \
    uv pip install --python /venv/main/bin/python --no-cache \
      -c /tmp/dsnn-constraints.txt -r /tmp/custom-node-requirements.txt; \
    expected="$(sed -n 's/^[Tt]orch==//p' /tmp/dsnn-constraints.txt)"; \
    actual="$(/venv/main/bin/python -c 'import torch; print(torch.__version__)')"; \
    test -n "$expected"; \
    test "$actual" = "$expected"; \
    rm /tmp/custom-node-requirements.txt /tmp/dsnn-constraints.txt

# dlib is source-only and is required by ComfyUI_FaceAnalysis. Build it once
# instead of compiling it on a rented instance during workflow installation.
RUN set -eu; \
    added_toolchain=0; \
    if ! command -v c++ >/dev/null 2>&1; then \
      apt-get update; apt-get install -y --no-install-recommends build-essential; added_toolchain=1; \
    fi; \
    export MAKEFLAGS=-j4 CMAKE_BUILD_PARALLEL_LEVEL=4; \
    uv pip install --python /venv/main/bin/python --no-cache cmake 'dlib==20.0.1'; \
    /venv/main/bin/python -c 'import dlib; print(dlib.__version__)'; \
    uv pip uninstall --python /venv/main/bin/python cmake; \
    if [ "$added_toolchain" = 1 ]; then apt-get purge -y build-essential && apt-get autoremove -y; fi; \
    rm -rf /var/lib/apt/lists/*

COPY launcher/ /opt/dsnn/launcher/
COPY catalog/ /opt/dsnn/catalog/
COPY docker/vast/dsnn-launcher.sh /opt/supervisor-scripts/dsnn-launcher.sh
COPY docker/vast/dsnn-launcher.conf /etc/supervisor/conf.d/dsnn-launcher.conf

RUN chmod +x /opt/supervisor-scripts/dsnn-launcher.sh \
    && install -d -m 0700 /opt/dsnn/secrets

EXPOSE 1111 3000 8080 8188 8288 8384

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl --fail --silent http://127.0.0.1:13000/api/health || exit 1

# Keep the ENTRYPOINT and CMD inherited from vastai/comfy. Its boot process
# creates /workspace, starts the portal/Jupyter/ComfyUI, and launches this
# image's additional service through Supervisor.
