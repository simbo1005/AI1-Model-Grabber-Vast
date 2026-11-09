# SNN KRAST resource installer

Select **SNN KRAST** in the launcher and keep the default resource checkboxes
selected to install the original full-precision workflow stack. The tile is
separate from Krea 2 Extended and does not add KRAST or its LoRAs to WAN/MiniMax.

The 15 checked-by-default resources total **48,427,687,495 bytes (48.4 GB)**:

- KRAST Raw BF16 v3 from the public, ungated dataset `simbo926/KrastRaw`.
- Original Qwen3-VL 4B BF16 text encoder and Wan 2.1 VAE.
- Four original LoRAs: turbo distill r256, TextFusion, Lenovo UltraReal and
  RealisticSnapshot Krea2 V2. The latter uses a SHA-256-identical HF mirror.
- Qwen3-VL 8B Q8_0 vision model and its original F16 projector.
- Segformer and ViTMatte models, each with its config and preprocessor config.

All model URLs are pinned Hugging Face revisions. They use the existing
`hf download` path, `HF_XET_HIGH_PERFORMANCE=1` and the launcher's existing
range-request settings. The new tile does not change the transfer method or
speed settings of existing pipelines. Actual throughput still depends on the
host's bandwidth, disk and route to HF.

The main model download is equivalent to:

```bash
hf download simbo926/KrastRaw Krast-Raw-bf16.safetensors \
  --repo-type dataset --revision a23a479cd6a0c22359b18a1cc6df5f1636617eb1 \
  --local-dir /tmp/krast-download
```

The launcher places it at
`models/diffusion_models/krastBf16_v3_bf16.safetensors`, preserving the filename
expected by the unchanged workflow. Its SHA-256 is
`3b2c89871d08bb13e8bf166edb9fd2e23c1349b9b48a0e1cad095f3f03a03335`.
No quantization or model replacement is performed. `HF_TOKEN` from your
template is used when present; these public resources do not require a token.
No Civitai token is needed for this tile.

## Custom nodes and readiness

The installer adds pinned rgthree, LC123, LC MaskMaker and LC Vision packages,
the original SamplerV3 and lazy prompt selector, plus Bond Node Suite and
ExifTool/FFmpeg. The two bundled node packages are fetched from a checksum-pinned
8.8 KB public image layer; this never pulls a complete Docker image, and their
source is not republished in this repository. Existing modified local copies
are preserved and reported instead of overwritten.

LC123 supplies the original `LC_Crushed_Blacks.cube` LUT. LC Vision runs its
reviewed installer using ComfyUI's Python to select and verify a prebuilt,
vision-capable llama.cpp CUDA wheel. It uses `--no-deps`, never builds from source,
and does not replace Torch or numpy. Small supporting dependencies are installed
under constraints protecting the existing CUDA packages.

After restart the launcher checks every workflow-required custom-node type,
the 16 required core node types, Krea 2 encoder support and the LUT. The graph's
neutral `studio_placeholder.png` input is created through ComfyUI's upload API
only if it is absent, so customized input directories work and existing images
are preserved. Select your own reference before using image-caption mode.

## Open the workflow

This tile installs resources; it does not redistribute product workflow files.
Download the editable graph from your private repository:

[custom_wf / SNN-KRAST](https://github.com/simbo1005/custom_wf/tree/main/SNN-KRAST)

Load `SNN-KRAST-Editable-ComfyUI.json` into ComfyUI, enter your description in
**PROMPT SOURCE**, and queue it. All node links and sampling values in that
file remain unchanged. No replacement sampler is used.

For an existing container, restart the launcher/container to fetch the latest
`main` source through its configured GitHub updater. A template pinned to another
source ref will need the corresponding ref selected. These changes do not
require baking weights or credentials into an image.

Validation covers resource hashes, dataset-to-model placement, the HF CLI route,
safe atomic bundle installation, the vision installer and readiness checks.
GPU generation still requires testing on a real instance.
