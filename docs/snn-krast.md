# SNN KRAST and FaceSwap / SeedVR2

The SNN KRAST tile installs the original image-generation stack and the resources
referenced by `FaceSwap_WIPtestedSeedVR.json` in your private `custom_wf` repository.
The uploaded workflow is never edited or redistributed by this installer.

Keep the default checkboxes selected: **41 resources, approximately 101.3 GB**.
These include full BF16 KRAST, Krea2 Engineer BF16, Qwen3-VL 8B BF16, the exact
RedQW FP8 model, Qwen 2.1 VAE and Fun ControlNet Union, the enabled workflow LoRAs,
Florence-2-base, SAM2 Small, Depth Anything Large, SeedVR2 3B FP16 and its FP16 VAE.
The original image-generation/vision resources remain available as well.
No model is replaced by a smaller quantization.

## Downloads and credentials

39 resources use the existing official `hf download` path, with Xet high
performance and the existing range settings. All HF URLs pin a repository commit
and all resources have size/SHA-256 checks. Depth's Transformers cache is seeded
from the verified CLI downloads so its first execution does not download again.
Florence's tokenizer/config files and SAM2 weights are installed ahead of time.

Set `HF_TOKEN` in your template for the private `simbo926/rmdl` LoRA. Set
`CIVITAI_TOKEN` for the existing RedQW model and companion Civitai LoRA. An exact
HF mirror of those two Civitai files has not been verified; the available HF
RedQW candidate has a different hash and precision and is not substituted.
Credentials are never written into source or image layers.

## Nodes and versions

The installer pins ComfyUI to `989d93d0d968e704c589ce0146bb18c77f6b50cc` for this
tile and verifies all **62 distinct node types** referenced by the uploaded graph.
It installs 15 pinned node packages including BatchFolderTools, Florence2, SAM2,
LayerStyle, Easy Use, controlnet_aux, Custom Scripts and SeedVR2, together with
rgthree, the original sampler, LC nodes and Bond Node Suite. ExifTool is provided
by `libimage-exiftool-perl`.

Node/core dependency installs constrain Torch, torchvision, torchaudio, torchcodec,
numpy, SageAttention and Triton to protect the native CUDA ABI. The original
sampler and caption selector are fetched from the reviewed, checksum-pinned
8.8 KB public image layer. LC Vision's reviewed installer uses its prebuilt wheel
without replacing Torch or compiling llama.cpp on your rented GPU.

## SageAttention 2 before startup

Use the new **Vast `1.4-test` / RunPod `2.4-test` image**, not just an auto-updated
launcher inside an older image. Docker builds SageAttention 2.2.0 from immutable
upstream source for SM 80, 86, 89, 90 and 120 (Ampere, Ada, Hopper and Blackwell).
The source archive is SHA-256 pinned; Hopper kernels use the reviewed
Hopper-only compilation target fix also used by Comfy-Org's wheel builds.

Before ComfyUI starts, its own Python checks the wheel's Python/Torch/CUDA/C++ ABI,
imports all CUDA extensions, and executes small FP16/BF16 GPU kernel checks against
PyTorch SDPA. A missing wheel in an existing venv is installed from the image,
with `--no-index --no-deps`; there is no runtime source compilation or download.
ComfyUI starts with `--use-sage-attention`. On cores that support it,
`--disable-cuda-graphs` prevents the stale-output replay path reported in
[SageAttention issue 393](https://github.com/thu-ml/SageAttention/issues/393).
A small DSNN prestartup guard reapplies this setting before compiler
initialization on Manager reboots too. Before warmup, setup checks
`/dsnn/attention` to confirm that the running ComfyUI really has SageAttention
2.2.0 enabled and CUDA graphs disabled. A missing/inactive guard fails readiness.
The workflow's explicit SeedVR2 SDPA
selection is preserved.

## Automatic warmup

After downloads, node installation, restart and registration checks, setup runs
four small scratch jobs: Florence/SAM2/depth, SeedVR2, Qwen 2.1/ControlNet and KRAST.
They use a neutral placeholder image, 512-pixel test inputs, one sampling step,
fixed test seeds and the selected workflow LoRAs. Their outputs go only to
ComfyUI's temporary preview directory. Your saved graph, prompts, photos,
resolution, model precision, sampler settings and output directory are unchanged.

Warmup waits for real execution completion. It skips a busy queue, reports failures,
and cancellation removes/interrupts only its own prompt ID. It never clears the
user's queue or history. `warmup_status` is available in `/api/status`; set
`LAUNCHER_WARMUP=0` to disable this optional preparation. Restarting ComfyUI loses
resident models, so installation must finish its warmup again after a restart.

Warmup shifts some initialization work into setup. A 32 GB GPU cannot retain all
of these models at once; later model offloading/reloading and shape-specific
compilation can still occur. CPU CI validates builds and controls; actual GPU
execution is checked on deployment and is not implied by a successful image build.

## Your workflow inputs

Get the original graph from your private
[custom_wf repository](https://github.com/simbo1005/custom_wf/blob/main/FaceSwap_WIPtestedSeedVR.json).
Supply its two reference photos and the `input/tasca` folder yourself; the
installer never fabricates replacements for those private inputs.

Disabled LoRA slots are not required for execution. Exact sources for
`MysticXXX_KREA2_v1.safetensors`, `NaturallySaggingBreasts.safetensors`,
`HORNY AMATEUR LORA.safetensors` and `Krea2_Realistic_Skin_Texture.safetensors`
have not been resolved and are not substituted or enabled.

The source graph has `SeedVR2LoadDiTModel` node 4559's `cache_model` set to the
string `"sdpa"`; its pinned node schema expects a boolean. ComfyUI's validator
coerces a nonempty string to `true`, so this enables model caching rather than
causing a validation rejection. At your request the original is kept unchanged.
The separate scratch warmup uses `false` for its own job to avoid retaining the
upscaler on the GPU before warming the diffusion models.


## Publishing launcher-only changes

The publishing workflow accepts an optional `runtime_base` containing a digest
from this image's own Docker Hub repository. This reuses the tested native
Torch/SageAttention layers and replaces only launcher/catalog files. A full
build remains the default for native dependency or startup-script changes.
The immutable base is validated before building; provider entrypoints, native
libraries and wheel fingerprints are inherited unchanged.
