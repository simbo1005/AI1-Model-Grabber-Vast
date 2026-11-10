"""Small, disposable ComfyUI jobs. Never load or modify the private source graph."""
from __future__ import annotations
import asyncio
import copy
import time
from pathlib import Path
from uuid import uuid4
import httpx
from launcher.krast_setup import placeholder_png


def node(kind, **inputs):
    return {'class_type': kind, 'inputs': inputs}


def jobs(image_name="dsnn-warmup/neutral.png [temp]"):
    image = {'1': node('LoadImage', image=image_name),
             '2': node('ImageScale', image=['1', 0], upscale_method='bilinear',
                       width=512, height=512, crop='disabled')}
    helpers = dict(image)
    helpers.update({
        '3': node('DownloadAndLoadFlorence2Model', model='microsoft/Florence-2-base', precision='fp16', convert_to_safetensors=False),
        '4': node('Florence2Run', image=['2', 0], florence2_model=['3', 0], text_input='', task='caption', fill_mask=False, max_new_tokens=16, num_beams=1, do_sample=False, keep_model_loaded=False, seed=1),
        '5': node('PreviewImage', images=['4', 0]),
        '6': node('DownloadAndLoadSAM2Model', model='sam2_hiera_small.safetensors', segmentor='single_image', device='cuda', precision='fp16'),
        '7': node('Sam2Segmentation', sam2_model=['6', 0], image=['2', 0], keep_model_loaded=False, coordinates_positive='[{"x":256,"y":256}]'),
        '8': node('MaskToImage', mask=['7', 0]),
        '9': node('PreviewImage', images=['8', 0]),
        '10': node('AIO_Preprocessor', image=['2', 0], preprocessor='DepthAnythingPreprocessor', resolution=512),
        '11': node('PreviewImage', images=['10', 0]),
    })
    seed = dict(image)
    seed.update({
        '3': node('SeedVR2LoadDiTModel', model='seedvr2_ema_3b_fp16.safetensors', device='cuda:0', blocks_to_swap=0, swap_io_components=False, offload_device='cpu', cache_model=False, attention_mode='sdpa'),
        '4': node('SeedVR2LoadVAEModel', model='ema_vae_fp16.safetensors', device='cuda:0', encode_tiled=False, decode_tiled=False, tile_debug='false', offload_device='none', cache_model=False),
        '5': node('SeedVR2VideoUpscaler', image=['2', 0], dit=['3', 0], vae=['4', 0], seed=1, resolution=512, max_resolution=512, batch_size=1, uniform_batch_size=False, color_correction='lab', temporal_overlap=0, prepend_frames=0, input_noise_scale=0, latent_noise_scale=0, offload_device='cpu', enable_debug=False),
        '6': node('PreviewImage', images=['5', 0]),
    })
    qwen = dict(image)
    qwen.update({
        '3': node('UNETLoader', unet_name='redgptK2T2W6A8REDQW21_redqw21UNLOCKEDV2_fp8.safetensors', weight_dtype='default'),
        '4': node('CLIPLoader', clip_name='qwen3vl_8b_bf16.safetensors', type='qwen_image', device='default'),
        '5': node('VAELoader', vae_name='qwen_image_2.1_vae_bf16.safetensors'),
        '6': node('TextEncodeQwenImage21', clip=['4', 0], vae=['5', 0], prompt='A neutral grey studio test image.', negative_prompt='', resolution=512, **{'images.image_1': ['2', 0]}),
        '7': node('ModelSamplingAuraFlow', model=['3', 0], shift=1.31, sampling='flow'),
        '8': node('ModelPatchLoader', name='Qwen-Image-2.1-Fun-Controlnet-Union.safetensors'),
        '9': node('QwenImageDiffsynthControlnet', model=['7', 0], model_patch=['8', 0], vae=['5', 0], image=['2', 0], strength=.9, start_percent=0., end_percent=1.),
        '10': node('KSampler', model=['9', 0], positive=['6', 0], negative=['6', 1], latent_image=['6', 2], seed=1, steps=1, cfg=1., sampler_name='er_sde', scheduler='beta', denoise=1.),
        '11': node('VAEDecode', samples=['10', 0], vae=['5', 0]),
        '12': node('PreviewImage', images=['11', 0]),
    })
    krast = dict(image)
    krast.update({
        '3': node('UNETLoader', unet_name='krastBf16_v3_bf16.safetensors', weight_dtype='default'),
        '4': node('CLIPLoader', clip_name='Krea2-Engineer-V1-bf16.safetensors', type='krea2', device='default'),
        '5': node('VAELoader', vae_name='wan_2.1_vae.safetensors'),
        '6': node('CLIPTextEncode', clip=['4', 0], text='A neutral grey studio test image.'),
        '7': node('CLIPTextEncode', clip=['4', 0], text=''),
        '8': node('VAEEncode', pixels=['2', 0], vae=['5', 0]),
        '9': node('ModelSamplingAuraFlow', model=['3', 0], shift=3., sampling='flow'),
        '10': node('KSampler', model=['9', 0], positive=['6', 0], negative=['7', 0], latent_image=['8', 0], seed=1, steps=1, cfg=1., sampler_name='euler', scheduler='simple', denoise=1.),
        '11': node('VAEDecode', samples=['10', 0], vae=['5', 0]),
        '12': node('PreviewImage', images=['11', 0]),
    })
    return [('Florence, SAM2 and depth', helpers), ('SeedVR2', seed),
            ('Qwen 2.1 and ControlNet', qwen), ('KRAST BF16', krast)]


def add_selected_loras(graph, selected, names):
    """Exercise selected LoRA patches as well as the base model kernels."""
    current = ["3", 0]
    for index, (filename, strength) in enumerate(names):
        if filename not in selected:
            continue
        key = str(100+index)
        graph[key] = node("LoraLoaderModelOnly", model=current, lora_name=filename, strength_model=strength)
        current = [key, 0]
    if "9" in graph and graph["9"]["class_type"] == "ModelSamplingAuraFlow":
        graph["9"]["inputs"]["model"] = current
    else:
        graph["7"]["inputs"]["model"] = current

LORAS = {
    "KRAST BF16": [("krea2_turbo_distill_r256.safetensors", .5),
        ("RealisticSnapshotKrea2V2.safetensors", 1.), ("nicegirls_krea2_000006000.safetensors", .35),
        ("r0mdl_000002750.safetensors", 1.), ("krea-smartphone-photo-slider.safetensors", 1.),
        ("KNP_000003000.safetensors", .85)],
    "Qwen 2.1 and ControlNet": [("civitai_lora_3357315_3244894.safetensors", 1.),
        ("lenovo_qwen21.safetensors", .7), ("bfs_head_v1.1_alternative_qwen_2.1.safetensors", 1.)],
}


def model_names(graph):
    fields = {'unet_name', 'clip_name', 'vae_name', 'name'}
    result = set()
    for n in graph.values():
        for key, value in n['inputs'].items():
            if key in fields or (key == 'model' and isinstance(value, str) and '/' not in value):
                if isinstance(value, str):
                    result.add(value)
    return result


async def upload_scratch(client):
    """A unique neutral temp image never reuses or overwrites a user's input."""
    name = "dsnn-warmup-"+uuid4().hex+".png"
    response = await client.post('/upload/image',
        files={'image': (name, placeholder_png(), 'image/png')},
        data={'type': 'temp', 'subfolder': 'dsnn-warmup', 'overwrite': 'false'})
    response.raise_for_status()
    data = response.json()
    if data.get('name') != name or data.get('type') != 'temp' or data.get('subfolder') != 'dsnn-warmup':
        raise RuntimeError('ComfyUI returned an unexpected warmup image location')
    return 'dsnn-warmup/'+name+' [temp]'


async def run_warmup(url, directory, workflow, cancelled, notify, warn, *, client=None, timeout=900):
    """Skip on a busy queue; cancel only our own job. A failure never claims readiness."""
    if workflow.get('warmup_profile') != 'krast-faceswap':
        return 'not-requested'
    own_client = client is None
    client = client or httpx.AsyncClient(base_url=url, timeout=30)
    outcomes = []
    scratch = None
    selected = {Path(f['destination']).name for f in workflow.get('files', [])}
    try:
        for name, graph in jobs():
            if name in LORAS:
                add_selected_loras(graph, selected, LORAS[name])
            cancelled()
            missing = model_names(graph) - selected
            if missing:
                warn('Warmup skipped for '+name+': required resources were not selected.')
                outcomes.append(False)
                continue
            queue = (await client.get('/queue')).raise_for_status().json()
            if queue.get('queue_running') or queue.get('queue_pending'):
                warn('Automatic warmup deferred because ComfyUI has queued work. Run installation again while idle to warm it.')
                return 'deferred'
            if scratch is None:
                scratch = await upload_scratch(client)
            graph['1']['inputs']['image'] = scratch
            notify('Warming '+name+' (scratch job; no saved workflow changes)…')
            response = await client.post('/prompt', json={'prompt': copy.deepcopy(graph), 'client_id': 'dsnn-warmup-'+uuid4().hex})
            if response.is_error or response.json().get('node_errors'):
                warn('Warmup validation failed for '+name+': '+str(response.json().get('error', response.json().get('node_errors', {})))[:800])
                outcomes.append(False)
                continue
            prompt_id = response.json()['prompt_id']
            deadline = time.monotonic()+timeout
            try:
                while True:
                    cancelled()
                    history = (await client.get('/history/'+prompt_id)).raise_for_status().json().get(prompt_id)
                    if history is not None:
                        state = history.get('status', {})
                        if state.get('status_str') == 'error':
                            messages = state.get('messages', [])
                            detail = next((x[1].get('exception_message', '') for x in messages if x[0]=='execution_error'), '')
                            raise RuntimeError(name+' warmup failed: '+detail[:600])
                        if not state.get('completed', False):
                            raise RuntimeError(name+' warmup did not complete')
                        outcomes.append(True)
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError(name+' warmup exceeded its time limit')
                    await asyncio.sleep(1)
            except (RuntimeError, TimeoutError) as exc:
                warn(str(exc))
                outcomes.append(False)
            finally:
                # Never interrupt another user's job or clear their queue/history.
                queue = (await client.get('/queue')).raise_for_status().json()
                if any(row[1] == prompt_id for row in queue.get('queue_pending', [])):
                    await client.post('/queue', json={'delete': [prompt_id]})
                if any(row[1] == prompt_id for row in queue.get('queue_running', [])):
                    await client.post('/interrupt', json={'prompt_id': prompt_id})
        return 'complete' if outcomes and all(outcomes) else 'partial'
    finally:
        if own_client:
            await client.aclose()
