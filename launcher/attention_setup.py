"""Keep SageAttention safety settings active across ComfyUI Manager restarts."""
from pathlib import Path
from uuid import uuid4
import httpx

PRESTART = '''# DSNN SageAttention startup guard v1
from comfy.cli_args import args
if getattr(args, "use_sage_attention", False) and hasattr(args, "disable_cuda_graphs"):
    args.disable_cuda_graphs = True
    print("DSNN: SageAttention CUDA graph replay disabled before compiler initialization.")
'''
INIT = '''# DSNN SageAttention startup guard v1
import importlib.metadata
from aiohttp import web
from comfy.cli_args import args
from server import PromptServer

@PromptServer.instance.routes.get("/dsnn/attention")
async def attention_status(request):
    try:
        version = importlib.metadata.version("sageattention")
    except importlib.metadata.PackageNotFoundError:
        version = "missing"
    return web.json_response({
        "sage_enabled": bool(getattr(args, "use_sage_attention", False)),
        "sage_version": version,
        "cuda_graphs_disabled": bool(getattr(args, "disable_cuda_graphs", False)),
    })

NODE_CLASS_MAPPINGS = {}
'''


def install_guard(root):
    root = Path(root).resolve()
    package = root/'DSNN_Attention_Guard'
    if package.is_symlink():
        raise RuntimeError('SageAttention guard directory may not be a symlink')
    package.mkdir(parents=True, exist_ok=True)
    for name, data in [('prestartup_script.py', PRESTART), ('__init__.py', INIT)]:
        path = package/name
        if path.is_symlink() or (path.exists() and path.read_text() != data):
            raise RuntimeError('An existing SageAttention guard file was modified; preserving it: '+name)
        if not path.exists():
            temporary = path.with_name(name+'.'+uuid4().hex+'.part')
            try:
                with temporary.open('x') as output:
                    output.write(data)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


async def verify_attention(url, *, client=None):
    owned = client is None
    client = client or httpx.AsyncClient(base_url=url, timeout=30)
    try:
        response = await client.get('/dsnn/attention')
        if response.status_code != 200:
            raise RuntimeError('ComfyUI did not load the SageAttention startup guard; restart ComfyUI and check its startup log.')
        state = response.json()
        if (state.get('sage_enabled') is not True
                or state.get('sage_version') != '2.2.0'
                or state.get('cuda_graphs_disabled') is not True):
            raise RuntimeError('ComfyUI is not running SageAttention 2.2.0 with CUDA graph replay disabled.')
    finally:
        if owned:
            await client.aclose()
