"""Provider launch hook: local wheel only, kernel check, then ComfyUI in this process."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path('/opt/dsnn-attention')

def ensure_attention():
    import torch
    build = json.loads((ROOT/'build.json').read_text())
    actual = dict(python=list(sys.version_info[:2]), torch=torch.__version__,
                  cuda=torch.version.cuda, cxx11abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    if any(actual[k] != build[k] for k in actual):
        raise RuntimeError('SageAttention wheel does not match ComfyUI Python/Torch/CUDA ABI; use a fresh image environment')
    try:
        version = importlib.metadata.version('sageattention')
    except importlib.metadata.PackageNotFoundError:
        version = ''
    if version != '2.2.0':
        wheel = ROOT/'wheels'/build['wheel']
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != build['sha256']:
            raise RuntimeError('Baked SageAttention wheel checksum mismatch')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--force-reinstall', str(wheel)], check=True, timeout=120)
    for ext in ('_fused', '_qattn_sm80', '_qattn_sm89', '_qattn_sm90'):
        importlib.import_module('sageattention.'+ext)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is not accessible to CUDA; SageAttention and ComfyUI cannot start. Check the host GPU passthrough.')
    from sageattention import sageattn
    # Run before exposing ComfyUI; prime the JIT and verify both precision paths.
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            for width in (64, 128):
                tensors = [torch.randn(1, 4, 256, width, device='cuda', dtype=dtype) for _ in range(3)]
                result = sageattn(*tensors)
                expected = torch.nn.functional.scaled_dot_product_attention(*tensors)
                torch.cuda.synchronize()
                error = (result.float()-expected.float()).square().mean().sqrt()
                scale = expected.float().square().mean().sqrt().clamp_min(1e-6)
                if not torch.isfinite(result).all() or (error/scale).item() > .15:
                    raise RuntimeError('SageAttention GPU kernel numerical self-test failed')
    print('DSNN SageAttention 2.2.0: GPU kernel checks passed; enabled before ComfyUI startup.', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-directory', required=True)
    opts, args = p.parse_known_args()
    ensure_attention()
    root = Path(opts.base_directory).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    args = [a for a in args if a not in ('--use-pytorch-cross-attention', '--use-flash-attention', '--use-sage-attention')]
    sys.argv = [str(root/'main.py'), '--use-sage-attention', *args]
    runpy.run_path(str(root/'main.py'), run_name='__main__')

if __name__ == '__main__':
    main()
