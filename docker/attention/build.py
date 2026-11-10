"""Build SageAttention once, against this image's exact Torch ABI (no GPU needed)."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path('/opt/dsnn-attention')
REF = 'd1a57a546c3d395b1ffcbeecc66d81db76f3b4b5'
SHA = '74b6667164f9e368e3799bc2ab59b9b08c4591630f1c6029560208b6fcf354c4'
ARCHES = '8.0;8.6;8.9;9.0;12.0'

def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)

def fingerprint():
    import torch
    return dict(python=list(sys.version_info[:2]), torch=torch.__version__,
                cuda=torch.version.cuda, cxx11abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))

def patch_source(source):
    # Same Hopper-only target fix as Comfy-Org/wheels/patches/sageattention.py.
    old = """extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=['-lcuda'],"""
    if source.count(old) != 1:
        raise RuntimeError('SageAttention source changed: Hopper target patch no longer matches')
    new = 'extra_compile_args={"cxx": CXX_FLAGS, "nvcc": [f for f in NVCC_FLAGS if "gencode" not in f and "arch=" not in f] + ["-gencode", "arch=compute_90a,code=sm_90a"]},\n                libraries=["cuda"],'
    return source.replace(old, new).replace('"--threads=8"', '"--threads=1"')

def main():
    before = fingerprint()
    if before['python'] != [3, 12] or not before['cuda']:
        raise RuntimeError('Expected CUDA Torch with Python 3.12')
    if not shutil.which('c++'):
        run('apt-get', 'update')
        run('apt-get', 'install', '-y', '--no-install-recommends', 'build-essential')
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    if not (cuda/'bin/nvcc').is_file():
        run('apt-get', 'update')
        run('apt-get', 'install', '-y', '--no-install-recommends', 'build-essential',
            'cuda-nvcc-12-8', 'cuda-cudart-dev-12-8', 'cuda-cccl-12-8', 'cuda-driver-dev-12-8')
        cuda = Path('/usr/local/cuda-12.8')
    run(sys.executable, '-m', 'pip', 'install', '--break-system-packages', '--no-cache-dir',
        'ninja==1.13.0', 'packaging', 'wheel', 'setuptools')
    env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST=ARCHES,
               MAX_JOBS='1', EXT_PARALLEL='1', NVCC_APPEND_FLAGS='--threads=1',
               LIBRARY_PATH=str(cuda/'lib64/stubs')+':'+str(cuda/'targets/x86_64-linux/lib/stubs'))
    ROOT.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp)/'sage.tar.gz'
        urllib.request.urlretrieve('https://codeload.github.com/thu-ml/SageAttention/tar.gz/'+REF, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA:
            raise RuntimeError('SageAttention source checksum mismatch')
        with tarfile.open(archive) as tar:
            tar.extractall(temp, filter='data')
        source = Path(temp)/('SageAttention-'+REF)
        setup = source/'setup.py'
        setup.write_text(patch_source(setup.read_text()))
        run(sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
            '--no-cache-dir', '--wheel-dir', ROOT/'wheels', source, env=env)
    wheels = list((ROOT/'wheels').glob('sageattention-2.2.0-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Expected exactly one SageAttention 2.2.0 wheel')
    run(sys.executable, '-m', 'pip', 'install', '--break-system-packages', '--no-deps',
        '--force-reinstall', wheels[0])
    # Resolve the driver's link stub for build-time import only. Never add
    # toolkit stubs to the runtime environment or image LD_LIBRARY_PATH.
    import ctypes
    for stub in (cuda/'lib64/stubs/libcuda.so', cuda/'targets/x86_64-linux/lib/stubs/libcuda.so'):
        if stub.is_file():
            ctypes.CDLL(str(stub), mode=ctypes.RTLD_GLOBAL)
            break
    for extension in ('_qattn_sm80', '_qattn_sm89', '_qattn_sm90', '_fused'):
        importlib.import_module('sageattention.'+extension)
    if fingerprint() != before:
        raise RuntimeError('SageAttention build changed the Torch runtime')
    (ROOT/'build.json').write_text(json.dumps(dict(**before, source=REF, architectures=ARCHES,
        wheel=wheels[0].name, sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest()), indent=2))
    run('rm', '-rf', '/var/lib/apt/lists')

if __name__ == '__main__':
    main()
