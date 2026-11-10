"""Build one CUDA wheel for this image's exact Torch ABI, with an early SDK check."""
import hashlib
import json
import os
from pathlib import Path
import re
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
HEADERS = {'cusparse.h': 'libcusparse-dev', 'cublas_v2.h': 'libcublas-dev',
           'cublasLt.h': 'libcublas-dev', 'cusolverDn.h': 'libcusolver-dev'}


def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def fingerprint():
    import torch
    return dict(python=list(sys.version_info[:2]), torch=torch.__version__,
                cuda=torch.version.cuda, cxx11abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))


def parallel_jobs(memory):
    # nvcc uses a single thread; two translation units fit the 16 GB runners.
    return 2 if memory >= 14 * 1024**3 else 1


def memory_limit():
    memory = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    for file in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = Path(file).read_text().strip()
            if value.isdigit():
                memory = min(memory, int(value))
        except OSError:
            pass
    return memory


def missing_header_packages(roots):
    return {package for header, package in HEADERS.items()
            if not any((Path(root)/header).is_file() for root in roots)}


def prepare_headers(cuda, torch_root):
    roots = [cuda/'include', Path(sys.prefix)/'include']
    nvidia = torch_root.parent/'nvidia'
    roots += [p for p in nvidia.glob('*/include') if p.parent.name not in {'cuda_runtime', 'cuda_nvcc', 'cuda_cccl'}]
    roots = [p for p in roots if p.is_dir()]
    missing = missing_header_packages(roots)
    if missing:
        version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
        match = re.search(r'release (\d+)\.(\d+)', version)
        if not match:
            raise RuntimeError('Could not identify the CUDA toolkit for its development headers')
        suffix = '-'.join(match.groups())
        run('apt-get', 'update')
        for package in sorted(missing):
            # Extract only headers. Do not install/replace any runtime CUDA math
            # libraries that the provider's Torch already supplies.
            with tempfile.TemporaryDirectory(prefix='dsnn-cuda-headers-') as temporary:
                directory = Path(temporary)
                run('apt-get', 'download', package+'-'+suffix, cwd=directory)
                archives = list(directory.glob('*.deb'))
                if len(archives) != 1:
                    raise RuntimeError('Expected one CUDA development-header archive')
                extracted = directory/'extracted'
                run('dpkg-deb', '--extract', archives[0], extracted)
                expected = next(header for header, owner in HEADERS.items() if owner == package)
                sources = [p.parent for p in extracted.rglob(expected) if p.is_file()]
                if not sources:
                    raise RuntimeError('CUDA development package lacks '+expected)
                target = ROOT/'headers'/package
                shutil.copytree(sources[0], target, dirs_exist_ok=True)
                roots.append(target)
    if missing_header_packages(roots):
        raise RuntimeError('CUDA development headers are still incomplete')
    return roots


def preflight(cuda, torch_root, env):
    with tempfile.TemporaryDirectory(prefix='dsnn-cuda-preflight-') as temporary:
        source = Path(temporary)/'check.cu'
        source.write_text('#include <ATen/cuda/CUDAContext.h>\nvoid dsnn_cuda_header_check() {}\n')
        run(cuda/'bin/nvcc', '-std=c++17', '--expt-relaxed-constexpr', '-arch=sm_80',
            '-I'+str(torch_root/'include'), '-I'+str(torch_root/'include/torch/csrc/api/include'),
            '-c', source, '-o', Path(temporary)/'check.o', env=env)
    print('CUDA development-header compile check passed.', flush=True)


def patch_source(source):
    # Same Hopper-only target fix as Comfy-Org/wheels/patches/sageattention.py.
    old = """extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=['-lcuda'],"""
    if source.count(old) != 1 or source.count('\nsetup(\n') != 1:
        raise RuntimeError('SageAttention source changed: reviewed build patch no longer matches')
    new = 'extra_compile_args={"cxx": CXX_FLAGS, "nvcc": [f for f in NVCC_FLAGS if "gencode" not in f and "arch=" not in f] + ["-gencode", "arch=compute_90a,code=sm_90a"]},\n                libraries=["cuda"],'
    source = source.replace(old, new).replace('"--threads=8"', '"--threads=1"')
    source = source.replace('"-Xptxas=-v",', '')
    # The fused module includes Torch's complete CUDA context header. Compile it
    # first so future SDK/header problems fail before the long attention builds.
    return source.replace('\nsetup(\n', '\next_modules.sort(key=lambda ext: 0 if ext.name.endswith("._fused") else 1)\n\nsetup(\n')


def main():
    import torch
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
    ROOT.mkdir(exist_ok=True, parents=True)
    torch_root = Path(torch.__file__).resolve().parent
    roots = prepare_headers(cuda, torch_root)
    jobs = parallel_jobs(memory_limit())
    env = dict(os.environ, CUDA_HOME=str(cuda), TORCH_CUDA_ARCH_LIST=ARCHES,
               MAX_JOBS=str(jobs), EXT_PARALLEL='1', CXX_APPEND_FLAGS='-g0',
               NVCC_APPEND_FLAGS='--threads=1',
               CPATH=os.pathsep.join([*map(str, roots), *filter(None, os.environ.get('CPATH', '').split(os.pathsep))]),
               LIBRARY_PATH=str(cuda/'lib64/stubs')+':'+str(cuda/'targets/x86_64-linux/lib/stubs'))
    print('SageAttention build:', json.dumps(dict(**before, jobs=jobs, architectures=ARCHES)), flush=True)
    preflight(cuda, torch_root, env)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary)/'sage.tar.gz'
        urllib.request.urlretrieve('https://codeload.github.com/thu-ml/SageAttention/tar.gz/'+REF, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA:
            raise RuntimeError('SageAttention source checksum mismatch')
        with tarfile.open(archive) as tar:
            tar.extractall(temporary, filter='data')
        source = Path(temporary)/('SageAttention-'+REF)
        setup = source/'setup.py'
        setup.write_text(patch_source(setup.read_text()))
        run(sys.executable, '-m', 'pip', 'wheel', '--verbose', '--no-deps', '--no-build-isolation',
            '--no-cache-dir', '--wheel-dir', ROOT/'wheels', source, env=env)
    wheels = list((ROOT/'wheels').glob('sageattention-2.2.0-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Expected exactly one SageAttention 2.2.0 wheel')
    run(sys.executable, '-m', 'pip', 'install', '--break-system-packages', '--no-deps',
        '--force-reinstall', wheels[0])
    if fingerprint() != before:
        raise RuntimeError('SageAttention build changed the Torch runtime')
    (ROOT/'build.json').write_text(json.dumps(dict(**before, source=REF, architectures=ARCHES,
        toolkit=str(cuda.resolve()), wheel=wheels[0].name,
        sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest()), indent=2))
    run('rm', '-rf', '/var/lib/apt/lists')

if __name__ == '__main__':
    main()
