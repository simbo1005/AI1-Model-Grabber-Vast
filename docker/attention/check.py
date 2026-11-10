"""Separate import check: preserve the compiled wheel layer if a loader fix is needed."""
import ctypes
import importlib
import json
from pathlib import Path
from build import ROOT, fingerprint

build = json.loads((ROOT/'build.json').read_text())
actual = fingerprint()
if any(build[k] != actual[k] for k in actual):
    raise RuntimeError('Compiled SageAttention wheel ABI does not match Torch')
cuda = Path(build['toolkit'])
# A build-only link stub. Never export this path into the runtime environment.
for stub in (cuda/'lib64/stubs/libcuda.so', cuda/'targets/x86_64-linux/lib/stubs/libcuda.so'):
    if stub.is_file():
        ctypes.CDLL(str(stub), mode=ctypes.RTLD_GLOBAL)
        break
for extension in ('_fused', '_qattn_sm80', '_qattn_sm89', '_qattn_sm90'):
    importlib.import_module('sageattention.'+extension)
core = importlib.import_module('sageattention.core')
if not all(getattr(core, flag, False) for flag in ('SM80_ENABLED','SM89_ENABLED','SM90_ENABLED')):
    raise RuntimeError('SageAttention Python dispatch could not import all compiled backends')
print('SageAttention 2.2.0: all compiled modules imported and Torch ABI preserved.', flush=True)
