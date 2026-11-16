#!/usr/bin/env python3
"""Linux CPU-runner validation: real CUDA wheels/core/node installs, no weights.

This checks software compatibility, not GPU kernels or generation quality.
The production driver/GPU gates are separately covered by regression tests.
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from launcher import app
from launcher.runtime_profiles import RuntimeController,PROFILE,VERSIONS
from launcher.node_setup import node_registration_checks


async def main():
    with tempfile.TemporaryDirectory(prefix='dsnn-runtime-validation-') as temporary:
        folder=Path(temporary);native=folder/'native';native.mkdir()
        runtime=RuntimeController(native,Path(sys.prefix),'http://127.0.0.1:18189',feature=ROOT/'docker/runtime/bridge.py')
        job=app.JobController();job.isolated=True;app.runtime_controller=runtime
        async def run(*args,**kwargs):
            if args[0]=='nvidia-smi':return 0,'580.0.0\n' # software-only runner, never production
            return await job._run_process(*args,**kwargs)
        await runtime.prepare(run,lambda message:print(message,flush=True),lambda:None,require_gpu=False)
        for workflow in app.load_catalog()['workflows']:
            if workflow.get('runtime_profile')!=PROFILE:continue
            print('Validating node installation:',workflow['id'],flush=True)
            for node in workflow['custom_nodes']:
                print('Installing',node['name'],node['ref'],flush=True)
                await job._install_custom_node(node)
                if node['name']=='ComfyUI-Impact-Pack':
                    code,result=await run('node',ROOT/'tests/impact_frontend_regression.cjs',runtime.nodes/node['name']/'js/impact-pack.js')
                    if code:raise RuntimeError(result)
                    print(result,flush=True)
            log=folder/(workflow['id']+'.log')
            with log.open('wb') as output:
                process=await asyncio.create_subprocess_exec(str(runtime.python),str(runtime.core/'main.py'),
                    '--cpu','--listen','127.0.0.1','--port','18189','--disable-auto-launch',
                    cwd=runtime.core,env=runtime.environment(),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                async with httpx.AsyncClient(timeout=10,trust_env=False) as client:
                    for attempt in range(180):
                        if process.returncode is not None:raise RuntimeError('CPU ComfyUI exited:\n'+log.read_text()[-20000:])
                        try:
                            response=await client.get('http://127.0.0.1:18189/object_info')
                            if response.status_code==200:break
                        except httpx.HTTPError:pass
                        await asyncio.sleep(1)
                    else:raise RuntimeError('CPU ComfyUI readiness timed out:\n'+log.read_text()[-20000:])
                checks=node_registration_checks(workflow['custom_nodes'],response.json())
                # These packages require a real NVIDIA device even at import.
                gpu_only={'ComfyUI_Nvidia_VFX_Nodes'}
                failed=[check for check in checks if check['status']!='verified' and check['name'] not in gpu_only]
                print(json.dumps({'workflow':workflow['id'],'checks':checks,'gpu_validation_pending':sorted(gpu_only)},indent=2),flush=True)
                if failed:raise RuntimeError('Node registration failed:\n'+log.read_text()[-24000:])
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:await asyncio.wait_for(process.wait(),20)
                    except asyncio.TimeoutError:process.kill();await process.wait()
        code,result=await runtime.isolated_run(run)(runtime.python,'-c',"import json,torch,torchvision,torchaudio;print(json.dumps({n:__import__(n).__version__ for n in ('torch','torchvision','torchaudio')}))")
        if code or json.loads(result.strip().splitlines()[-1])!=VERSIONS:raise RuntimeError('A node changed the protected CUDA wheels: '+result)
        print('Isolated CUDA 13 software validation passed. Real-GPU validation remains required.',flush=True)


if __name__=='__main__':asyncio.run(main())
