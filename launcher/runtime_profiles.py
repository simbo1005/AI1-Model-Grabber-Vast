"""Keep new upstream CUDA 13 workflows separate from the native Sage runtime."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from uuid import uuid4
from contextlib import contextmanager

import httpx
from launcher import ngx_setup
from launcher.node_setup import install_requirements

PROFILE='minimax-h3-cu130'
CORE_REF='e9afbccc2669677076feb08e52658f129cd8c12b'
CORE_REPO='https://github.com/Comfy-Org/ComfyUI.git'
PORT=18189
PACKAGES=('torch==2.10.0+cu130','torchvision==0.25.0+cu130','torchaudio==2.10.0+cu130')
VERSIONS={'torch':'2.10.0+cu130','torchvision':'0.25.0+cu130','torchaudio':'2.10.0+cu130'}
WORKER_INIT='''# DSNN isolated runtime status v1
import os,sys,torch
from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get('/dsnn/runtime-worker')
async def status(request):
    return web.json_response({'profile':'minimax-h3-cu130','nonce':os.environ.get('DSNN_WORKER_NONCE'),
        'pid':os.getpid(),'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.is_available()})
NODE_CLASS_MAPPINGS={}
'''


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise RuntimeError('Preserving a symlinked runtime configuration')
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as output:
            json.dump(data,output);output.flush();os.fsync(output.fileno())
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


class RuntimeController:
    def __init__(self,root,base_venv,url,*,feature=Path('/opt/dsnn-attention/runtime_bridge.py')):
        self.root=Path(root).resolve();self.base_venv=Path(base_venv).resolve();self.url=url.rstrip('/')
        self.feature=feature;self.folder=self.root.parent/'.dsnn-runtimes'/PROFILE
        self.core=self.folder/'ComfyUI';self.venv=self.folder/'venv'
        self.python=self.venv/'bin/python';self.nodes=self.core/'custom_nodes'
        self.marker=self.root/'.dsnn/runtime.json';self.pidfile=self.folder/'worker.json'
        self.report={'profile':None,'phase':'native runtime','versions':{},'issues':[]}

    def selection(self):
        if not self.marker.exists():return {'active':None,'busy':False}
        data=json.loads(self.marker.read_text())
        if not isinstance(data,dict) or data.get('active') not in (None,PROFILE) or not isinstance(data.get('busy',False),bool):
            raise RuntimeError('Invalid DSNN runtime selection')
        return {'active':data.get('active'),'busy':data.get('busy',False)}

    def active(self):return self.selection()['active']==PROFILE

    def release(self):
        if self.marker.exists():atomic_json(self.marker,{**self.selection(),'busy':False})

    def environment(self):
        env=os.environ.copy()
        for name in ('PIP_CONSTRAINT','PIP_BUILD_CONSTRAINT','PYTHONPATH','PYTHONHOME'):
            env.pop(name,None)
        env['PYTHONNOUSERSITE']='1';env['VIRTUAL_ENV']=str(self.venv)
        env['PATH']=str(self.venv/'bin')+os.pathsep+env.get('PATH','')
        # NVIDIA's VFX 0.2 Linux SDK bundles NPP 12 libraries, while its
        # ComfyUI node requires Torch CUDA 13. Expose only the small companion
        # CUDA 12 runtime; never put the SDK's cuDNN shim ahead of Torch.
        companions=[str(path) for path in self.venv.glob('lib/python3.*/site-packages/nvidia/cuda_runtime/lib')
                    if path.is_dir() and path.resolve().is_relative_to(self.venv.resolve())]
        if companions:
            env['LD_LIBRARY_PATH']=os.pathsep.join(filter(None,[env.get('LD_LIBRARY_PATH',''),*companions]))
        return env

    def isolated_run(self,run):
        async def command(*args,**kwargs):
            kwargs.setdefault('env',self.environment())
            return await run(*args,**kwargs)
        return command

    @contextmanager
    def requirements(self, source):
        """Pin reviewed SAM2/VFX dependencies without editing node checkouts."""
        source=Path(source)
        unpinned='git+https://github.com/facebookresearch/sam2'
        reviewed={unpinned:unpinned+'@2b90b9f5ceec907a1c18123530e92e794ad901a4'}
        if source.parent.name=='ComfyUI_Nvidia_VFX_Nodes':
            reviewed['nvidia-vfx>=0.2']='nvidia-vfx==0.2.0.0\nnvidia-cuda-runtime-cu12==12.9.79'
        lines=source.read_text().splitlines()
        if not any(line.strip() in reviewed for line in lines):
            yield source;return
        with tempfile.TemporaryDirectory(prefix='dsnn-node-source-') as temporary:
            target=Path(temporary)/'requirements.txt'
            target.write_text('\n'.join(reviewed.get(line.strip(),line) for line in lines)+'\n')
            yield target

    async def idle(self):
        async with httpx.AsyncClient(timeout=10,trust_env=False) as client:
            response=await client.get(self.url+'/queue')
            if response.status_code==503 and self.active() and not self._pid():
                # A persisted selection can outlive a container/worker restart.
                # Only recover when the private port is also unoccupied; never
                # interrupt or adopt another process.
                try:
                    reader,writer=await asyncio.open_connection('127.0.0.1',PORT)
                except OSError:
                    atomic_json(self.marker,{'active':None,'busy':self.selection()['busy']})
                    response=await client.get(self.url+'/queue')
                else:
                    writer.close();await writer.wait_closed()
            response.raise_for_status();queue=response.json()
        if queue.get('queue_running') or queue.get('queue_pending'):
            raise RuntimeError('ComfyUI has active or queued work. Let it finish before switching runtimes; no jobs were interrupted.')

    async def prepare(self,run,notify,cancelled,*,require_gpu=True,require_ngx=False):
        if not self.feature.is_file():
            raise RuntimeError('This upstream workflow needs the new isolated-runtime image: Vast 1.5-test / RunPod 2.5-test. Existing pipelines still use the native image runtime.')
        self.report={'profile':PROFILE,'phase':'checking driver','versions':{},'issues':[]}
        cancelled()
        code,driver=await run('nvidia-smi','--query-gpu=driver_version','--format=csv,noheader')
        driver=next((x.strip() for x in driver.splitlines() if x.strip()),'')
        if code or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?',driver) or int(driver.split('.')[0])<580:
            raise RuntimeError('The isolated CUDA 13 workflow requires NVIDIA driver 580 or newer. The native Torch/Sage environment was left intact.')
        self.report['versions']['nvidia-driver']=driver
        if self.active():
            await self.idle();atomic_json(self.marker,{**self.selection(),'busy':True})
        if self.folder.is_symlink():raise RuntimeError('Preserving a symlinked runtime directory')
        owner=self.folder/'owner.json'
        if self.folder.exists() and not owner.is_file() and any(self.folder.iterdir()):
            raise RuntimeError('The isolated runtime directory already contains unrecognized files; preserving them.')
        self.folder.mkdir(parents=True,exist_ok=True)
        if owner.exists() and json.loads(owner.read_text()).get('profile')!=PROFILE:
            raise RuntimeError('Unexpected isolated runtime owner')
        atomic_json(owner,{'profile':PROFILE,'version':1})
        command=self.isolated_run(run)
        async def checked(*args,**kwargs):
            cancelled();code,output=await command(*args,**kwargs)
            if code:raise RuntimeError('Isolated runtime setup failed: '+output[-1200:])
            return output
        if not self.core.exists():
            notify('Preparing the isolated ComfyUI checkout…')
            with tempfile.TemporaryDirectory(prefix='dsnn-comfy-',dir=self.folder) as staging:
                checkout=Path(staging)/'ComfyUI'
                await checked('git','clone','--filter=blob:none','--no-checkout',CORE_REPO,checkout)
                await checked('git','-C',checkout,'fetch','--depth=1','origin',CORE_REF)
                await checked('git','-C',checkout,'checkout','--detach',CORE_REF)
                checkout.rename(self.core)
        else:
            head=(await checked('git','-C',self.core,'rev-parse','HEAD')).strip()
            if head!=CORE_REF:
                changes=await checked('git','-C',self.core,'status','--porcelain','--untracked-files=no')
                if changes.strip():raise RuntimeError('Preserving local edits in the isolated ComfyUI checkout.')
                await checked('git','-C',self.core,'fetch','--depth=1','origin',CORE_REF)
                await checked('git','-C',self.core,'checkout','--detach',CORE_REF)
        models=self.core/'models';shared=self.root/'models';shared.mkdir(parents=True,exist_ok=True)
        if models.is_symlink():
            if models.resolve()!=shared.resolve():raise RuntimeError('The isolated models link points elsewhere; preserving it.')
        else:
            if models.exists():
                backup=self.core/'.dsnn-stock-models'
                if backup.exists():raise RuntimeError('Unexpected isolated models directory; preserving both copies.')
                models.rename(backup)
            models.symlink_to(shared,target_is_directory=True)
        if self.venv.is_symlink():raise RuntimeError('Preserving a symlinked Python environment')
        if not self.python.is_file():
            notify('Creating an isolated Python environment…')
            creator=self.base_venv/'bin/python'
            await checked(creator,'-m','venv',self.venv)
        cfg=self.venv/'pyvenv.cfg'
        if not cfg.is_file() or 'include-system-site-packages = true' in cfg.read_text().lower():
            raise RuntimeError('The isolated runtime may not inherit the native Torch/Sage packages.')
        probe="import importlib.metadata as m,json,sys; import torch,torchvision,torchaudio; print('DSNN_RUNTIME='+json.dumps({'versions':{n:m.version(n) for n in ('torch','torchvision','torchaudio')},'prefix':sys.prefix,'origins':{n:__import__(n).__file__ for n in ('torch','torchvision','torchaudio')},'cuda':torch.version.cuda,'gpu':torch.cuda.is_available()}))"
        async def inspect():
            code,output=await command(self.python,'-c',probe)
            for line in output.splitlines():
                if line.startswith('DSNN_RUNTIME='):
                    try:
                        data=json.loads(line.partition('=')[2])
                        good=(not code and data['versions']==VERSIONS and data['cuda']=='13.0'
                              and Path(data['prefix']).resolve()==self.venv.resolve()
                              and all(Path(p).resolve().is_relative_to(self.venv.resolve()) for p in data['origins'].values())
                              and (data['gpu'] or not require_gpu))
                        return good,data
                    except (ValueError,KeyError):pass
            return False,{}
        good,details=await inspect()
        for attempt in range(2):
            if good:break
            notify('Installing isolated PyTorch CUDA 13 wheels…' if not attempt else 'Repairing the isolated CUDA 13 wheels…')
            self.report['phase']='installing CUDA 13';self.report['attempt']=attempt+1
            args=[self.python,'-m','pip','install','--no-cache-dir','--timeout','60','--retries','3',*PACKAGES,'--index-url','https://download.pytorch.org/whl/cu130']
            if attempt:args+=['--force-reinstall']
            code,output=await command(*args)
            good,details=await inspect()
        if not good:raise RuntimeError('Isolated CUDA 13 imports/GPU access could not be verified. The native environment was not modified.')
        self.report['versions'].update(details['versions']);self.report['versions']['cuda']='13.0'
        # Python 3.12 venvs no longer include setuptools. These small build
        # tools are needed by reviewed node sdists (not CUDA kernel builds).
        await checked(self.python,'-m','pip','install','--disable-pip-version-check',
                      'setuptools<83','wheel','Cython>=3,<4')
        stamp=self.folder/'core-requirements.json'
        if not stamp.exists() or json.loads(stamp.read_text()).get('core')!=CORE_REF:
            notify('Installing the isolated ComfyUI dependencies…')
            code,output=await install_requirements(self.core/'requirements.txt',self.python,command,lambda text:'no module named' in text.lower())
            if code:raise RuntimeError('Isolated ComfyUI dependencies failed: '+output[-1200:])
            atomic_json(stamp,{'core':CORE_REF})
        if require_ngx:
            self.report['versions']['nvidia-ngx']=await ngx_setup.ensure(command,driver,notify)
        status=self.nodes/'DSNN_Runtime_Status';status.mkdir(parents=True,exist_ok=True)
        target=status/'__init__.py'
        if target.is_symlink() or (target.exists() and target.read_text()!=WORKER_INIT):
            raise RuntimeError('Preserving a modified isolated-runtime status package.')
        target.write_text(WORKER_INIT)
        for folder in ('input','output'):(self.root/folder).mkdir(parents=True,exist_ok=True)
        (self.folder/'user').mkdir(exist_ok=True)
        self.report['phase']='packages verified'
        atomic_json(self.folder/'ready.json',{'profile':PROFILE,'core':CORE_REF,'gpu_verified':require_gpu,'versions':details['versions']})

    def _pid(self):
        if not self.pidfile.is_file():return None
        try:
            data=json.loads(self.pidfile.read_text());pid=int(data['pid'])
            argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            start=Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19]
            if os.fsencode(self.core/'main.py') not in argv or start!=data['start']:return None
            return data
        except (ValueError,KeyError,OSError,IndexError):return None

    async def stop(self):
        data=self._pid()
        if data:
            try:os.killpg(data['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            for _ in range(50):
                if not self._pid():break
                await asyncio.sleep(.1)
            else:
                try:os.killpg(data['pid'],signal.SIGKILL)
                except ProcessLookupError:pass
        self.pidfile.unlink(missing_ok=True)

    async def _start(self):
        if not self.python.is_file() or not (self.folder/'ready.json').is_file():
            raise RuntimeError('The isolated runtime has not been prepared.')
        await self.stop()
        # Refuse to take over a port owned by another service.
        try:
            reader,writer=await asyncio.open_connection('127.0.0.1',PORT)
        except OSError:pass
        else:
            writer.close();await writer.wait_closed()
            raise RuntimeError('Loopback port 18189 is occupied by another service; it was left untouched.')
        nonce=uuid4().hex;env=self.environment();env['DSNN_WORKER_NONCE']=nonce
        log_path=self.folder/'comfyui.log'
        with log_path.open('ab') as log:
            process=await asyncio.create_subprocess_exec(str(self.python),str(self.core/'main.py'),
                '--listen','127.0.0.1','--port',str(PORT),'--disable-auto-launch',
                '--use-pytorch-cross-attention','--input-directory',str(self.root/'input'),
                '--output-directory',str(self.root/'output'),'--user-directory',str(self.folder/'user'),
                cwd=self.core,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            start=Path(f'/proc/{process.pid}/stat').read_text().rpartition(')')[2].split()[19]
            atomic_json(self.pidfile,{'pid':process.pid,'start':start,'nonce':nonce})
            async with httpx.AsyncClient(timeout=5,trust_env=False) as client:
                for _ in range(180):
                    if process.returncode is not None:raise RuntimeError('The isolated ComfyUI worker exited; inspect its runtime log.')
                    try:
                        response=await client.get(f'http://127.0.0.1:{PORT}/dsnn/runtime-worker')
                        info=response.json()
                        if (response.status_code==200 and info.get('nonce')==nonce and info.get('torch')==VERSIONS['torch']
                                and info.get('cuda')=='13.0' and info.get('gpu') is True):return
                    except (httpx.HTTPError,ValueError):pass
                    await asyncio.sleep(1)
            raise RuntimeError('The isolated ComfyUI worker did not become ready; inspect its runtime log.')
        except BaseException:
            await self.stop();raise

    async def activate(self):
        previous=self.selection();atomic_json(self.marker,{**previous,'busy':True})
        restarting=False
        try:
            await self.idle()
            async with httpx.AsyncClient(timeout=15,trust_env=False) as client:
                response=await client.get(self.url+'/dsnn/runtime');response.raise_for_status()
                if response.json().get('supports_isolation') is not True:raise RuntimeError('Native ComfyUI did not load the runtime bridge.')
                # The queue is reserved and idle; release resident native models.
                await client.post(self.url+'/free',json={'unload_models':True,'free_memory':True})
            restarting=True
            await self._start()
            atomic_json(self.marker,{'active':PROFILE,'busy':False})
            self.report['phase']='active'
        except BaseException:
            atomic_json(self.marker,{'active':None if restarting else previous['active'],'busy':False})
            self.report['phase']='activation failed; previous jobs preserved'
            raise

    async def deactivate(self):
        if not self.active():return
        atomic_json(self.marker,{'active':PROFILE,'busy':True})
        try:await self.idle();await self.stop()
        except BaseException:
            self.release();raise
        atomic_json(self.marker,{'active':None,'busy':False});self.report['phase']='native runtime'

    async def restart(self):await self.activate()
