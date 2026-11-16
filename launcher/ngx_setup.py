"""Repair only the driver-matched NGX userspace library, never the GPU driver."""
from pathlib import Path
import os
import re
import tempfile

VERSION = re.compile(r'\d+\.\d+(?:\.\d+)?')


async def probe(run, expected, install_dir=Path('/usr/local/lib')):
    code,output=await run('ldconfig','-p')
    if code:return False
    for target in re.findall(r'^\s*libnvidia-ngx\.so(?:\.1)?\s+.*?=>\s+(\S+)\s*$',output,re.M):
        path=Path(target)
        try:
            resolved=path.resolve(strict=True)
            if resolved.name=='libnvidia-ngx.so.'+expected:return True
            candidate=install_dir/('libnvidia-ngx.so.'+expected)
            if candidate.is_file() and os.path.samefile(candidate,path):return True
        except OSError:pass
    return False


async def ensure(run, driver, notify, install_dir=Path('/usr/local/lib')):
    if not VERSION.fullmatch(driver):raise RuntimeError('NGX setup needs a complete NVIDIA driver version.')
    if await probe(run,driver,install_dir):return driver
    name='libnvidia-ngx.so.'+driver
    destination=install_dir/name
    notify('Checking the driver-matched NVIDIA NGX library…')
    if not destination.is_file():
        major=driver.split('.')[0]
        packages=[f'libnvidia-gl-{major}-server',f'libnvidia-gl-{major}']
        with tempfile.TemporaryDirectory(prefix='dsnn-ngx-') as temporary:
            folder=Path(temporary);selection=None
            for attempt in range(2):
                for package in packages:
                    code,output=await run('apt-cache','madison',package)
                    versions=[line.split('|')[1].strip() for line in output.splitlines() if '|' in line]
                    version=next((v for v in versions if v.split(':')[-1].startswith(driver+'-')),None)
                    if not code and version:selection=package+'='+version;break
                if selection:break
                if attempt==0:
                    code,_=await run('apt-get','-o','Acquire::Retries=3','update')
                    if code:break
            if not selection:
                raise RuntimeError('No signed Ubuntu NGX package matches NVIDIA driver '+driver+'. Use a host with matching NGX installed; the GPU driver was left intact.')
            code,_=await run('apt-get','download',selection,cwd=folder)
            if code:raise RuntimeError('The matching NGX package could not be downloaded.')
            archives=list(folder.glob('*.deb'))
            if len(archives)!=1:raise RuntimeError('NGX download did not produce one package.')
            extracted=folder/'extracted'
            code,_=await run('dpkg-deb','-x',archives[0],extracted)
            if code:raise RuntimeError('NGX package extraction failed.')
            candidates=[p for p in extracted.rglob(name) if p.is_file() and p.resolve().is_relative_to(extracted.resolve())]
            if len(candidates)!=1:raise RuntimeError('The signed package lacks a unique driver-matched NGX library.')
            install_dir.mkdir(parents=True,exist_ok=True)
            code,_=await run('install','-m','0644',candidates[0],destination)
            if code:raise RuntimeError('NGX userspace library installation failed.')
    for source,target in [(name,install_dir/'libnvidia-ngx.so.1'),('libnvidia-ngx.so.1',install_dir/'libnvidia-ngx.so')]:
        code,_=await run('ln','-sfn',source,target)
        if code:raise RuntimeError('NGX linker alias repair failed.')
    code,_=await run('ldconfig')
    if code or not await probe(run,driver,install_dir):
        raise RuntimeError('NGX could not be verified through the dynamic linker after repair.')
    return driver
