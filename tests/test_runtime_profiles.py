"""Exercise runtime isolation, job preservation and the real HTTP/WS bridge."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web, ClientSession
from yarl import URL

from launcher import runtime_profiles as runtime
from launcher import ngx_setup


def controller(tmp_path):
    feature=tmp_path/'bridge.py';feature.touch()
    return runtime.RuntimeController(tmp_path/'native',tmp_path/'native-venv','http://native',feature=feature)


def bridge_module():
    path=Path(__file__).resolve().parents[1]/'docker/runtime/bridge.py'
    spec=importlib.util.spec_from_file_location('tested_bridge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def mock_http(monkeypatch, handler):
    client=httpx.AsyncClient
    monkeypatch.setattr(runtime.httpx,'AsyncClient',lambda **kw:client(transport=httpx.MockTransport(handler),**kw))


def test_child_environment_does_not_inherit_native_cuda_constraints(tmp_path, monkeypatch):
    c=controller(tmp_path)
    for name in ('PIP_CONSTRAINT','PIP_BUILD_CONSTRAINT','PYTHONPATH','PYTHONHOME'):
        monkeypatch.setenv(name,'native-only')
    monkeypatch.setenv('HF_TOKEN','provider-secret')
    env=c.environment()
    assert all(name not in env for name in ('PIP_CONSTRAINT','PIP_BUILD_CONSTRAINT','PYTHONPATH','PYTHONHOME'))
    assert env['HF_TOKEN']=='provider-secret'
    assert env['VIRTUAL_ENV']==str(c.venv)
    assert env['PATH'].split(':')[0]==str(c.venv/'bin')
    assert __import__('os').environ['PIP_CONSTRAINT']=='native-only'


def test_driver_gate_runs_before_any_environment_mutation(tmp_path):
    c=controller(tmp_path);commands=[]
    async def run(*args,**kw):commands.append(args);return 0,'570.124.06\n'
    with pytest.raises(RuntimeError,match='580'):
        asyncio.run(c.prepare(run,lambda _:None,lambda:None))
    assert len(commands)==1 and commands[0][0]=='nvidia-smi'
    assert not c.folder.exists()


def test_impact_source_dependency_is_pinned_without_editing_checkout(tmp_path):
    c=controller(tmp_path);source=tmp_path/'requirements.txt'
    original='torch\ngit+https://github.com/facebookresearch/sam2\nnumpy\n';source.write_text(original)
    with c.requirements(source) as reviewed:
        assert reviewed!=source
        assert 'sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4' in reviewed.read_text()
        assert source.read_text()==original
    assert not reviewed.exists()


def test_switch_refuses_running_or_pending_jobs_and_preserves_selection(tmp_path,monkeypatch):
    c=controller(tmp_path)
    runtime.atomic_json(c.marker,{'active':runtime.PROFILE,'busy':False})
    mock_http(monkeypatch,lambda r:httpx.Response(200,json={'queue_running':[], 'queue_pending':[[1,'job']]}))
    async def forbidden():raise AssertionError('must not stop/start a busy worker')
    monkeypatch.setattr(c,'_start',forbidden);monkeypatch.setattr(c,'stop',forbidden)
    with pytest.raises(RuntimeError,match='queued work'):asyncio.run(c.activate())
    assert c.selection()=={'active':runtime.PROFILE,'busy':False}
    with pytest.raises(RuntimeError,match='queued work'):asyncio.run(c.deactivate())
    assert c.selection()=={'active':runtime.PROFILE,'busy':False}


def test_failed_worker_start_returns_to_native_runtime(tmp_path,monkeypatch):
    c=controller(tmp_path)
    mock_http(monkeypatch,lambda r:httpx.Response(200,json={'supports_isolation':True,'queue_running':[],'queue_pending':[]}))
    async def fail():raise RuntimeError('worker import failed')
    monkeypatch.setattr(c,'_start',fail)
    with pytest.raises(RuntimeError,match='worker import'):asyncio.run(c.activate())
    assert c.selection()=={'active':None,'busy':False}


def test_stale_selection_recovers_only_when_worker_and_private_port_are_absent(tmp_path,monkeypatch):
    c=controller(tmp_path);runtime.atomic_json(c.marker,{'active':runtime.PROFILE,'busy':True})
    def response(request):return httpx.Response(503 if c.active() else 200,json={'queue_pending':[]})
    mock_http(monkeypatch,response)
    async def refused(*args):raise ConnectionRefusedError()
    monkeypatch.setattr(runtime.asyncio,'open_connection',refused)
    asyncio.run(c.idle())
    assert c.selection()=={'active':None,'busy':True}


def test_modified_bridge_and_symlinks_are_preserved(tmp_path):
    bridge=bridge_module();bridge.install(tmp_path);bridge.install(tmp_path)
    path=tmp_path/'custom_nodes/DSNN_Runtime_Bridge/__init__.py'
    path.write_text('local customization')
    with pytest.raises(RuntimeError,match='modified'):bridge.install(tmp_path)
    assert path.read_text()=='local customization'


def test_bridge_streams_http_and_websockets_and_respects_native_guard(tmp_path):
    async def exercise():
        bridge=bridge_module();seen=[]
        async def echo(request):
            seen.append((request.raw_path,request.headers.get('Authorization'),request.headers.get('Origin')))
            if request.path=='/ws':
                ws=web.WebSocketResponse();await ws.prepare(request)
                async for message in ws:
                    if message.type==web.WSMsgType.TEXT:await ws.send_str(message.data)
                    else:await ws.send_bytes(message.data)
                return ws
            return web.Response(body=await request.read(),headers={'X-Worker':'isolated'},status=201)
        worker=web.Application();worker.router.add_route('*','/{path:.*}',echo)
        worker_runner=web.AppRunner(worker);await worker_runner.setup()
        site=web.TCPSite(worker_runner,'127.0.0.1',0);await site.start()
        bridge.PORT=site._server.sockets[0].getsockname()[1]
        @web.middleware
        async def native_guard(request,handler):
            if request.headers.get('Origin')=='http://untrusted':return web.Response(status=403)
            return await handler(request)
        native=web.Application(middlewares=[native_guard]);routes=web.RouteTableDef()
        bridge.register(SimpleNamespace(app=native,routes=routes),tmp_path)
        native.add_routes(routes)
        async def native_handler(request):return web.json_response({'native':True})
        native.router.add_route('*','/{path:.*}',native_handler)
        native_runner=web.AppRunner(native);await native_runner.setup()
        public=web.TCPSite(native_runner,'127.0.0.1',0);await public.start()
        url='http://127.0.0.1:'+str(public._server.sockets[0].getsockname()[1])
        try:
            async with ClientSession() as client:
                async with client.get(url+'/queue') as response:assert (await response.json())['native']
                runtime.atomic_json(tmp_path/'.dsnn/runtime.json',{'active':runtime.PROFILE,'busy':False})
                payload=b'\x00\xffbinary-upload'*10000
                async with client.post(URL(url+'/upload/image?name=a%2Fb',encoded=True),data=payload,headers={'Authorization':'Bearer test','Origin':url}) as response:
                    assert response.status==201 and await response.read()==payload
                    assert response.headers['X-Worker']=='isolated'
                assert seen[-1]==('/upload/image?name=a%2Fb','Bearer test',f'http://127.0.0.1:{bridge.PORT}')
                async with client.ws_connect(url+'/ws?clientId=test',headers={'Origin':url}) as ws:
                    await ws.send_str('status');assert (await ws.receive()).data=='status'
                    await ws.send_bytes(b'preview');assert (await ws.receive()).data==b'preview'
                async with client.post(url+'/prompt',headers={'Origin':'http://untrusted'}) as response:assert response.status==403
                runtime.atomic_json(tmp_path/'.dsnn/runtime.json',{'active':runtime.PROFILE,'busy':True})
                async with client.post(url+'/prompt') as response:assert response.status==409
                async with client.get(url+'/dsnn/runtime') as response:assert (await response.json())['supports_isolation']
                runtime.atomic_json(tmp_path/'.dsnn/runtime.json',{'active':'http://arbitrary-host','busy':False})
                async with client.get(url+'/queue') as response:assert response.status==503
        finally:await native_runner.cleanup();await worker_runner.cleanup()
    asyncio.run(exercise())


def test_ngx_alias_resolves_versioned_library_without_package_changes(tmp_path):
    library=tmp_path/'libnvidia-ngx.so.595.91.07';library.touch()
    alias=tmp_path/'libnvidia-ngx.so.1';alias.symlink_to(library.name)
    commands=[]
    async def run(*args,**kw):
        commands.append(args);assert args==('ldconfig','-p')
        return 0,f' libnvidia-ngx.so.1 (libc6,x86-64) => {alias}\n'
    assert asyncio.run(ngx_setup.ensure(run,'595.91.07',lambda _:None,tmp_path))=='595.91.07'
    assert len(commands)==1


def test_ngx_never_installs_driver_packages_when_no_exact_match(tmp_path):
    commands=[]
    async def run(*args,**kw):
        commands.append(args)
        return 0,'libnvidia-gl-595 | 595.50.01-0ubuntu1 | archive' if args[0]=='apt-cache' else ''
    with pytest.raises(RuntimeError,match='matches NVIDIA driver'):
        asyncio.run(ngx_setup.ensure(run,'595.91.07',lambda _:None,tmp_path))
    assert not any('install' in args for args in commands)
    assert not list(tmp_path.glob('libnvidia*'))


def test_ngx_repairs_only_the_exact_userspace_library(tmp_path):
    driver='595.91.07';install=tmp_path/'lib';install.mkdir();commands=[]
    async def run(*args,**kw):
        commands.append(tuple(map(str,args)))
        if args[:2]==('ldconfig','-p'):
            alias=install/'libnvidia-ngx.so.1'
            return (0,f' libnvidia-ngx.so.1 (libc6) => {alias}\n' if alias.exists() else '')
        if args[0]=='apt-cache':return 0,'pkg | 595.91.07-0ubuntu0.24.04.1 | signed-archive'
        if args[:2]==('apt-get','download'):(kw['cwd']/'package.deb').touch()
        if args[0]=='dpkg-deb':
            target=Path(args[3])/'usr/lib/libnvidia-ngx.so.595.91.07'
            target.parent.mkdir(parents=True);target.write_text('userspace ngx')
        if args[0]=='install':__import__('shutil').copyfile(args[3],args[4])
        if args[0]=='ln':Path(args[3]).symlink_to(args[2])
        return 0,''
    assert asyncio.run(ngx_setup.ensure(run,driver,lambda _:None,install))==driver
    assert (install/'libnvidia-ngx.so').resolve().read_text()=='userspace ngx'
    assert not any(command[:2]==('apt-get','install') for command in commands)
    assert {p.name for p in install.iterdir()}=={'libnvidia-ngx.so','libnvidia-ngx.so.1','libnvidia-ngx.so.595.91.07'}
