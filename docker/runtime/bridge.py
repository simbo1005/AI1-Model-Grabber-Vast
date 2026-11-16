"""Native ComfyUI bridge for an optional, isolated MiniMax worker on loopback."""
from __future__ import annotations
import json
from pathlib import Path
from uuid import uuid4

PROFILE = 'minimax-h3-cu130'
PORT = 18189
PLUGIN = '''# DSNN runtime bridge v1
import runpy
from pathlib import Path
import folder_paths
from server import PromptServer
runpy.run_path('/opt/dsnn-attention/runtime_bridge.py')['register'](PromptServer.instance, Path(folder_paths.base_path))
NODE_CLASS_MAPPINGS = {}
'''
HOP = {'connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade'}


def state(root):
    path=Path(root)/'.dsnn/runtime.json'
    if not path.exists():return {'active':None,'busy':False}
    data=json.loads(path.read_text())
    if data.get('active') not in (None,PROFILE) or not isinstance(data.get('busy',False),bool):
        raise ValueError('Invalid DSNN runtime selection')
    return {'active':data.get('active'),'busy':data.get('busy',False)}


def install(root):
    package=Path(root)/'custom_nodes/DSNN_Runtime_Bridge'
    if package.is_symlink():raise RuntimeError('Preserving a symlinked runtime bridge directory')
    package.mkdir(parents=True,exist_ok=True)
    path=package/'__init__.py'
    if path.is_symlink() or (path.exists() and path.read_text()!=PLUGIN):
        raise RuntimeError('Preserving a modified DSNN runtime bridge')
    if not path.exists():
        temporary=path.with_name('__init__.'+uuid4().hex+'.tmp')
        try:
            temporary.write_text(PLUGIN);temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)


def register(server, root):
    from aiohttp import ClientSession, ClientTimeout, ClientError, WSMsgType, web
    import asyncio
    from yarl import URL
    key=web.AppKey('dsnn_runtime_session',ClientSession)
    async def lifetime(app):
        app[key]=ClientSession(auto_decompress=False,timeout=ClientTimeout(total=None,connect=10))
        yield
        await app[key].close()
    server.app.cleanup_ctx.append(lifetime)

    @server.routes.get('/dsnn/runtime')
    async def status(request):
        try:return web.json_response({**state(root),'supports_isolation':True})
        except (ValueError,OSError):return web.json_response({'error':'Invalid runtime selection'},status=503)

    @web.middleware
    async def proxy(request,handler):
        try:selection=state(root)
        except (ValueError,OSError):return web.json_response({'error':'Invalid runtime selection'},status=503)
        if selection['busy'] and request.method=='POST' and request.path in {'/prompt','/interrupt','/queue'}:
            return web.json_response({'error':'ComfyUI runtime is switching; retry when setup finishes.'},status=409)
        if request.path=='/dsnn/runtime' or selection['active'] is None:
            return await handler(request)
        target=f'http://127.0.0.1:{PORT}'
        headers={k:v for k,v in request.headers.items() if k.lower() not in HOP|{'host','origin','content-length'}}
        # The native server's existing origin/auth middleware runs before this
        # bridge. Align the private hop's Origin with its fixed loopback Host.
        if 'Origin' in request.headers:headers['Origin']=target
        try:
            if web.WebSocketResponse().can_prepare(request).ok:
                async with request.app[key].ws_connect(URL(target+request.raw_path,encoded=True),headers=headers,autoping=False,max_msg_size=0) as upstream:
                    downstream=web.WebSocketResponse(autoping=False,max_msg_size=0)
                    await downstream.prepare(request)
                    async def relay(source,destination):
                        async for message in source:
                            if message.type==WSMsgType.TEXT:await destination.send_str(message.data)
                            elif message.type==WSMsgType.BINARY:await destination.send_bytes(message.data)
                            elif message.type==WSMsgType.PING:await destination.ping(message.data)
                            elif message.type==WSMsgType.PONG:await destination.pong(message.data)
                            elif message.type in (WSMsgType.ERROR,WSMsgType.CLOSE):break
                        await destination.close()
                    tasks=[asyncio.create_task(relay(downstream,upstream)),asyncio.create_task(relay(upstream,downstream))]
                    try:
                        await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:task.cancel()
                        await asyncio.gather(*tasks,return_exceptions=True)
                    return downstream
            body=request.content.iter_chunked(65536) if request.can_read_body else None
            async with request.app[key].request(request.method,URL(target+request.raw_path,encoded=True),headers=headers,data=body,allow_redirects=False) as response:
                output=web.StreamResponse(status=response.status,headers={k:v for k,v in response.headers.items() if k.lower() not in HOP})
                await output.prepare(request)
                async for chunk in response.content.iter_chunked(65536):await output.write(chunk)
                await output.write_eof();return output
        except (ClientError,ConnectionError,TimeoutError):
            return web.json_response({'error':'The isolated ComfyUI worker is not ready. Restart it from the Model Grabber.'},status=503)
    server.app.middlewares.append(proxy)
