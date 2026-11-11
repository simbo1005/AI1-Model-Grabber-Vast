import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import threading
import time

import httpx
import pytest

from launcher import app, civitai_download as transfer


@pytest.mark.parametrize('outcome', ['complete', 'cancel', 'error'])
def test_private_requests_actual_byte_progress_and_safe_interruption(tmp_path, monkeypatch, outcome):
    destination=tmp_path/'model.part'
    destination.write_bytes(b'\0'*1024)  # A large sparse/preallocated length is not progress.
    control=destination.with_name(destination.name+'.aria2')
    if outcome=='cancel':control.write_bytes(b'piece map')
    seen=[]; arguments=[]; requests=[]; request_path=[]
    token='private-civitai-test-token'
    class FakeProcess:
        def __init__(self):self.returncode=None;self.done=asyncio.Event()
        async def wait(self):await self.done.wait();return self.returncode
        def terminate(self):self.returncode=0;self.done.set()
        def kill(self):self.terminate()
    process=None
    async def spawn(*args, **kwargs):
        nonlocal process
        arguments.extend(args)
        path=Path(next(x.split('=',1)[1] for x in args if x.startswith('--input-file=')))
        request_path.append(path)
        assert path.stat().st_mode & 0o777==0o600
        assert ('token='+token) in path.read_text()
        assert 'fileId=42' in path.read_text()
        assert '  out=model.part\n' in path.read_text()
        process=FakeProcess();return process
    def handler(request):
        payload=json.loads(request.content);requests.append(payload)
        if payload['method']=='aria2.shutdown':
            process.terminate();result='OK'
        else:
            result={'status':'active','completedLength':'128','totalLength':'1024','downloadSpeed':'64','connections':'4'}
            if seen and outcome!='cancel':
                result.update(status=outcome, completedLength='1024' if outcome=='complete' else '128',errorCode='24' if outcome=='error' else '0')
        return httpx.Response(200,json={'result':result})
    client=httpx.AsyncClient
    async def source(url,headers):return url,True
    monkeypatch.setattr(transfer,'_source',source)
    monkeypatch.setattr(transfer.asyncio,'create_subprocess_exec',spawn)
    monkeypatch.setattr(transfer.httpx,'AsyncClient',lambda **kw:client(transport=httpx.MockTransport(handler),**kw))
    def check():
        if seen and outcome=='cancel':raise app.InstallCancelled()
    async def run():
        await transfer.download('https://civitai.com/api/download/models/1?token='+token+'&fileId=42',{},destination,
                                lambda *values:seen.append(values),check)
    if outcome=='complete':asyncio.run(run())
    elif outcome=='cancel':
        with pytest.raises(app.InstallCancelled):asyncio.run(run())
        assert control.read_bytes()==b'piece map'
    else:
        with pytest.raises(RuntimeError,match='authentication') as error:asyncio.run(run())
        assert token not in str(error.value)
    assert seen[0]==(128,1024,64.,4)
    assert all(token not in value for value in arguments)
    assert token not in json.dumps(requests)
    assert not request_path[0].exists()


def test_wget_never_treats_sparse_segmented_data_as_a_contiguous_prefix(tmp_path):
    destination=tmp_path/'redgpt.safetensors'
    staging=tmp_path/'.dsnn-downloads';staging.mkdir()
    (staging/'redgpt.safetensors.part').write_bytes(b'sparse')
    (staging/'redgpt.safetensors.part.aria2').write_bytes(b'pieces')
    controller=app.JobController()
    with pytest.raises(RuntimeError,match='aria2 to resume safely'):
        asyncio.run(controller._download_file_with_wget('https://civitai.com/file',{},'REDGPT',destination,
                    tmp_path/'redgpt.safetensors.part',100,'',0,1,0,100,88))


def test_civitai_is_routed_to_aria2_and_still_verifies_model_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(app,'COMFYUI_DIR',tmp_path)
    monkeypatch.setenv('CIVITAI_TOKEN','test-token')
    monkeypatch.setattr(app.shutil,'which',lambda name:'/usr/bin/'+name)
    controller=app.JobController();calls=[]
    payload=b'correct model'
    async def download(url, headers, destination, progress, cancelled):
        assert 'fileId=42' in url and 'token=test-token' in url
        calls.append(destination)
        destination.write_bytes(payload)
        progress(len(payload),len(payload),100,4)
    monkeypatch.setattr(app,'download_civitai',download)
    async def forbidden(*args,**kwargs):pytest.fail('Civitai should use aria2 here')
    monkeypatch.setattr(controller,'_download_file_with_wget',forbidden)
    spec={'name':'REDGPT','url':'https://civitai.com/api/download/models/1?fileId=42','auth':'civitai',
          'destination':'models/test.safetensors','size_bytes':len(payload),'sha256':'0'*64}
    with pytest.raises(RuntimeError,match='Checksum verification failed'):
        asyncio.run(controller._download_file(None,spec,0,1,0,len(payload),88))
    assert not (tmp_path/'models/test.safetensors').exists()
    assert (tmp_path/'models/test.safetensors.part').read_bytes()==payload
    assert len(calls)==1


@pytest.fixture
def origin():
    payload=bytes(range(256))*65536  # 16 MiB, generated locally; no model download.
    ranges=[];support={'ranges':True}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            raw=self.headers.get('Range','');ranges.append(raw)
            start=0;end=len(payload)-1
            match=re.fullmatch(r'bytes=(\d+)-(\d*)',raw) if support['ranges'] else None
            if match:
                start=int(match[1]);end=min(int(match[2]) if match[2] else end,end)
                self.send_response(206)
                self.send_header('Content-Range',f'bytes {start}-{end}/{len(payload)}')
            else:self.send_response(200)
            if support['ranges']:self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(end-start+1));self.end_headers()
            try:
                for offset in range(start,end+1,32768):
                    self.wfile.write(payload[offset:min(offset+32768,end+1)]);self.wfile.flush();time.sleep(.01)
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/file?fileId=42&token=fake',payload,ranges,support
    server.shutdown();server.server_close();thread.join()


@pytest.mark.skipif(not shutil.which('aria2c'),reason='CI installs aria2 for native transfer tests')
@pytest.mark.parametrize('supports_ranges',[True,False])
def test_real_transfer_handles_range_and_non_range_hosts(tmp_path,origin,supports_ranges):
    url,payload,ranges,support=origin;support['ranges']=supports_ranges
    destination=tmp_path/'test.part';updates=[]
    asyncio.run(transfer.download(url,{},destination,lambda *x:updates.append(x),lambda:None,connections=4,min_split='1M'))
    assert hashlib.sha256(destination.read_bytes()).digest()==hashlib.sha256(payload).digest()
    assert all(0<=current<=total for current,total,speed,connections in updates)
    if supports_ranges:assert sum(bool(re.match(r'bytes=[1-9]\d*-',r)) for r in ranges)>=2
    assert not destination.with_name(destination.name+'.aria2').exists()


@pytest.mark.skipif(not shutil.which('aria2c'),reason='CI installs aria2 for native transfer tests')
def test_real_cancel_and_resume_retains_the_piece_map(tmp_path,origin):
    url,payload,ranges,support=origin
    destination=tmp_path/'test.part';updates=[]
    def cancelled():
        if any(x[0]>131072 for x in updates):raise app.InstallCancelled()
    with pytest.raises(app.InstallCancelled):
        asyncio.run(transfer.download(url,{},destination,lambda *x:updates.append(x),cancelled,connections=4,min_split='1M'))
    control=destination.with_name(destination.name+'.aria2')
    assert destination.exists() and control.exists()
    before=len(ranges)
    asyncio.run(transfer.download(url,{},destination,lambda *x:None,lambda:None,connections=4,min_split='1M'))
    assert hashlib.sha256(destination.read_bytes()).digest()==hashlib.sha256(payload).digest()
    assert any(ranges[before:])
    assert not control.exists()


@pytest.mark.skipif(not shutil.which('aria2c'),reason='CI installs aria2 for native transfer tests')
def test_real_resume_adopts_an_existing_wget_prefix(tmp_path,origin):
    url,payload,ranges,support=origin
    destination=tmp_path/'test.part';destination.write_bytes(payload[:1048576])
    asyncio.run(transfer.download(url,{},destination,lambda *x:None,lambda:None,connections=4,min_split='1M'))
    assert hashlib.sha256(destination.read_bytes()).digest()==hashlib.sha256(payload).digest()
    assert any(ranges)


def test_metadata_errors_do_not_expose_civitai_tokens(monkeypatch):
    client=httpx.AsyncClient
    monkeypatch.setattr(transfer.httpx,'AsyncClient',lambda **kw:client(transport=httpx.MockTransport(lambda req:httpx.Response(403)),**kw))
    with pytest.raises(RuntimeError,match='HTTP 403') as error:
        asyncio.run(transfer._source('https://civitai.com/file?token=private-secret',{}))
    assert 'private-secret' not in str(error.value)
