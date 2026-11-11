"""Segment Civitai transfers with private input files and authenticated local RPC."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import secrets
import socket
import tempfile
from typing import Callable
from uuid import uuid4

import httpx

KEYS = ['status', 'totalLength', 'completedLength', 'downloadSpeed', 'connections', 'errorCode']
ERRORS = {'2': 'connection timed out', '3': 'file not found', '8': 'server refused resume',
          '9': 'insufficient disk space', '22': 'host returned an HTTP error',
          '24': 'host rejected authentication', '28': 'server returned an invalid response'}


def connection_count() -> int:
    try:
        return max(1, min(16, int(os.getenv('LAUNCHER_CIVITAI_CONNECTIONS', '8'))))
    except ValueError:
        return 8


def input_file(path: Path, url: str, headers: dict[str, str]) -> None:
    values = [url, *headers.keys(), *headers.values()]
    if any('\n' in value or '\r' in value for value in values):
        raise RuntimeError('Invalid newline in download request')
    path.write_text(url+'\n'+''.join(f'  header={key}: {value}\n' for key,value in headers.items()))
    path.chmod(0o600)


async def _stop(process, waiting) -> None:
    if process.returncode is None:
        # SIGTERM lets aria2 save its piece map beside the partial file.
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(waiting), 6)
        except asyncio.TimeoutError:
            process.kill()
            await waiting
    else:
        await waiting


async def download(
    url: str, headers: dict[str, str], destination: Path,
    progress: Callable[[int, int, float, int], None], check_cancelled: Callable[[], None],
    *, connections: int | None = None, min_split: str = '16M',
) -> None:
    """Keep .aria2 piece maps on failure/cancel; never infer progress from file size."""
    connections = connection_count() if connections is None else max(1, min(16, connections))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    secret = secrets.token_urlsafe(32)
    gid = uuid4().hex[:16]
    with tempfile.TemporaryDirectory(prefix='dsnn-civitai-') as temporary:
        request = Path(temporary)/'request.txt'
        input_file(request, url, headers)
        command = [
            'aria2c', '--no-conf=true', '--input-file='+str(request),
            '--dir='+str(destination.parent), '--out='+destination.name,
            '--continue=true', '--auto-file-renaming=false', '--allow-overwrite=true',
            '--file-allocation=none', '--auto-save-interval=1',
            '--max-concurrent-downloads=1', '--split='+str(connections),
            '--max-connection-per-server='+str(connections), '--min-split-size='+min_split,
            '--max-tries=4', '--retry-wait=3', '--connect-timeout=15', '--timeout=30',
            '--enable-rpc=true', '--rpc-listen-all=false', '--rpc-listen-port='+str(port),
            '--rpc-secret='+secret, '--gid='+gid, '--stop=5',
            '--summary-interval=0', '--download-result=hide', '--quiet=true',
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        waiting = asyncio.create_task(process.wait())
        failure = ''
        try:
            # Never send localhost RPC through a Pod's external HTTP proxy.
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                async def rpc(method, *params):
                    response = await client.post(f'http://127.0.0.1:{port}/jsonrpc', json={
                        'jsonrpc':'2.0', 'id':gid, 'method':'aria2.'+method,
                        'params':['token:'+secret, *params],
                    })
                    response.raise_for_status()
                    result = response.json()
                    if 'error' in result:
                        return None
                    return result.get('result')

                while not waiting.done():
                    check_cancelled()
                    try:
                        status = await rpc('tellStatus', gid, KEYS)
                    except (httpx.HTTPError, ValueError):
                        status = None  # Local RPC may still be starting.
                    if isinstance(status, dict):
                        progress(max(0,int(status.get('completedLength',0))),
                                 max(0,int(status.get('totalLength',0))),
                                 max(0,float(status.get('downloadSpeed',0))),
                                 max(0,int(status.get('connections',0))))
                        if status.get('status') in {'complete','error','removed'}:
                            if status['status'] != 'complete':
                                code = str(status.get('errorCode','1'))
                                failure = f"Civitai transfer failed: {ERRORS.get(code, 'aria2 error')} (code {code})."
                            try:
                                await rpc('shutdown')
                            except (httpx.HTTPError,ValueError):
                                pass
                            break
                    await asyncio.wait({waiting}, timeout=.4)
                if not waiting.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(waiting), 6)
                    except asyncio.TimeoutError:
                        pass
        finally:
            await _stop(process, waiting)
        if failure:
            raise RuntimeError(failure)
        if process.returncode or not destination.is_file() or destination.with_name(destination.name+'.aria2').exists():
            raise RuntimeError(f'Civitai transfer did not complete (aria2 exit {process.returncode}); its partial data was retained for resume.')
