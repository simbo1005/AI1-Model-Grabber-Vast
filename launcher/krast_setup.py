"""Small non-model assets needed by the unchanged KRAST graph."""
import struct
import zlib

import httpx


def placeholder_png() -> bytes:
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    pixels = (b'\0' + b'\x80\x80\x80' * 16) * 16
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', 16, 16, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


async def ensure_placeholder(comfy_url: str) -> None:
    """Use ComfyUI's configured input directory, preserving any existing image."""
    name = 'studio_placeholder.png'
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream('GET', comfy_url + '/view',
                                     params={'filename': name, 'type': 'input'}) as response:
                if response.status_code == 200:
                    return
                if response.status_code != 404:
                    raise RuntimeError('Could not check the KRAST reference-image placeholder.')
            result = await client.post(comfy_url + '/upload/image',
                files={'image': (name, placeholder_png(), 'image/png')},
                data={'type': 'input', 'overwrite': 'false'})
            result.raise_for_status()
            if result.json().get('name') != name:
                raise RuntimeError('ComfyUI returned an unexpected placeholder filename.')
    except (httpx.HTTPError, ValueError):
        raise RuntimeError('Could not prepare the KRAST reference-image placeholder in ComfyUI.') from None
