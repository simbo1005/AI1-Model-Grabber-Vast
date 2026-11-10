import asyncio
import copy
import hashlib
import io
from pathlib import Path
import tarfile

import httpx
import pytest

from launcher import app, bundled_nodes, krast_setup, node_setup


def workflow():
    return next(w for w in app.load_catalog()['workflows'] if w['id'] == 'snn-krast')


def test_krast_uses_the_exact_public_dataset_and_original_filenames():
    plan = workflow()
    resources = plan['files']
    assert len(resources) == 41
    assert sum(f['size_bytes'] for f in resources) == 101266013047
    assert all(f['default_selected'] for f in resources)
    assert sum(f['auth'] == 'civitai' for f in resources) == 2
    assert sum(bool(app.huggingface_file_reference(f['url'])) for f in resources) == 39
    krast = resources[0]
    reference = app.huggingface_file_reference(krast['url'])
    assert reference == ('simbo926/KrastRaw', 'dataset',
        'a23a479cd6a0c22359b18a1cc6df5f1636617eb1', 'Krast-Raw-bf16.safetensors')
    assert krast['destination'] == 'models/diffusion_models/krastBf16_v3_bf16.safetensors'
    assert krast['sha256'] == '3b2c89871d08bb13e8bf166edb9fd2e23c1349b9b48a0e1cad095f3f03a03335'
    loras = [f for f in resources if f['destination'].startswith('models/loras/')]
    assert len(loras) == 11
    mirror = next(f for f in loras if 'RealisticSnapshot' in f['destination'])
    assert '/stormi2/realisticsnapshot2/' in mirror['url']
    assert mirror['sha256'] == 'bc064297da569e9f25fe362d8210b291093fad5fa4f59ea982fa4f7c4151e2bc'
    assert plan['prepare_krast_placeholder'] is True
    assert len(plan['required_core_node_types']) == 37
    assert {n['name'] for n in plan['custom_nodes']} >= {
        'rgthree-comfy', 'ComfyUI-SamplerV3', 'mrteez_prompt_source',
        'ComfyUI_LC123_nodes', 'ComfyUI_LC_MaskMaker_nodes', 'ComfyUI_LC_Vision_nodes',
        'Bond-Node-Suite',
    }


def test_every_krast_resource_takes_hf_cli_even_if_wget_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(app.shutil, 'which', lambda executable: '/usr/bin/' + executable)
    controller = app.JobController()
    calls = []
    monkeypatch.setenv('HF_TOKEN', 'hf_test_fixture')

    async def hf(reference, name, destination, partial, expected_size, expected_sha,
                 index, count, completed, total, ceiling, *, requires_token):
        calls.append((reference, destination, requires_token))
        return expected_size

    async def wget(*args, **kwargs):
        pytest.fail('KRAST Hugging Face downloads must never fall back to wget')

    monkeypatch.setattr(controller, '_download_file_with_hf_cli', hf)
    monkeypatch.setattr(controller, '_download_file_with_wget', wget)

    async def run():
        for index, resource in enumerate(f for f in workflow()['files'] if app.huggingface_file_reference(f['url'])):
            await controller._download_file(None, resource, index, 15, 0, 0, 88)

    asyncio.run(run())
    assert len(calls) == 39
    assert calls[0][0][1] == 'dataset'
    assert sum(token is True for _, _, token in calls) == 1


def test_krast_public_resources_default_to_selected():
    public = next(w for w in app.public_catalog()['workflows'] if w['id'] == 'snn-krast')
    assert len(public['resources']) == 41
    assert all(f['selected_by_default'] for f in public['resources'])
    assert all('url' not in f and 'auth' not in f for f in public['resources'])


def archive_for(name, members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for relative, value in members.items():
            member = tarfile.TarInfo('opt/app/custom_nodes/' + name + '/' + relative)
            if isinstance(value, bytes):
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = value
                archive.addfile(member)
    return buffer.getvalue()


def bundle_spec(name='ComfyUI-SamplerV3'):
    return {'name': name, 'source': 'image_bundle',
            'source_image': bundled_nodes.IMAGE, 'source_layer': bundled_nodes.LAYER}


def test_bundle_installs_atomically_and_reruns_without_network(tmp_path, monkeypatch):
    name = 'ComfyUI-SamplerV3'
    content = {'__init__.py': b'fixture', 'nodes.py': b'fixture node'}
    monkeypatch.setitem(bundled_nodes.FILES, name,
                        {k: hashlib.sha256(v).hexdigest() for k, v in content.items()})
    calls = []

    async def fetch():
        assert not (tmp_path/name).exists()
        calls.append(True)
        return archive_for(name, content)

    monkeypatch.setattr(bundled_nodes, 'fetch_layer', fetch)
    for _ in range(2):
        result = asyncio.run(bundled_nodes.prepare_bundled_node(bundle_spec(), tmp_path))
        assert (result/'nodes.py').read_bytes() == content['nodes.py']
    assert len(calls) == 1
    assert not list(tmp_path.glob('.dsnn-*'))


def test_bundle_rejects_unreviewed_sources_and_preserves_existing_files(tmp_path):
    bad = bundle_spec()
    bad['source_image'] = 'other/image:latest'
    with pytest.raises(RuntimeError, match='Unsupported or unpinned'):
        asyncio.run(bundled_nodes.prepare_bundled_node(bad, tmp_path))
    existing = tmp_path/'ComfyUI-SamplerV3'
    existing.mkdir()
    (existing/'nodes.py').write_text('my edited sampler')
    with pytest.raises(RuntimeError, match='differs from the pinned original'):
        asyncio.run(bundled_nodes.prepare_bundled_node(bundle_spec(), tmp_path))
    assert (existing/'nodes.py').read_text() == 'my edited sampler'


@pytest.mark.parametrize('member,value', [('../escape.py', b'bad'), ('__init__.py', '/etc/passwd')])
def test_bundle_rejects_traversal_and_symlinks(member, value):
    with pytest.raises(RuntimeError, match='Unsafe|Unexpected'):
        bundled_nodes.extract_files(archive_for('ComfyUI-SamplerV3', {member:value}), 'ComfyUI-SamplerV3')


def test_bundle_rejects_a_modified_source():
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        bundled_nodes.extract_files(archive_for('ComfyUI-SamplerV3',
            {'__init__.py': b'modified'}), 'ComfyUI-SamplerV3')


@pytest.mark.parametrize('oversized', [False, True])
def test_bundle_download_rejects_corrupt_or_oversized_layer(monkeypatch, oversized):
    original = httpx.AsyncClient
    monkeypatch.setattr(bundled_nodes, '_layer_cache', None)
    monkeypatch.setattr(bundled_nodes, 'MAX_LAYER_BYTES', 100)

    def handler(request):
        if request.url.host == 'auth.docker.io':
            return httpx.Response(200, json={'token': 'fake-public-registry-token'})
        assert request.headers['Authorization'] == 'Bearer fake-public-registry-token'
        return httpx.Response(200, content=b'x' * (101 if oversized else 10))

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(bundled_nodes.httpx, 'AsyncClient',
                        lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(RuntimeError, match='limit|checksum mismatch'):
        asyncio.run(bundled_nodes.fetch_layer())


def test_registry_rejects_missing_krea2_support_or_lut():
    node = {'name':'critical', 'required_node_types':['CLIPLoader', 'LCApplyLUT'],
            'required_input_options':{'CLIPLoader':{'type':['krea2']},
                                      'LCApplyLUT':{'lut_name':['LC_Crushed_Blacks.cube']}}}
    registry = {'CLIPLoader':{'input':{'required':{'type':[['sdxl']]}}},
                'LCApplyLUT':{'input':{'required':{'lut_name':[['none']]}}}}
    check = node_setup.node_registration_checks([node], registry)[0]
    assert check['status'] == 'missing'
    assert 'krea2' in check['message'] and 'LC_Crushed_Blacks.cube' in check['message']
    registry['CLIPLoader']['input']['required']['type'] = ['COMBO', {'options':['krea2']}]
    registry['LCApplyLUT']['input']['required']['lut_name'] = [['LC_Crushed_Blacks.cube']]
    assert node_setup.node_registration_checks([node], registry)[0]['status'] == 'verified'


def test_vision_installer_is_pinned_and_uses_comfys_python(tmp_path, monkeypatch):
    node = next(n for n in workflow()['custom_nodes'] if n['name']=='ComfyUI_LC_Vision_nodes')
    python = tmp_path/'python'; python.touch()
    installer = tmp_path/'install.py'; installer.write_bytes(b'reviewed fixture')
    monkeypatch.setattr(node_setup, 'VISION_INSTALLER_SHA256', hashlib.sha256(installer.read_bytes()).hexdigest())
    support = tmp_path/'requirements.txt'; support.write_text('diskcache>=5.6.3,<6\n')
    calls=[]

    async def run(*command):
        calls.append(command)
        if 'freeze' in command:return 0,'torch==2.10.0+cu128\nnumpy==2.2.6\n'
        if '--constraint' in command:
            constraints=Path(command[command.index('--constraint')+1]).read_text()
            assert 'torch==2.10.0+cu128' in constraints and 'numpy==2.2.6' in constraints
        return 0,''

    asyncio.run(node_setup.install_reviewed_post_install(node,tmp_path,python,run,support))
    assert calls[-1] == (python,installer)
    changed=copy.deepcopy(node);changed['ref']='a'*40
    with pytest.raises(RuntimeError,match='Unsupported'):
        asyncio.run(node_setup.install_reviewed_post_install(changed,tmp_path,python,run,support))


@pytest.mark.parametrize('present',[True,False])
def test_placeholder_respects_configured_input_directory_and_existing_images(monkeypatch,present):
    original=httpx.AsyncClient
    requests=[]

    def handler(request):
        requests.append(request)
        if request.method=='GET':return httpx.Response(200 if present else 404)
        assert b'filename="studio_placeholder.png"' in request.content
        assert krast_setup.placeholder_png() in request.content
        assert b'false' in request.content
        return httpx.Response(200,json={'name':'studio_placeholder.png'})

    transport=httpx.MockTransport(handler)
    monkeypatch.setattr(krast_setup.httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    asyncio.run(krast_setup.ensure_placeholder('http://comfy.local'))
    assert [r.method for r in requests] == (['GET'] if present else ['GET','POST'])


def test_krast_placeholder_failure_never_reports_success(monkeypatch):
    original=httpx.AsyncClient
    transport=httpx.MockTransport(lambda request:httpx.Response(403))
    monkeypatch.setattr(krast_setup.httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    with pytest.raises(RuntimeError,match='Could not check'):
        asyncio.run(krast_setup.ensure_placeholder('http://comfy.local'))
