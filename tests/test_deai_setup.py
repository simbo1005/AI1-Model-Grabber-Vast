import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from launcher import app, node_setup


def plan():
    return next(w for w in app.load_catalog()['workflows'] if w['id'] == 'snn-krast')


def test_deai_dependencies_precede_reviewed_stack_installer():
    nodes = plan()['custom_nodes']
    stack = nodes[-1]
    assert stack['name'] == 'DSNN-DeAI-Nodes'
    assert stack['ref'] == node_setup.DEAI_REF
    assert stack['post_install'] == 'install.py'
    assert stack['install_requirements'] is False
    names = {n['name']: n for n in nodes[:-1]}
    assert names['ComfyUI_VintageEffect']['required_node_types'] == ['Vintage Effect']
    assert names['radiance']['required_node_types'] == ['RadianceCompressionArtifacts']
    assert 'LCLensProfile' in names['ComfyUI_LC123_nodes']['required_node_types']
    assert 'libimage-exiftool-perl' in plan()['system_packages']


@pytest.mark.parametrize('changed', [False, True])
def test_deai_post_install_uses_active_python_without_reinstalling_cuda(tmp_path, monkeypatch, changed):
    repository = tmp_path / 'ComfyUI/custom_nodes/DSNN-DeAI-Nodes'
    repository.mkdir(parents=True)
    source = repository / 'install.py'
    source.write_text('reviewed fixture')
    monkeypatch.setattr(node_setup, 'DEAI_FILES', {
        'install.py': hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    if changed:
        source.write_text('unexpected changes')
    python = tmp_path / 'venv/bin/python'
    calls = []

    async def run(*command):
        calls.append(command)
        return 0, ''

    operation = node_setup.install_reviewed_post_install(
        plan()['custom_nodes'][-1], repository, python, run, tmp_path/'unused.txt',
    )
    if changed:
        with pytest.raises(RuntimeError, match='missing or changed'):
            asyncio.run(operation)
        assert not calls
    else:
        asyncio.run(operation)
        assert calls == [(python, source, '--comfyui', tmp_path/'ComfyUI',
                          '--python', python, '--skip-deps', '--skip-system-tools')]


def test_redqw_mirror_preserves_existing_workflow_model_name():
    files = plan()['files']
    v2 = next(f for f in files if Path(f['destination']).name ==
              'redgptK2T2W6A8REDQW21_redqw21UNLOCKEDV2_fp8.safetensors')
    assert '/simbo926/dsnn-krea2-resources/' in v2['url']
    assert v2['auth'] == 'none'
    assert v2['sha256'] == '0efb5aeb2b372025042e320c2e35c66ec6681ef54ad5de88a652ab19cc63ad92'
    assert not any('3370753' in f['url'] for f in files)
    for name in ['REDQW21-UNLOCKED-v3-FP8-ComfyMCP-builtwithqwen.safetensors',
                 'Verite_realism.safetensors', 'HORNY AMATEUR LORA.safetensors',
                 'texture_fix_vae_for_qwen_image_2.1_bf16.safetensors',
                 '4xNomosWebPhoto_RealPLKSR.pth', 'krea2RealVae_v10.safetensors']:
        resource = next(f for f in files if Path(f['destination']).name == name)
        assert resource['default_selected'] and resource['auth'] == 'none'
        assert app.huggingface_file_reference(resource['url'])
