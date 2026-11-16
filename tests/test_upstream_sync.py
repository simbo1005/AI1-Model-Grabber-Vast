"""Protect the existing provider pipelines and download paths during upstream sync."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit,parse_qsl

from launcher import app

ROOT=Path(__file__).resolve().parents[1]
BASELINE=json.loads((ROOT/'tests/fixtures/upstream_preservation.json').read_text())


def test_existing_workflows_preserved_except_reviewed_gate_and_krqw_additions():
    workflows={w['id']:w for w in app.load_catalog()['workflows']}
    for identifier,snapshot in BASELINE['workflows'].items():
        workflow=copy.deepcopy(workflows[identifier])
        if identifier=='krqw2-1':workflow['files']=workflow['files'][:snapshot['files']]
        for node in workflow['custom_nodes']:
            if node['name']=='ComfyUI-10SOR-Regeneration-Gate':
                assert node['ref']=='2180872a3fcf8bc5f7a9626aeaecc33fb79bebd3'
                assert node['required_node_types']==['10SORRegenerationGate','10SORRegenerationControlPanel']
                node.pop('ref');node.pop('required_node_types')
        assert hashlib.sha256(json.dumps(workflow,sort_keys=True).encode()).hexdigest()==snapshot['sha256'],identifier


def test_hf_and_civitai_transfer_implementations_unchanged():
    tree=ast.parse((ROOT/'launcher/app.py').read_text())
    functions={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
               for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
               and n.name in BASELINE['download_functions']}
    assert functions==BASELINE['download_functions']
    assert hashlib.sha256((ROOT/'launcher/civitai_download.py').read_bytes()).hexdigest()==BASELINE['civitai_sha256']


def test_new_resources_are_default_selected_sanitized_and_isolated():
    workflows=app.load_catalog()['workflows'];count=0
    for workflow in workflows:
        added=workflow['files'][BASELINE['workflows'][workflow['id']]['files']:] if workflow['id'] in BASELINE['workflows'] else workflow['files']
        count+=len(added)
        if workflow['id'] not in BASELINE['workflows']:
            assert workflow['runtime_profile']==app.ISOLATED_PROFILE
            assert not workflow.get('update_comfyui')
            assert 'manager_security_level' not in workflow
        for file in added:
            assert file['default_selected'] is True
            url=urlsplit(file['url']);assert not url.username and not url.password
            assert not {'token','key','api_key','apikey'} & {k.lower() for k,v in parse_qsl(url.query)}
            if file['auth']=='civitai':assert url.hostname=='civitai.com'
            if url.hostname=='huggingface.co':assert len(url.path.split('/resolve/')[1].split('/')[0])==40
            if Path(file['destination']).name in {'MysticXXX_MMH3-V5.safetensors','Kurvy_Krea_2_V1.safetensors'}:
                assert file['destination'].startswith('models/loras/')
    assert count==37


def test_estimates_follow_default_selection_and_exclude_unchecked_krea_raw():
    for workflow in app.public_catalog()['workflows']:
        estimate=workflow['download_estimates']
        total=sum(f['size_bytes'] for f in workflow['resources'] if f['selected_by_default'])
        assert estimate['total_bytes']==total
        assert estimate['default_minutes']==max(1,(total+57_000_000*60-1)//(57_000_000*60))
        assert estimate['transfer_only'] is True
