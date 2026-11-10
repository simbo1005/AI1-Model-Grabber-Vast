import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from launcher import app, hf_cache, warmup


def workflow():
    return next(w for w in app.load_catalog()['workflows'] if w['id']=='snn-krast')


def test_warmup_jobs_use_disposable_inputs_and_selected_precisions():
    graphs = dict(warmup.jobs())
    assert len(graphs) == 4
    for graph in graphs.values():
        assert graph['1']['inputs']['image'] == 'studio_placeholder.png'
        assert all(n['class_type'] not in {'SaveImage', 'BondSaveWithCustomMetadata', 'FolderImageLoader_BFT'} for n in graph.values())
        assert all(n['inputs'].get('steps', 1)==1 for n in graph.values())
    assert graphs['KRAST BF16']['3']['inputs']['weight_dtype']=='default'
    assert 'bf16' in graphs['KRAST BF16']['4']['inputs']['clip_name']
    assert graphs['SeedVR2']['3']['inputs']['attention_mode']=='sdpa'
    assert graphs['SeedVR2']['3']['inputs']['cache_model'] is False
    assert all(warmup.model_names(g) <= {Path(f['destination']).name for f in workflow()['files']} for g in graphs.values())


def test_loras_are_warmed_only_when_selected():
    graph = dict(warmup.jobs())['KRAST BF16']
    before=copy.deepcopy(graph)
    warmup.add_selected_loras(graph, {'KNP_000003000.safetensors'}, warmup.LORAS['KRAST BF16'])
    applied = [n for n in graph.values() if n['class_type']=='LoraLoaderModelOnly']
    assert len(applied)==1
    assert applied[0]['inputs']['lora_name']=='KNP_000003000.safetensors'
    assert graph['9']['inputs']['model'] != before['9']['inputs']['model']


@pytest.mark.parametrize('busy', [False, True])
def test_warmup_waits_for_execution_completion_and_never_takes_over_queue(busy):
    calls=[]
    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path=='/queue':
            return httpx.Response(200,json={'queue_running':[[0,'user-job']] if busy else [],'queue_pending':[]})
        if request.url.path=='/prompt':
            payload=json.loads(request.content)
            assert payload['client_id'].startswith('dsnn-warmup-')
            return httpx.Response(200,json={'prompt_id':'own-job','node_errors':{}})
        if request.url.path=='/history/own-job':
            return httpx.Response(200,json={'own-job':{'status':{'completed':True,'status_str':'success'}}})
        pytest.fail('Unexpected destructive request: '+request.url.path)
    warnings=[]
    async def run():
        async with httpx.AsyncClient(base_url='http://comfy',transport=httpx.MockTransport(handler)) as client:
            return await warmup.run_warmup('http://comfy',Path('/unused'),workflow(),lambda:None,lambda x:None,warnings.append,client=client)
    assert asyncio.run(run()) == ('deferred' if busy else 'complete')
    assert sum(path=='/prompt' for _,path in calls)==(0 if busy else 4)
    assert not any(path=='/interrupt' for _,path in calls)


def test_validation_errors_do_not_claim_warmup_completed():
    def handler(request):
        if request.url.path=='/queue':return httpx.Response(200,json={})
        return httpx.Response(400,json={'error':{'message':'missing input'},'node_errors':{'5':{}}})
    warnings=[]
    async def run():
        async with httpx.AsyncClient(base_url='http://comfy',transport=httpx.MockTransport(handler)) as client:
            return await warmup.run_warmup('',Path('/unused'),workflow(),lambda:None,lambda x:None,warnings.append,client=client)
    assert asyncio.run(run())=='partial'
    assert len(warnings)==4


@pytest.mark.parametrize('queued', [False, True])
def test_cancellation_cleans_only_our_warmup_job(queued):
    count=0
    requests=[]
    def cancelled():
        nonlocal count
        count+=1
        if count>1:raise app.InstallCancelled()
    def handler(request):
        requests.append((request.url.path,json.loads(request.content) if request.content else None))
        if request.url.path=='/prompt':return httpx.Response(200,json={'prompt_id':'own-job'})
        if request.url.path=='/queue' and request.method=='GET':
            return httpx.Response(200,json={} if len(requests)==1 else {'queue_running':[[1,'other-job']] if queued else [[0,'own-job']], 'queue_pending':[[2,'own-job']] if queued else [[3,'other-job']]})
        return httpx.Response(200,json={})
    async def run():
        async with httpx.AsyncClient(base_url='http://comfy',transport=httpx.MockTransport(handler)) as client:
            await warmup.run_warmup('',Path('/unused'),workflow(),cancelled,lambda x:None,lambda x:None,client=client)
    with pytest.raises(app.InstallCancelled):asyncio.run(run())
    mutation=[(p,j) for p,j in requests if p in {'/interrupt','/queue'} and j is not None]
    assert mutation == ([('/queue',{'delete':['own-job']})] if queued else [('/interrupt',{'prompt_id':'own-job'})])


def test_depth_snapshot_is_offline_and_does_not_move_the_cli_files(tmp_path):
    root=tmp_path/'ComfyUI'; local=root/'models/annotators/depth'
    local.mkdir(parents=True)
    for name in ['config.json','model.safetensors']:(local/name).write_bytes(b'fixture')
    spec={'repo':'LiheYoung/depth-anything-large-hf','revision':'a'*40,'directory':'models/annotators/depth','files':['config.json','model.safetensors']}
    cache=tmp_path/'hub'
    hf_cache.seed(root,[spec],cache)
    store=cache/'models--LiheYoung--depth-anything-large-hf'
    assert (store/'refs/main').read_text()=='a'*40
    for name in spec['files']:
        target=store/'snapshots'/('a'*40)/name
        assert target.is_symlink() and target.read_bytes()==b'fixture'
        assert (local/name).is_file()
    spec['files'].append('../escape')
    with pytest.raises(RuntimeError):hf_cache.seed(root,[spec],cache)


def test_sage_source_patch_is_fail_closed_and_hopper_only(tmp_path):
    path=Path(__file__).parents[1]/'docker/attention/build.py'
    spec=importlib.util.spec_from_file_location('sage_build',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(RuntimeError,match='source changed'):module.patch_source('changed upstream')
    source='extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},\n                extra_link_args=[\'-lcuda\'],'
    patched=module.patch_source(source)
    assert 'arch=compute_90a,code=sm_90a' in patched
    assert set(module.ARCHES.split(';')) >= {'8.0','8.6','8.9','9.0','12.0'}


def test_old_images_fail_clearly_before_downloading(monkeypatch):
    controller=app.JobController()
    async def wait():pytest.fail('An old image must fail before waiting on ComfyUI or downloading')
    monkeypatch.setattr(controller,'_wait_for_comfyui',wait)
    with pytest.raises(RuntimeError,match='new SageAttention image'):
        asyncio.run(controller._install_workflow(workflow()))


def test_all_62_source_node_types_are_checked_at_installation():
    root=Path(__file__).parents[1]
    expected=json.loads((root/'catalog/snn-krast-faceswap-requirements.json').read_text())
    plan=workflow()
    actual=set(plan['required_core_node_types'])
    actual.update(t for n in plan['custom_nodes'] for t in n['required_node_types'])
    assert len(expected['required_node_types'])==62
    assert set(expected['required_node_types']) <= actual
    assert all(len(n.get('ref', ''))==40 for n in plan['custom_nodes'] if n.get('source')!='image_bundle')
    assert plan['comfyui_ref']=='989d93d0d968e704c589ce0146bb18c77f6b50cc'


def test_local_attention_wheel_remains_version_constrained(tmp_path):
    from launcher.node_setup import install_requirements
    seen=[]
    async def run(*command):
        if "freeze" in command:return 0, "torch==2.10.0+cu128\nsageattention @ file:///opt/dsnn-attention/sage.whl\ntriton==3.6.0"
        if "-c" in command:return 0, "sageattention==2.2.0\ntriton==3.6.0\n"
        constraint=Path(command[command.index('--constraint')+1]).read_text()
        seen.append(constraint)
        return 0, "installed"
    asyncio.run(install_requirements(tmp_path/'requirements.txt',Path('/comfy/python'),run,lambda _:False))
    assert len(seen)==1
    assert 'sageattention==2.2.0' in seen[0]
    assert 'torch==2.10.0+cu128' in seen[0]
