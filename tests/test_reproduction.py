"""Portable contract and failure gates; no held-out inference in tests."""
import json,sys,subprocess,importlib.util,shutil,hashlib
from pathlib import Path
import pytest
from p57.reproduction import asset_path,new_directory,aggregate_run,ROOT,read,library_hashes,authenticate
@pytest.mark.parametrize('value',['../outside','https://unsafe/file.zip','safe/../outside','bad\\file'])
def test_unsafe_paths_rejected(tmp_path,value):
    with pytest.raises(ValueError):asset_path(tmp_path,value)
def test_symlink_assets_rejected(tmp_path):
    (tmp_path/'original').mkdir();(tmp_path/'linked').symlink_to(tmp_path/'original',target_is_directory=True)
    with pytest.raises(ValueError):asset_path(tmp_path,'linked/model.pt')
def test_occupied_output_protection(tmp_path):
    path=new_directory(tmp_path/'out');(path/'evidence').write_text('preserve')
    with pytest.raises(FileExistsError):new_directory(path)
    assert (path/'evidence').read_text()=='preserve'
def test_missing_corrupt_checkpoint_and_inputs_rejected(tmp_path):
    source=read(ROOT/'configs/reproduction_assets.json');cp=tmp_path/'bad.pt';ap=tmp_path/'assets.json'
    for content in [None,b'not canonical']:
        if content:cp.write_bytes(content)
        ap.write_text(json.dumps({**source,'checkpoints':{**source['checkpoints'],'A':str(cp)}}))
        with pytest.raises(ValueError,match='checkpoint'):authenticate(ROOT,ap)
    ap.write_text(json.dumps({**source,'core':str(tmp_path/'absent_core')}))
    with pytest.raises(ValueError,match='input'):authenticate(ROOT,ap)
def test_library_and_setting_identity():
    cfg=read(ROOT/'configs/reproduction_v1.json');m=read(ROOT/cfg['manifest'])
    assert library_hashes()==m['library_hashes']
    assert m['settings']['imgsz']==416 and m['settings']['batch']==4 and m['settings']['conf']==.001 and m['settings']['iou']==.7
    assert len(m['splits']['test'])==45 and len(m['condition_rows'])==315 and len(m['classes'])==6
def test_wrong_classes_manifest_rejected(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'docs/dataset_v2').mkdir(parents=True)
    cfg=read(ROOT/'configs/reproduction_v1.json');m=read(ROOT/cfg['manifest']);m['classes']['0']='WRONG'
    target=tmp_path/cfg['manifest'];target.parent.mkdir(parents=True);target.write_text(json.dumps(m))
    cfg['manifest_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();(tmp_path/'configs/reproduction_v1.json').write_text(json.dumps(cfg))
    shutil.copy2(ROOT/'configs/evaluation_v1.json',tmp_path/'configs/evaluation_v1.json')
    shutil.copy2(ROOT/'docs/dataset_v2/dataset.lock.json',tmp_path/'docs/dataset_v2/dataset.lock.json')
    with pytest.raises(ValueError,match='Class'):authenticate(tmp_path)
def test_incomplete_run_never_generates_table(tmp_path):
    cfg=read(ROOT/'configs/reproduction_v1.json');m=read(ROOT/cfg['manifest'])
    with pytest.raises(FileNotFoundError):aggregate_run(tmp_path,m,cfg,'cpu')
    assert not (tmp_path/'artifacts').exists()
def test_failure_exit_record_propagation(tmp_path):
    spec=importlib.util.spec_from_file_location('repro_runner',ROOT/'scripts/reproduce.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(RuntimeError,match='actual exit7'):module.supervise([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'logs',{'model':'fixture'})
    assert read(tmp_path/'logs/exit.json')['returncode']==7
def test_default_runner_missing_assets_does_not_train(tmp_path):
    p=tmp_path/'assets.json';p.write_text(json.dumps({'core':'missing','checkpoints':{'A':'missing.pt','C':'missing.pt'}}))
    run=subprocess.run(['bash',str(ROOT/'run_all.sh'),'--reproduce','--assets',str(p),'--output',str(tmp_path/'run')],capture_output=True,text=True)
    assert run.returncode!=0 and 'checkpoint' in run.stderr and not (tmp_path/'run').exists()

def test_wrong_source_membership_and_label_mapping_rejected(tmp_path):
    import yaml
    from p57.reproduction import PortableInputs
    view=tmp_path/'view';images=view/'val/images';labels=view/'val/labels';images.mkdir(parents=True);labels.mkdir()
    names={i:str(i) for i in range(6)};rows=[];generation=[]
    for i in range(45):
        sid=f'source{i:02}';im=images/(sid+'_hash.jpg');lab=labels/(sid+'_hash.txt');im.write_bytes(b'synthetic not decoded in this membership gate');lab.write_text('0 .5 .5 .2 .2\n')
        h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({'source_id':sid,'image_path':'val/images/'+im.name,'label_sha256':h(lab)})
        generation.append({'source_id':sid,'image':'val/images/'+im.name,'image_sha256':h(im),'label_sha256':h(lab)})
    (view/'dataset.yaml').write_text(yaml.safe_dump({'train':None,'val':str(images.resolve()),'test':None,'nc':6,'names':names}))
    (view/'generation.json').write_text(json.dumps(generation))
    PortableInputs(view,names,rows,'clean','val')
    generation[0]['source_id']='foreign';(view/'generation.json').write_text(json.dumps(generation))
    with pytest.raises(ValueError,match='source'):PortableInputs(view,names,rows,'clean','val')
    generation[0]['source_id']=rows[0]['source_id'];generation[0]['label_sha256']='0'*64;(view/'generation.json').write_text(json.dumps(generation))
    with pytest.raises(ValueError,match='pairing'):PortableInputs(view,names,rows,'clean','val')


@pytest.fixture
def supervisor_case(tmp_path,monkeypatch,request):
    """Retain genuine child logs when explicitly requested by the verification command."""
    import os,signal
    spec=importlib.util.spec_from_file_location('repro_runner',ROOT/'scripts/reproduce.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=Path(os.environ.get('P57_SUPERVISOR_EVIDENCE_DIR',tmp_path))
    logdir=base/request.node.name
    spawned=[];original=module.subprocess.Popen
    def launch(*args,**kwargs):
        child=original(*args,**kwargs);spawned.append(child);return child
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    yield module,logdir,spawned
    assert {sig:signal.getsignal(sig) for sig in handlers}==handlers
    for child in spawned:
        try:
            assert child.returncode is not None and child.poll() is not None
            with pytest.raises(ChildProcessError):os.waitpid(child.pid,os.WNOHANG)
        finally:
            if child.poll() is None:child.kill();child.wait()


def unavailable_counter():raise OSError('controlled optional counter unavailable')


@pytest.mark.parametrize('code',[0,7])
def test_swap_unavailable_preserves_real_exit(supervisor_case,monkeypatch,code):
    module,logs,children=supervisor_case
    from types import SimpleNamespace
    monkeypatch.setattr(module.psutil,'swap_memory',unavailable_counter)
    monkeypatch.setattr(module.psutil,'Process',lambda pid:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=456)))
    monkeypatch.setattr(module.psutil,'virtual_memory',lambda:SimpleNamespace(available=789))
    command=[sys.executable,'-c',f'import time;time.sleep(.45);raise SystemExit({code})']
    if code:
        with pytest.raises(RuntimeError,match='actual exit7'):module.supervise(command,logs,{})
    else:assert module.supervise(command,logs,{})['status']=='passed'
    event=read(logs/'exit.json');samples=[json.loads(row) for row in (logs/'resources.jsonl').read_text().splitlines()]
    assert event['returncode']==code and event['pid']==children[0].pid
    assert event['elapsed_seconds_including_cleanup']>=.45 and len(samples)>=2
    assert all(row['global_swap_bytes'] is None and row['unavailable']['global_swap_bytes']['type']=='OSError' for row in samples)
    assert all(row['rss_bytes']==456 and row['system_available_bytes']==789 for row in samples)
    assert (logs/'stderr.log').read_text()==''


def test_rss_and_system_unavailable_independently(supervisor_case,monkeypatch):
    from types import SimpleNamespace
    module,logs,_=supervisor_case
    def no_rss(pid):raise module.psutil.AccessDenied(pid)
    monkeypatch.setattr(module.psutil,'Process',no_rss)
    monkeypatch.setattr(module.psutil,'virtual_memory',unavailable_counter)
    monkeypatch.setattr(module.psutil,'swap_memory',lambda:SimpleNamespace(used=123))
    assert module.supervise([sys.executable,'-c','import time;time.sleep(.3)'],logs,{})['returncode']==0
    samples=[json.loads(row) for row in (logs/'resources.jsonl').read_text().splitlines()]
    assert samples
    for row in samples:
        assert row['rss_bytes'] is None and row['system_available_bytes'] is None
        assert row['global_swap_bytes']==123
        assert row['unavailable']['rss_bytes']['type']=='AccessDenied'
        assert row['unavailable']['system_available_bytes']['message']=='controlled optional counter unavailable'


@pytest.mark.parametrize('fault',['required_log','programming_error','interrupt'])
def test_supervisor_error_reaps_child_and_records_failure(supervisor_case,monkeypatch,fault):
    module,logs,_=supervisor_case
    if fault=='required_log':
        original=Path.open
        def failing_open(path,*args,**kwargs):
            if path.name=='resources.jsonl':raise OSError('controlled required logging failure')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',failing_open);error=OSError
    elif fault=='programming_error':
        def bad_counter():raise ValueError('controlled programming error')
        monkeypatch.setattr(module.psutil,'swap_memory',bad_counter);error=ValueError
    else:
        def interrupted_sample(*args):raise KeyboardInterrupt('controlled interruption')
        monkeypatch.setattr(module,'resource_sample',interrupted_sample);error=KeyboardInterrupt
    with pytest.raises(error):module.supervise([sys.executable,'-c','import time;time.sleep(10)'],logs,{})
    event=read(logs/'exit.json')
    assert event['status']=='failed' and event['returncode']<0
    assert event['supervisor_error']['type']==error.__name__
    assert event['elapsed_seconds_including_cleanup']<5


def test_exit_record_write_failure_is_explicit_and_child_reaped(supervisor_case,monkeypatch):
    module,logs,_=supervisor_case;original=module.atomic_json
    def write(path,value):
        if path.name=='exit.json':raise OSError('controlled exit record write failure')
        return original(path,value)
    monkeypatch.setattr(module,'atomic_json',write)
    with pytest.raises(OSError,match='exit record write failure'):
        module.supervise([sys.executable,'-c','import time;time.sleep(.25)'],logs,{})
    assert not (logs/'exit.json').exists()


@pytest.mark.parametrize('ignores_signal',[False,True])
def test_forwarded_signal_is_bounded_and_not_success(supervisor_case,ignores_signal):
    import signal
    module,logs,_=supervisor_case
    # Signal only our current supervisor; its temporary handler forwards to this child.
    handler='signal.SIG_IGN' if ignores_signal else 'lambda *args:sys.exit(0)'
    command=[sys.executable,'-c',f'import os,sys,signal,time;signal.signal(signal.SIGTERM,{handler});time.sleep(.2);os.kill(os.getppid(),signal.SIGTERM);time.sleep(10)']
    with pytest.raises(InterruptedError,match='signal'):module.supervise(command,logs,{})
    event=read(logs/'exit.json')
    assert event['status']=='failed' and event['returncode']==(-signal.SIGKILL if ignores_signal else 0)
    assert event['interrupted_signal']==signal.SIGTERM
    assert .2<=event['elapsed_seconds_including_cleanup']<7
