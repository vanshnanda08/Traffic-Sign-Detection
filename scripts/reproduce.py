"""Evaluation-only main-table reproduction. No training entry points imported."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/reproduction_ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/reproduction_matplotlib'))
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import argparse
import json
import platform
import subprocess
import sys
import time
import uuid
import signal
import threading
import psutil
sys.path.insert(0,str(ROOT/'src'))
from p57.reproduction import (authenticate,read,new_directory,build_view,PortableInputs,execute,
                              aggregate_run,CONDITIONS)
from p57.assets import file_sha256
from p57.durability import atomic_json


SUPERVISION_VERSION='w8a.1'


def optional_measurement(getter):
    """Only OS/psutil telemetry failures are optional; programming errors propagate."""
    try:return getter(),None
    except (OSError,psutil.Error) as error:
        return None,{'type':type(error).__name__,'message':str(error)}


def resource_sample(pid,started):
    sample={'seconds':time.monotonic()-started};errors={}
    getters={'rss_bytes':lambda:psutil.Process(pid).memory_info().rss,
             'system_available_bytes':lambda:psutil.virtual_memory().available,
             'global_swap_bytes':lambda:psutil.swap_memory().used}
    for name,getter in getters.items():
        sample[name],error=optional_measurement(getter)
        if error:errors[name]=error
    sample['unavailable']=errors
    return sample


def stop_and_reap(child):
    """Bounded graceful stop followed by kill; always wait for our owned child."""
    if child.poll() is None:
        try:child.terminate()
        except ProcessLookupError:pass
        try:return child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:child.kill()
            except ProcessLookupError:pass
    return child.wait()


def supervise(command,logdir,metadata):
    logdir=new_directory(logdir);started=time.monotonic();wall=time.time()
    atomic_json(logdir/'command.json',{**metadata,'argv':command,'cwd':str(ROOT),
                                      'supervision_version':SUPERVISION_VERSION})
    child=None;code=None;old={};failure=None;interruption=None
    try:
        with (logdir/'stdout.log').open('wb') as stdout,(logdir/'stderr.log').open('wb') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                                   env={**os.environ,'PYTHONHASHSEED':'42','PYTHONPATH':str(ROOT/'src')})
            def forward(signum,frame):
                nonlocal interruption
                if interruption is None:interruption=(signum,time.monotonic())
                if child.poll() is None:
                    try:child.send_signal(signum)
                    except ProcessLookupError:pass
            try:
                for sig in [signal.SIGINT,signal.SIGTERM]:old[sig]=signal.signal(sig,forward)
                with (logdir/'resources.jsonl').open('x') as resource:
                    while child.poll() is None:
                        resource.write(json.dumps(resource_sample(child.pid,started))+'\n');resource.flush()
                        if interruption and time.monotonic()-interruption[1]>=2:
                            # A child ignoring forwarded signals must not keep the supervisor alive.
                            code=stop_and_reap(child);break
                        time.sleep(.2)
                    code=child.wait()
                    resource.flush();os.fsync(resource.fileno())
                for stream in (stdout,stderr):stream.flush();os.fsync(stream.fileno())
                if interruption:raise InterruptedError(f'Supervisor interrupted by signal {interruption[0]}')
                if code!=0:raise RuntimeError(f'Child failed (actual exit{code}); preserved {logdir}')
            finally:
                try:
                    if code is None or child.poll() is None:code=stop_and_reap(child)
                finally:
                    for sig,handler in old.items():signal.signal(sig,handler)
    except BaseException as error:
        # Lifecycle cleanup is mandatory; this exception is re-raised, never treated as success.
        failure={'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        if child is not None:
            event={**metadata,'supervision_version':SUPERVISION_VERSION,'returncode':code,'pid':child.pid,
                   'status':'failed' if failure else 'passed','supervisor_error':failure,
                   'interrupted_signal':interruption[0] if interruption else None,
                   'elapsed_seconds_including_cleanup':time.monotonic()-started,'wall_span_seconds':time.time()-wall}
            # Required evidence writes remain fatal (including failure to write this record).
            atomic_json(logdir/'exit.json',event)
    return event


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--assets',default=str(ROOT/'configs/reproduction_assets.json'))
    p.add_argument('--output')
    p.add_argument('--worker',action='store_true');p.add_argument('--model',choices=['A','C'])
    p.add_argument('--condition',choices=CONDITIONS);args=p.parse_args()
    cfg,m,core,models,versions=authenticate(ROOT,args.assets)
    if args.worker:
        if not args.output or not args.model or not args.condition:p.error('Worker requires authenticated output/model/condition')
        run=Path(args.output);gate=read(run/'reproduction_freeze.json')
        if gate['config_sha256']!=file_sha256(ROOT/'configs/reproduction_v1.json') or gate['manifest_sha256']!=cfg['manifest_sha256']:
            raise ValueError('Reproduction freeze drift')
        for name,wanted in gate['source_sha256'].items():
            if file_sha256(ROOT/name)!=wanted:raise ValueError('Reproduction code changed after freeze')
        auth=PortableInputs(run/'views'/args.condition,{int(k):v for k,v in m['classes'].items()},m['splits']['test'],args.condition,expected_conditions=m['condition_rows'])
        execute(auth,models[args.model],run/'runs'/f'{args.model}_{args.condition}',args.device,m,cfg)
        print('Completed fresh model-source evaluation;45sources; model',args.model,'condition',args.condition,flush=True)
        return 0
    run=Path(args.output) if args.output else ROOT/'results/reproduction_runs'/('run_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    run=run.absolute();new_directory(run)
    # Only require the small development verification receipt, never old training stores.
    preflight=ROOT/'docs/reproduction_v1/development_preflight.json'
    dev=read(preflight)
    if dev['status']!='passed' or dev['config_sha256']!=file_sha256(ROOT/'configs/reproduction_v1.json'):
        raise ValueError('Development-only portability gate not passed for current config')
    if dev['implementation_sha256']!=file_sha256(ROOT/'src/p57/reproduction.py'):raise ValueError('Development preflight implementation drift')
    if args.device not in dev['supported_devices']:raise ValueError('Requested backend not verified in development')
    files=['src/p57/reproduction.py','scripts/reproduce.py','run_all.sh','configs/reproduction_v1.json','docs/reproduction_v1/release_manifest.json','docs/reproduction_v1/development_preflight.json']
    memory_bytes,memory_error=optional_measurement(lambda:psutil.virtual_memory().total)
    gate={'supervision_version':SUPERVISION_VERSION,'config_sha256':file_sha256(ROOT/'configs/reproduction_v1.json'),'manifest_sha256':cfg['manifest_sha256'],
          'source_sha256':{name:file_sha256(ROOT/name) for name in files},'checkpoints':{k:file_sha256(p) for k,p in models.items()},
          'device':args.device,'seed':42,'versions':versions,'hardware':{'platform':platform.platform(),'machine':platform.machine(),'memory_bytes':memory_bytes,'memory_unavailable':memory_error},
          'command':sys.argv,'fresh_inference':True,'historical_results_reference_only':True}
    atomic_json(run/'reproduction_freeze.json',gate)
    print(json.dumps(gate,indent=2),flush=True)
    started=time.monotonic();events=[]
    try:
        # Same lock file as training, without importing/revalidating historic training code.
        import fcntl
        lockpath=ROOT/'.cache/p57_training.lock';lockpath.parent.mkdir(exist_ok=True)
        with lockpath.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            names={int(k):v for k,v in m['classes'].items()}
            for condition in CONDITIONS:
                build_view(run/'views'/condition,m['splits']['test'],names,core,condition,m['condition_rows'])
            atomic_json(run/'generation_receipt.json',{'sources_per_condition':45,'image_label_pairs':315,'conditions':CONDITIONS,
                       'each_view_matches_W6_file_and_decoded_pixel_hashes':True,'unchanged_labels':True})
            for condition in CONDITIONS:
                for model in ['A','C']:
                    key=f'{model}_{condition}';print('Starting',key,flush=True)
                    command=[sys.executable,str(ROOT/'scripts/reproduce.py'),'--worker','--output',str(run),'--device',args.device,'--assets',str(Path(args.assets).absolute()),'--model',model,'--condition',condition]
                    event=supervise(command,run/'logs'/key,{'model':model,'condition':condition});events.append(event)
                    atomic_json(run/'progress.json',{'completed':events,'expected':14,'status':'in_progress'})
            comparison=aggregate_run(run,m,cfg,args.device)
        verdict='passed' if comparison['all_within_predeclared_tolerances'] else 'comparison_failed'
        atomic_json(run/'status.json',{'status':verdict,'fresh_evaluations':14,'model_source_evaluations':630,
                    'actual_child_exits':[e['returncode'] for e in events],'elapsed_seconds_including_children_cleanup':time.monotonic()-started,
                    'comparison':comparison,'device':args.device,'second_machine':False})
        print('REPRODUCTION',verdict,'OUTPUT',run,flush=True)
        return 0 if verdict=='passed' else 1
    except BaseException as error:
        atomic_json(run/'status.json',{'status':'failed','error':str(error),'completed_evaluations':len(events),
                                      'actual_child_exits':[e['returncode'] for e in events],
                                      'elapsed_seconds_including_children_cleanup':time.monotonic()-started})
        raise


if __name__=='__main__':raise SystemExit(main())
