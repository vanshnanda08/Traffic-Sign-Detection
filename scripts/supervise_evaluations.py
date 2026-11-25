"""Sequential foreground14-run evaluator, with real stdout/stderr/exits/cleanup."""
import os,json,subprocess,sys,time,threading,signal,psutil
from pathlib import Path
from p57.dataset import ROOT
from p57.evaluation_data import config
from p57.evaluation_protocol import verify_freeze
from p57.durability import atomic_json
from p57.protocol import training_job_lock

if __name__=='__main__':
    if sys.argv[1:]!=['--execute-heldout']:raise SystemExit('Explicit --execute-heldout required')
    verify_freeze();cfg=config();parent=ROOT/cfg['output_root'];summary=[]
    with training_job_lock():
        for condition in cfg['conditions']:
            for model in ['A','C']:
                key=f'{model}_{condition}';logs=parent/'supervision'/key
                if logs.exists() or (parent/'runs'/key).exists():raise SystemExit('Occupied evaluation path; inspect/version explicitly, never delete')
                logs.mkdir(parents=True);cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/evaluate_heldout.py'),'--execute-heldout','--model',model,'--condition',condition]
                env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONHASHSEED':'42','PYTORCH_ENABLE_MPS_FALLBACK':'0','YOLO_AUTOINSTALL':'false'}
                started=time.monotonic();wall=time.time();child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                atomic_json(logs/'process.json',{'pid':child.pid,'parent_pid':os.getpid(),'command':cmd,'start_epoch_time':wall,'detached':False})
                def forward(signum,frame):
                    if child.poll() is None:child.send_signal(signum)
                for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,forward)
                def pump(pipe,path,dest):
                    with path.open('xb') as stream:
                        while True:
                            data=pipe.read1(65536)
                            if not data:break
                            stream.write(data);stream.flush();dest.buffer.write(data);dest.flush()
                        os.fsync(stream.fileno())
                threads=[threading.Thread(target=pump,args=(child.stdout,logs/'stdout.log',sys.stdout)),threading.Thread(target=pump,args=(child.stderr,logs/'stderr.log',sys.stderr))]
                for thread in threads:thread.start()
                with (logs/'resources.jsonl').open('x') as resource_log:
                    while child.poll() is None:
                        try:rss=psutil.Process(child.pid).memory_info().rss
                        except psutil.NoSuchProcess:rss=None
                        observed={'epoch_time':time.time(),'monotonic_seconds':time.monotonic()-started,'rss_bytes':rss,'available_bytes':psutil.virtual_memory().available,'global_swap_bytes':psutil.swap_memory().used,'disk_free_bytes':psutil.disk_usage(ROOT).free}
                        resource_log.write(json.dumps(observed)+'\n');resource_log.flush();os.fsync(resource_log.fileno())
                        atomic_json(logs/'heartbeat.json',observed);time.sleep(1)
                code=child.wait()
                for thread in threads:thread.join()
                event={'model':model,'condition':condition,'returncode':code,'monotonic_seconds_including_cleanup':time.monotonic()-started,'wall_span_seconds':time.time()-wall}
                atomic_json(logs/'exit.json',event);summary.append(event);atomic_json(parent/'progress.json',{'completed':summary,'expected':14,'status':'in_progress' if code==0 else 'failed'})
                if code!=0:raise SystemExit(code if code>0 else 128-code)
        atomic_json(parent/'progress.json',{'completed':summary,'expected':14,'status':'passed'});print('All14 evaluations exited0 through cleanup')
