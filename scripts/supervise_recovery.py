"""Foreground finite-job supervisor, records real exit including native teardown.

No session detachment/restart/automation. An explicit stop is forwarded to this
child only; a killed supervisor leaves exit unknown rather than inventing one.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import psutil
from p57.dataset import ROOT
from p57.durability import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['A','C'],required=True);p.add_argument('--run-dir',required=True,type=Path);args=p.parse_args()
    run=args.run_dir.absolute();logs=run.parent/(run.name+'.supervision')
    if run.exists() or logs.exists():p.error('Occupied run/supervision path; preserve evidence and version explicitly')
    logs.mkdir(parents=True,exist_ok=False)
    cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train_recovery.py'),'--protocol',str(ROOT/'configs/main_ac_recovery_v2.json'),'--model',args.model,'--run-dir',str(run)]
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONHASHSEED':'42','PYTORCH_ENABLE_MPS_FALLBACK':'0','YOLO_AUTOINSTALL':'false','YOLO_CONFIG_DIR':str(ROOT/'.cache/ultralytics'),'MPLCONFIGDIR':str(ROOT/'.cache/matplotlib')}
    start=time.monotonic();child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    metadata={'supervisor_pid':os.getpid(),'child_pid':child.pid,'command':cmd,'environment':{k:env[k] for k in ['PYTHONPATH','PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','YOLO_AUTOINSTALL','YOLO_CONFIG_DIR','MPLCONFIGDIR']},'start_epoch_time':time.time(),'detached':False}
    atomic_json(logs/'process.json',metadata)
    def forward(signum,frame):
        atomic_json(logs/'stop_signal.json',{'signal':signum,'at':time.time(),'child_pid':child.pid})
        if child.poll() is None:child.send_signal(signum)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,forward)
    def pump(pipe,path,output):
        with path.open('xb') as log:
            while True:
                data=pipe.read1(65536)
                if not data:break
                log.write(data);log.flush()
                output.buffer.write(data);output.flush()
            os.fsync(log.fileno())
    threads=[threading.Thread(target=pump,args=(child.stdout,logs/'stdout.log',sys.stdout)),threading.Thread(target=pump,args=(child.stderr,logs/'stderr.log',sys.stderr))]
    for thread in threads:thread.start()
    with (logs/'resources.jsonl').open('x') as observations:
        while child.poll() is None:
            try:rss=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:rss=None
            observed={'time':time.time(),'elapsed':time.monotonic()-start,'child_pid':child.pid,'rss_bytes':rss,'available_bytes':psutil.virtual_memory().available,'global_swap_bytes':psutil.swap_memory().used,'disk_free_bytes':psutil.disk_usage(ROOT).free}
            observations.write(json.dumps(observed)+'\n');observations.flush();os.fsync(observations.fileno())
            atomic_json(logs/'heartbeat.json',observed)
            time.sleep(2)
    code=child.wait()
    for thread in threads:thread.join()
    atomic_json(logs/'exit.json',{'child_pid':child.pid,'returncode':code,'elapsed_seconds_including_native_cleanup':time.monotonic()-start,'end_epoch_time':time.time(),'no_restart':True})
    raise SystemExit(code if code>=0 else 128-code)
