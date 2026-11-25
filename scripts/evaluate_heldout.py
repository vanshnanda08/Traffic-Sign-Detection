"""Explicit gated generation or one frozen held-out evaluation; never training."""
import os,argparse,threading,time,traceback,signal
from pathlib import Path
TASK_ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(TASK_ROOT/'.cache/ultralytics');os.environ['MPLCONFIGDIR']=str(TASK_ROOT/'.cache/matplotlib')
import torch
from p57.dataset import ROOT
from p57.evaluation_data import config,AuthInputs,generate_conditions
from p57.evaluation_protocol import verify_freeze
from p57.evaluation_engine import execute_evaluator
from p57.main_training import strict_runtime
from p57.feasibility import memory_sample
from p57.durability import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-heldout',action='store_true');p.add_argument('--generate',action='store_true');p.add_argument('--model',choices=['A','C']);p.add_argument('--condition',choices=config()['conditions']);args=p.parse_args()
    if not args.execute_heldout:p.error('Explicit --execute-heldout required; no final-test work performed')
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Explicit signal {signum}; no automatic retry')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupt)
    verify_freeze()
    if args.generate:
        manifest=generate_conditions(True);print('Verified315 paired source-condition images and labels');raise SystemExit(0)
    if not args.model or not args.condition:p.error('Specify model and condition')
    strict_runtime();torch.set_num_threads(2);cfg=config();out=ROOT/cfg['output_root']/'runs'/f'{args.model}_{args.condition}'
    if out.exists():raise SystemExit('Output exists: preserve evidence; no silent rerun')
    samples=[];stop=threading.Event();start=time.monotonic()
    def observe():
        while not stop.is_set():samples.append({'elapsed':time.monotonic()-start,**memory_sample()});stop.wait(1)
    thread=threading.Thread(target=observe,daemon=True);thread.start()
    try:
        auth=AuthInputs('heldout',args.condition,True)
        result=execute_evaluator(auth,ROOT/cfg['checkpoints'][args.model]['path'],out)
        atomic_json(out/'status.json',{'status':'passed','model':args.model,'condition':args.condition,'metrics_complete':True})
    except BaseException as error:
        out.mkdir(parents=True,exist_ok=True);atomic_json(out/'status.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc(),'metrics_complete':False});raise
    finally:
        stop.set();thread.join(timeout=3);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'resources.json',{'pid':os.getpid(),'command':__import__('sys').argv,'elapsed_before_native_cleanup':time.monotonic()-start,'observations':samples,'method':'one-second RSS/system available/global swap/MPS tensor+driver observations; not guaranteed peaks, not additive, swap not solely this job'})
