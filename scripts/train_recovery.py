"""One foreground recovery child; no resume, fallback, or training overrides."""
import argparse
import os
from pathlib import Path
import signal
ROOT_TASK=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT_TASK/'.cache/ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT_TASK/'.cache/matplotlib')
from p57.recovery_training import run_recovery


def interrupted(signum,frame):raise KeyboardInterrupt(f'Explicit signal {signum}; preserve incomplete run, do not restart')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--model',choices=['A','C'],required=True);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
    run_recovery(args.protocol,args.model,args.run_dir)
