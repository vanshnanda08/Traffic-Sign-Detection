"""Run exactly one frozen W5 detector. No overrides/resume/test entry points."""
import argparse
import contextlib
import os
from pathlib import Path
import sys
TASK_ROOT=Path(__file__).resolve().parents[1]
for key,relative in [("YOLO_CONFIG_DIR",".cache/ultralytics"),("MPLCONFIGDIR",".cache/matplotlib")]:
    os.environ[key]=str(TASK_ROOT/relative)
from ultralytics.utils import LOGGER
from p57.dataset import ROOT
from p57.main_training import run_main

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--model',choices=['A','C'],required=True);p.add_argument('--run-dir',required=True,type=Path)
    args=p.parse_args()
    if args.run_dir.exists():p.error('Existing run directory: refusing overwrite')
    for key,relative in [('YOLO_CONFIG_DIR','.cache/ultralytics'),('MPLCONFIGDIR','.cache/matplotlib')]:
        os.environ[key]=str(ROOT/relative);(ROOT/relative).mkdir(parents=True,exist_ok=True)
    log_path=args.run_dir.with_suffix('.console.log');log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('x') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        for handler in LOGGER.handlers:
            if hasattr(handler,'setStream'):handler.setStream(sys.stderr)
        run_main(args.protocol,args.model,args.run_dir)

if __name__=='__main__':main()
