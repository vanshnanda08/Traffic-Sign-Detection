"""Authorized three-epoch trial; train and validation only."""

import argparse
import contextlib
import sys
from pathlib import Path

from ultralytics.utils import LOGGER

from p57.dataset import ROOT
from p57.feasibility import run_trial, validation_gallery
from p57.environment import seed_everything


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["train","infer"])
    parser.add_argument("--config",default="configs/feasibility_v1.json")
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--gallery-name",default="validation_gallery_recheck")
    args=parser.parse_args()
    if args.mode=="train":
        if args.run_dir.exists():raise SystemExit("Run directory exists; choose a separate version")
        log_path=args.run_dir.with_suffix(".console.log")
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open("x") as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            for handler in LOGGER.handlers:
                if hasattr(handler,"setStream"):handler.setStream(sys.stderr)
            run_trial(ROOT/args.config,args.run_dir)
    else:
        seed_everything(42,True)
        validation_gallery(args.run_dir,args.run_dir/"fit/weights/best.pt",args.gallery_name)
