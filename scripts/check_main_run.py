"""Check one completed run; optional separate clean-validation gallery creation."""
import argparse
from pathlib import Path
from p57.main_verification import verify_single, validation_gallery

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--gallery',action='store_true');args=p.parse_args()
    verify_single(args.run_dir)
    if args.gallery:validation_gallery(args.run_dir)
    print('Passed: completed main run, clean-validation scope only')
