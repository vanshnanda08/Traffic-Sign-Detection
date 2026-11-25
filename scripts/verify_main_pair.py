"""Read-only recheck of retained A/C evidence (writes only an explicit report)."""
import argparse
from pathlib import Path
from p57.dataset import save_json
from p57.main_verification import verify_pair

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',required=True,type=Path);p.add_argument('--c',required=True,type=Path);p.add_argument('--report',required=True,type=Path);args=p.parse_args()
    result=verify_pair(args.a,args.c);save_json(args.report,result);print(result)
