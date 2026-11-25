"""Read-only recovery verifier; optional separately generated validation gallery."""
import argparse,os
from pathlib import Path
ROOT_TASK=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT_TASK/'.cache/ultralytics');os.environ['MPLCONFIGDIR']=str(ROOT_TASK/'.cache/matplotlib')
from p57.recovery_verification import verify_single,verify_pair,validation_gallery
from p57.durability import atomic_json
from p57.environment import seed_everything

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path);p.add_argument('--gallery',action='store_true');p.add_argument('--a',type=Path);p.add_argument('--c',type=Path);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    seed_everything(42,True)
    if args.run_dir:
        state,accounting,_=verify_single(args.run_dir)
        if args.gallery:validation_gallery(args.run_dir)
        result={'status':'passed','scope':'Recovery complete clean validation and one model; not final test','variant':state['variant'],'selected_epoch':state['selected_epoch'],'checkpoint_hashes':state['checkpoint_hashes'],'budget':{k:accounting[k] for k in ['epochs','optimizer_update_count','image_presentations','final_unstepped_images']}}
    elif args.a and args.c:result=verify_pair(args.a,args.c)
    else:p.error('Specify one run or both A/C')
    atomic_json(args.report,result);print(result)
