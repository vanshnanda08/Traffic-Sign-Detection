"""All45 clean validation sources only; historical A is a preflight asset."""
import argparse
import os
from pathlib import Path
ROOT_TASK=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT_TASK/'.cache/ultralytics');os.environ['MPLCONFIGDIR']=str(ROOT_TASK/'.cache/matplotlib')
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from p57.dataset import ROOT
from p57.complete_validation import CompleteValidator
from p57.online import ScheduledYOLODataset
from p57.main_training import strict_runtime
from p57.recovery_protocol import load_recovery
from p57.durability import atomic_json,Journal
from p57.assets import file_sha256

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    protocol,*_=load_recovery(ROOT/'configs/main_ac_recovery_v2.json','A');strict_runtime()
    args.output.mkdir(parents=True,exist_ok=False)
    hyp=get_cfg();hyp.imgsz=416
    dataset=ScheduledYOLODataset(view_yaml=ROOT/protocol['training']['data'],mode='val',imgsz=416,batch_size=8,hyp=hyp,rect=True)
    loader=build_dataloader(dataset,batch=8,workers=0,shuffle=False,rank=-1)
    settings={**protocol['training'],'data':str(ROOT/protocol['training']['data']),'device':'mps','batch':8,'plots':False,'save_json':False,'save_txt':False,'half':False,'split':'val','task':'detect','mode':'val'}
    validator=CompleteValidator(loader,save_dir=args.output,args=settings)
    journal=Journal(args.output/'events.jsonl');validator.journal=journal
    checkpoint=ROOT/'results/main_ac_v1/A/fit/weights/best.pt'
    try:
        metrics=validator(model=str(checkpoint))
        atomic_json(args.output/'result.json',{'status':'passed','scope':'Recovery completeness preflight on all45 clean validation images; does not retrospectively certify historical selection','checkpoint_sha256':file_sha256(checkpoint),'metrics_clean_validation_only':metrics,'completion':validator.last_completion})
        print('Preflight passed all45 clean validation sources')
    finally:journal.close()
