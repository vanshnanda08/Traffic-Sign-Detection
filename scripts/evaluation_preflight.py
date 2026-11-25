"""Compare both best checkpoints on development validation only, before freeze."""
import os
from pathlib import Path
TASK_ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(TASK_ROOT/'.cache/ultralytics');os.environ['MPLCONFIGDIR']=str(TASK_ROOT/'.cache/matplotlib')
import json,time
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.evaluation_data import config,CONFIG,AuthInputs
from p57.evaluation_engine import execute_evaluator,require_local_font
from p57.evaluation_protocol import authenticate_models
from p57.complete_validation import CompleteValidator
from p57.online import ScheduledYOLODataset
from p57.main_training import strict_runtime
from p57.feasibility import memory_sample
from p57.protocol import training_job_lock
from p57.durability import atomic_json

class ReferenceObserver(CompleteValidator):
    def update_metrics(self,preds,batch):
        super().update_metrics(preds,batch)
        for i,p in enumerate(preds):self.observed.append({'path':str(Path(batch['im_file'][i]).resolve()),'boxes':p['bboxes'].cpu().numpy(),'classes':p['cls'].cpu().numpy(),'confidence':p['conf'].cpu().numpy()})

if __name__=='__main__':
    authenticate_models();strict_runtime();torch.set_num_threads(2);require_local_font()
    cfg=config();out=ROOT/'results/w6_verification_v1/preflight';out.mkdir(parents=True,exist_ok=False);report={'status':'in_progress','scope':'Development val only; no final-test predictions','config_sha256':file_sha256(CONFIG),'tolerances':cfg['tolerances'],'models':{}}
    auth=AuthInputs('development');started=time.monotonic()
    with training_job_lock():
        for model,checkpoint in cfg['checkpoints'].items():
            settings={**cfg['settings'],'data':str(auth.yaml),'split':'val'}
            ds=ScheduledYOLODataset(view_yaml=auth.yaml,mode='val',imgsz=416,batch_size=4,hyp=get_cfg(overrides=settings),rect=True,stride=32)
            loader=build_dataloader(ds,batch=4,workers=0,shuffle=False,rank=-1)
            reference=ReferenceObserver(loader,save_dir=out/f'{model}_reference',args=settings);reference.observed=[]
            ref_metrics=reference(model=str(ROOT/checkpoint['path']))
            measured=execute_evaluator(auth,ROOT/checkpoint['path'],out/f'{model}_new')
            aliases={'precision':'metrics/precision(B)','recall':'metrics/recall(B)','mAP50':'metrics/mAP50(B)','mAP50_95':'metrics/mAP50-95(B)'}
            diffs={k:abs(float(ref_metrics[v])-measured['aggregate'][k]) for k,v in aliases.items()}
            if max(diffs.values())>cfg['tolerances']['development_metric_absolute']:raise ValueError('Development metric mismatch; tolerance is not widened')
            records=json.loads((out/f'{model}_new/predictions.json').read_text())['records']
            if len(reference.observed)!=45:raise ValueError('Reference coverage incomplete')
            max_error=0.
            for old,new in zip(reference.observed,records):
                if old['path']!=new['image_path']:raise ValueError('Development source order mismatch')
                for before,key in [('boxes','network_xyxy'),('classes','classes'),('confidence','confidence')]:
                    actual=np.asarray(new[key],dtype=np.float32).reshape(old[before].shape)
                    if not np.allclose(old[before],actual,atol=cfg['tolerances']['development_prediction_absolute'],rtol=0):raise ValueError('Prediction stream mismatch')
                    if actual.size:max_error=max(max_error,float(np.max(np.abs(old[before]-actual))))
            report['models'][model]={'status':'passed','metric_differences':diffs,'maximum_prediction_difference':max_error,'reference_complete':reference.last_completion,'new_complete':measured['completion'],'checkpoint_sha256':checkpoint['sha256'],'resources':memory_sample()}
    report.update(status='passed',runtime_seconds=time.monotonic()-started);atomic_json(out/'result.json',report);print('Both development-only comparisons passed')
