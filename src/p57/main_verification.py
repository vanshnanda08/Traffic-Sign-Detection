"""Read-only scientific pair verification and separate validation-only galleries."""
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from p57.assets import file_sha256
from p57.dataset import ROOT, save_json
from p57.development import development_records
from p57.feasibility import validate_detections
from p57.main_training import load_contract, check_exposures, OVERLAY
from p57.protocol import require_equal_budgets


def verify_single(run_dir,require_gallery=False,load_weights=True):
    run_dir=Path(run_dir).absolute()
    state=json.loads((run_dir/'provenance.json').read_text())
    protocol,receipt,overlay,rows,schedule,expected=load_contract('configs/main_ac_frozen_v1.json',state['variant'])
    if state['status']!='passed' or len(state['epochs'])!=50:
        raise ValueError('Run did not pass every main training epoch')
    for key,wanted in [('protocol_sha256',receipt['protocol_sha256']),('exposure_manifest_sha256',receipt['exposure_manifest_sha256']),
                       ('initial_weights_sha256',protocol['initial_weights_sha256']),('dataset_lock_sha256',protocol['dataset_lock_sha256']),
                       ('overlay_sha256',file_sha256(OVERLAY)),('frozen_receipt_sha256',file_sha256(ROOT/'docs/protocol_w4_v1/freeze_receipt.json'))]:
        if state[key]!=wanted:raise ValueError(f'Run hash drift: {key}')
    if not state['actual_device'].startswith('mps') or not state['strict_determinism']:
        raise ValueError('Device/determinism mismatch')
    for key,wanted in protocol['training'].items():
        wanted=str((ROOT/wanted).resolve()) if key in {'data','model'} else wanted
        if state['resolved_config'][key]!=wanted:
            raise ValueError(f'Resolved setting drift: {key}')
    actual=json.loads((run_dir/'actual_accounting.json').read_text())
    require_equal_budgets(actual,actual,expected)
    if actual['microbatch_sizes']!=expected['microbatch_sizes'] or actual['final_unstepped_images']!=42 or state['EMA_updates']!=list(range(1,186)):
        raise ValueError('Microbatch/EMA/tail mismatch')
    exposures=json.loads((run_dir/'exposures.json').read_text())
    check_exposures(exposures,rows,schedule,state['variant'])
    if [r['presentation'] for r in exposures]!=list(range(1,10501)):
        raise ValueError('Presentation indices are not contiguous')
    if state['pending_source_exposures']!=exposures[-42:] or state['optimizer_consumed_presentations']!=10458:
        raise ValueError('Pending/consumed source accounting mismatch')
    rows_csv=list(csv.DictReader((run_dir/'fit/results.csv').open()))
    if len(rows_csv)!=50 or len(state['checkpoint_writes'])!=50:
        raise ValueError('Missing epoch/checkpoint records')
    writes=state['checkpoint_writes']
    running=-math.inf
    for epoch,row in enumerate(writes,1):
        running=max(running,row['fitness'])
        if row['epoch']!=epoch or row['best_written']!=(row['fitness']==running) or row['best_fitness']!=running:
            raise ValueError('Checkpoint fitness/tie provenance mismatch')
        if float(rows_csv[epoch-1]['metrics/mAP50-95(B)'])!=row['fitness']:
            raise ValueError('Selection fitness disagrees with CSV')
    selected=[w for w in writes if w['best_written']][-1]
    if state['selected_epoch']!=selected['epoch'] or state['selected_fitness']!=selected['fitness']:
        raise ValueError('Selected epoch/fitness mismatch')
    for name in ['best.pt','last.pt']:
        path=run_dir/'fit/weights'/name
        if file_sha256(path)!=state['checkpoint_hashes'][name]:
            raise ValueError('Checkpoint bytes changed')
        if load_weights:
            model=YOLO(str(path),task='detect')
            if len(model.names)!=6:
                raise ValueError('Checkpoint classes mismatch')
    if require_gallery:
        gallery=json.loads((run_dir/'validation_gallery/predictions.json').read_text())
        val_rows={r['source_id']:r for r in development_records() if r['split']=='val'}
        if len(gallery['images'])!=6 or gallery['checkpoint_sha256']!=state['checkpoint_hashes']['best.pt']:
            raise ValueError('Validation gallery missing/wrong checkpoint')
        for record in gallery['images']:
            row=val_rows[record['source_id']]
            image_path=ROOT/protocol['training']['data'];image_path=image_path.parent/row['image_path']
            if file_sha256(image_path)!=record['source_sha256']:
                raise ValueError('Gallery source changed')
            validate_detections(np.asarray(record['xyxy']).reshape(-1,4),np.asarray(record['classes']),np.asarray(record['confidence']),tuple(record['shape']),6)
            if not (run_dir/'validation_gallery'/f"{record['source_id']}.jpg").is_file():
                raise ValueError('Gallery rendering missing')
    return state,actual,exposures


def verify_pair(a_dir,c_dir):
    a,ab,ae=verify_single(a_dir,require_gallery=True)
    c,cb,ce=verify_single(c_dir,require_gallery=True)
    if a['variant']!='A' or c['variant']!='C':raise ValueError('Wrong A/C ordering')
    for key in ['effective_initialization_sha256','initial_weights_sha256','protocol_sha256','overlay_sha256','exposure_manifest_sha256','dataset_lock_sha256','source_fingerprints','library_fingerprints','optimizer']:
        if a[key]!=c[key]:raise ValueError(f'Unequal scientific initialization/configuration: {key}')
    metadata={'project','save_dir','name'}
    if {k:v for k,v in a['resolved_config'].items() if k not in metadata}!={k:v for k,v in c['resolved_config'].items() if k not in metadata}:
        raise ValueError('Undeclared resolved A/C scientific difference')
    _,_,_,_,_,expected=load_contract('configs/main_ac_frozen_v1.json','A')
    require_equal_budgets(ab,cb,expected)
    if [{k:v for k,v in r.items() if k!='condition'} for r in ae]!=[{k:v for k,v in r.items() if k!='condition'} for r in ce]:
        raise ValueError('Unequal A/C source order or membership')
    return {'status':'passed','scope':'W5 trained detector pair and clean validation only; no held-out evaluation',
            'runs':{'A':str(a_dir),'C':str(c_dir)},'effective_initialization_sha256':a['effective_initialization_sha256'],
            'protocol_sha256':a['protocol_sha256'],'overlay_sha256':a['overlay_sha256'],
            'per_model_expected_budget':{k:ab[k] for k in ['epochs','optimizer_update_count','microbatches','image_presentations','final_unstepped_images']},
            'same_source_order':True,'full_budget_and_EMA_traces_match':True,'G05_reference_status':'blocked'}


def validation_gallery(run_dir):
    run_dir=Path(run_dir).absolute()
    state,_,_=verify_single(run_dir)
    output=run_dir/'validation_gallery';output.mkdir(exist_ok=False)
    rows=[r for r in development_records() if r['split']=='val']
    selected=[next(r for r in rows if r['remapped_class_ids']==str(c)) for c in range(6)]
    model=YOLO(str(run_dir/'fit/weights/best.pt'),task='detect');records=[]
    for row in selected:
        path=ROOT/'data/processed/p57_development_v1'/row['image_path']
        if not path.resolve().is_relative_to((ROOT/'data/processed/p57_development_v1/val/images').resolve()):
            raise ValueError('Forbidden inference input')
        image=cv2.imread(str(path));shape=image.shape
        pred=model.predict(source=image,imgsz=416,device='mps',half=False,conf=.001,iou=.7,max_det=300,augment=False,classes=None,agnostic_nms=False,save=False,verbose=False)[0]
        boxes=pred.boxes.xyxy.cpu().numpy();classes=pred.boxes.cls.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
        validate_detections(boxes,classes,scores,shape,6)
        scale=640/max(shape[:2]);shown=cv2.resize(image,None,fx=scale,fy=scale)
        visible=[r for r in sorted(zip(boxes,classes,scores),key=lambda r:-float(r[2])) if r[2]>=.25][:10]
        drawn=cv2.copyMakeBorder(shown,0,max(60,24*(len(visible)+1)),0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        for index,(box,c,score) in enumerate(visible):
            x0,y0,x1,y1=(box*scale).astype(int);cv2.rectangle(drawn,(x0,y0),(x1,y1),(0,170,0),2)
            cv2.putText(drawn,str(index+1),(max(0,x0),max(18,y0+18)),0,.6,(0,150,0),2)
            cv2.putText(drawn,f'{index+1}: {model.names[int(c)]} {score:.3f}',(8,shown.shape[0]+24*(index+1)),0,.5,(0,0,0),1)
        if not visible:cv2.putText(drawn,'No detection at display confidence >= 0.25',(8,shown.shape[0]+24),0,.5,(0,0,0),1)
        cv2.imwrite(str(output/f"{row['source_id']}.jpg"),drawn)
        records.append({'source_id':row['source_id'],'split':'val','source_sha256':file_sha256(path),'shape':list(shape),'xyxy':boxes.tolist(),'classes':classes.tolist(),'confidence':scores.tolist()})
    report={'scope':'50-epoch main detector clean-validation-only inference; not final test','variant':state['variant'],'checkpoint_sha256':state['checkpoint_hashes']['best.pt'],'smoke_confidence':.001,'display_confidence':.25,'images':records}
    save_json(output/'predictions.json',report);return report
