"""Recovery-v2 read-only audit of complete validation, journals and epoch commits."""
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
from p57.main_training import check_exposures, canonical_state_hash
from p57.recovery_protocol import load_recovery as load_contract, OVERLAY
from p57.complete_validation import CompletePredictor
from p57.durability import read_journal
from p57.protocol import require_equal_budgets


def verify_single(run_dir,require_gallery=False,load_weights=True):
    run_dir=Path(run_dir).absolute()
    state=json.loads((run_dir/'provenance.json').read_text())
    protocol,receipt,overlay,rows,schedule,expected=load_contract(ROOT/'configs/main_ac_recovery_v2.json',state['variant'])
    supervision=json.loads((run_dir.parent/(run_dir.name+'.supervision')/'exit.json').read_text())
    if supervision['returncode']!=0:raise ValueError('Process did not exit successfully through native cleanup')
    if state['status']!='passed' or len(state['epochs'])!=50:
        raise ValueError('Run did not pass every main training epoch')
    for key,wanted in [('protocol_sha256',receipt['protocol_sha256']),('exposure_manifest_sha256',receipt['exposure_manifest_sha256']),
                       ('initial_weights_sha256',protocol['initial_weights_sha256']),('dataset_lock_sha256',protocol['dataset_lock_sha256']),
                       ('overlay_sha256',file_sha256(OVERLAY)),('frozen_receipt_sha256',file_sha256(ROOT/'docs/main_ac_recovery_v2/freeze_receipt.json'))]:
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
    audit_durability(run_dir,state,actual,exposures)
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
            if not record.get('completion',{}).get('complete'):raise ValueError('Gallery processing incomplete')
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
    _,_,_,_,_,expected=load_contract(ROOT/'configs/main_ac_recovery_v2.json','A')
    require_equal_budgets(ab,cb,expected)
    if [{k:v for k,v in r.items() if k!='condition'} for r in ae]!=[{k:v for k,v in r.items() if k!='condition'} for r in ce]:
        raise ValueError('Unequal A/C source order or membership')
    return {'status':'passed','scope':'Recovery-v2 W5 replacement pair and complete clean validation only; no held-out evaluation',
            'runs':{'A':str(a_dir),'C':str(c_dir)},'effective_initialization_sha256':a['effective_initialization_sha256'],
            'protocol_sha256':a['protocol_sha256'],'overlay_sha256':a['overlay_sha256'],
            'per_model_expected_budget':{k:ab[k] for k in ['epochs','optimizer_update_count','microbatches','image_presentations','final_unstepped_images']},
            'same_source_order':True,'full_budget_and_EMA_traces_match':True,'all_validation_passes_complete':True,'durable_journals_and_commits_verified':True,'both_process_exit_codes':0,'G05_reference_status':'blocked'}


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
        pred=model.predict(source=str(path),predictor=CompletePredictor,imgsz=416,device='mps',half=False,conf=.001,iou=.7,max_det=300,augment=False,classes=None,agnostic_nms=False,save=False,verbose=False)[0]
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
        records.append({'source_id':row['source_id'],'split':'val','source_sha256':file_sha256(path),'shape':list(shape),'xyxy':boxes.tolist(),'classes':classes.tolist(),'confidence':scores.tolist(),'completion':model.predictor.last_completion})
    report={'scope':'Recovery-v2 50-epoch complete clean-validation-only inference; not final test','variant':state['variant'],'checkpoint_sha256':state['checkpoint_hashes']['best.pt'],'smoke_confidence':.001,'display_confidence':.25,'images':records}
    save_json(output/'predictions.json',report);return report


def audit_durability(run_dir,state,budget,exposures):
    journal=read_journal(run_dir/'events.jsonl')
    if journal['truncated_tail'] is not None:raise ValueError('Truncated journal cannot certify a complete run')
    events=journal['events']
    select=lambda name:[e['data'] for e in events if e['event']==name]
    batches=select('training_batch')
    observed=[row for batch in batches for row in batch['presentations']]
    if observed!=exposures or [len(b['presentations']) for b in batches]!=budget['microbatch_sizes']:
        raise ValueError('Journal actual deliveries differ from accounting')
    deliveries=select('sample_delivery')
    if deliveries!=[{k:r[k] for k in ['epoch','source_id','condition','image_path']} for r in exposures]:
        raise ValueError('Durable sample-delivery order mismatch')
    steps=select('optimizer_step')
    if [e['step'] for e in steps]!=budget['steps'] or [e['EMA_updates'] for e in steps]!=state['EMA_updates']:
        raise ValueError('Durable optimizer/EMA trace mismatch')
    if [i for e in steps for i in e['consumed_presentations']]!=list(range(1,10459)):
        raise ValueError('Optimizer-consumed source exposures mismatch')
    if select('epoch_train_end')!=budget['epoch_tails'] or [e['epoch'] for e in select('epoch_start')]!=list(range(1,51)):
        raise ValueError('Durable epoch/tail mismatch')
    val=select('validation_complete')
    if len(val)!=51 or val!=[v['completion'] for v in state['validation_passes']]:
        raise ValueError('Missing/extra complete validation passes')
    for index,report in enumerate(val):
        if not report['complete'] or report['expected_images']!=45 or report['metric_images']!=45 or report['nms_images']!=45:
            raise ValueError('Unverified complete validation')
        ids=[i for b in report['batches'] for i in b['source_ids']]
        if len(set(ids))!=45 or any('/val/images/' not in i for i in ids):raise ValueError('Duplicate/forbidden validation source')
        if index<50 and report['scope']!={'kind':'epoch','epoch':index+1}:raise ValueError('Validation epoch selection mismatch')
    if select('checkpoint_write')!=state['checkpoint_writes'] or not all(w['complete_validation'] for w in state['checkpoint_writes']):
        raise ValueError('Unverified checkpoint selection inputs')
    commits=select('epoch_commit');labels=[f'epoch-{i:03d}' for i in range(1,51)]+['final']
    if [c['label'] for c in commits]!=labels:raise ValueError('Epoch commit coverage incomplete')
    for commit in commits:
        directory=run_dir/'commits'/commit['label'];manifest_path=directory/'manifest.json'
        if file_sha256(manifest_path)!=commit['manifest_sha256']:raise ValueError('Commit manifest drift')
        manifest=json.loads(manifest_path.read_text())
        for name,wanted in manifest['snapshots'].items():
            if file_sha256(directory/name)!=wanted:raise ValueError('Atomic epoch snapshot changed')
        for row in manifest['checkpoints'].values():
            if file_sha256(run_dir/row['object'])!=row['sha256']:raise ValueError('Epoch checkpoint object changed')
        seq=manifest['journal_sequence_before_commit']
        if seq<1 or events[seq-1]['sha256']!=manifest['journal_sha256_before_commit']:
            raise ValueError('Commit/journal boundary mismatch')
    final=run_dir/'commits/final'
    if json.loads((final/'accounting.json').read_text())!=budget or json.loads((final/'exposures.json').read_text())!=exposures:
        raise ValueError('Final committed accounting/exposure mismatch')
    selected_manifest=json.loads((run_dir/'commits'/f"epoch-{state['selected_epoch']:03d}"/'manifest.json').read_text())
    raw=run_dir/selected_manifest['checkpoints']['best.pt']['object']
    source_model=YOLO(str(raw),task='detect').model
    final_model=YOLO(str(run_dir/'fit/weights/best.pt'),task='detect').model
    if canonical_state_hash(source_model)!=canonical_state_hash(final_model):
        raise ValueError('Final best checkpoint differs from validation-selected committed EMA')
