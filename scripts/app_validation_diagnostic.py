"""W8D predeclared original-validation audit through the ACTUAL application worker."""
import argparse,base64,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'app'))
from p57_ui.core import normalize,png,pixel_sha,model_manifest,checkpoint,CLASSES,exports,displayed
from p57_ui.client import call_worker,WorkerFailure
from p57_ui.diagnostics import match_image,aggregate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def authenticated_validation():
    cfg=read(ROOT/'configs/reproduction_v1.json');manifest_path=ROOT/cfg['manifest']
    if sha(manifest_path)!=cfg['manifest_sha256']:raise ValueError('Release manifest drift')
    m=read(manifest_path);lock=ROOT/'docs/dataset_v2/dataset.lock.json'
    if sha(lock)!=m['historical_core_lock_sha256']:raise ValueError('Dataset lock drift')
    for name,wanted in m['frozen_files'].items():
        if sha(ROOT/name)!=wanted:raise ValueError('Frozen scientific code/config drift: '+name)
    manifest,_=model_manifest(ROOT)
    if m['classes']!={str(i):name for i,name in enumerate(CLASSES)}:raise ValueError('Class order drift')
    for model in ['A','C']:checkpoint(ROOT,manifest,model)
    local=read(ROOT/'configs/reproduction_assets.json');core=ROOT/local['core']
    rows=sorted(m['splits']['val'],key=lambda r:r['source_id'])
    if len(rows)!=45 or len({r['source_id'] for r in rows})!=45:raise ValueError('Validation membership drift')
    for kind in ['image','label']:
        directory=core/'val'/('images' if kind=='image' else 'labels')
        if {p.name for p in directory.iterdir()}!={Path(r[kind+'_path']).name for r in rows}:raise ValueError('Extra/missing validation files')
    boxes=0
    for row in rows:
        for kind in ['image','label']:
            relative=Path(row[kind+'_path'])
            if not str(relative).startswith('val/') or '..' in relative.parts:raise ValueError('Non-validation path')
            path=core/relative
            if any(p.is_symlink() for p in [path,*path.parents]) or sha(path)!=row[kind+'_sha256']:raise ValueError('Validation source bytes drift')
        if (core/row['label_path']).read_text()!=row['label_text']:raise ValueError('Label text drift')
        image,meta=normalize((core/row['image_path']).read_bytes())
        if meta['exif_orientation']!=1 or meta['source_size']!=meta['normalized_size']:raise ValueError('Label coordinate basis requires separate EXIF audit; do not silently transform ground truth')
        boxes+=len(row['label_text'].splitlines())
    if boxes!=46:raise ValueError('Expected46validation boxes')
    return cfg,m,core,rows


def ground_truth(row,size):
    w,h=size;out=[]
    for line in row['label_text'].splitlines():
        c,x,y,bw,bh=map(float,line.split());c=int(c)
        box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
        if not 0<=c<6 or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h:raise ValueError('Invalid diagnostic GT')
        out.append({'class_id':c,'xyxy':box})
    return out


def identities():
    names=['app/worker.py','app/p57_ui/core.py','app/p57_ui/client.py','app/p57_ui/diagnostics.py','app/models.json','scripts/app_validation_diagnostic.py','requirements.txt','app/requirements-ui.txt','src/p57/complete_validation.py','src/p57/degradation.py']
    return {name:sha(ROOT/name) for name in names}


def freeze(protocol):
    _,m,core,rows=authenticated_validation()
    value={'version':'w8d_validation_diagnostic_v1','created_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'DEVELOPMENT validation diagnostic only. No held-out image read or model/protocol refinement.',
      'models':['A','C'],'checkpoint_hashes':{k:v['sha256'] for k,v in m['checkpoints'].items()},
      'device':'cpu','execution':'45 sequential original-image worker requests, A then C within each request;90model-source outputs',
      'condition':'original','regions':[],'display_confidence':.25,'matching_iou':.5,'seed':42,
      'matching':'Per image, predictions drawable and score>=.25; sort descending confidence, classID,xyxy,original index. Greedy same-class unmatched GT IoU>=.5; highestIoU then lowestGT index. Wrong class=FP and missedGT=FN. GT label-file order. No threshold sweep.',
      'aggregation':'TP/FP/FN per class; micro P=TP/(TP+FP), R=TP/GT; undefined rate null. Not AP, best-F1 or W6.',
      'failure_selection':'Per model and FP/FN category separately, choose up to3distinct sources sorted descending category count then source_id; omit zero-count sources. Save actual app exports of union, no extra inference.',
      'input_policy':'Exact45val originals/labels and46boxes, sorted sourceID; authenticate membership/hashes before inference. Actual UI normalize then actual worker normalize/LetterBox. EXIF!=1 or size changes stop for label-coordinate-basis audit.',
      'ui_python':sys.executable,'ui_python_version':sys.version,'source_sha256':identities(),
      'release_manifest_sha256':sha(ROOT/'docs/reproduction_v1/release_manifest.json'),'dataset_lock_sha256':sha(ROOT/'docs/dataset_v2/dataset.lock.json'),
      'val_sources':rows,'expected_images':45,'expected_boxes':46,'no_test_image_io':True}
    save(protocol,value);print('Protocol frozen before inference:',protocol,flush=True)


def execute(protocol,out):
    p=read(protocol);_,m,core,rows=authenticated_validation()
    if identities()!=p['source_sha256'] or rows!=p['val_sources']:raise ValueError('Diagnostic code/input drift after predeclaration')
    out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',p);save(out/'protocol_identity.json',{'sha256':sha(protocol)})
    start=time.monotonic();records=[]
    try:
        for index,row in enumerate(rows):
            sid=row['source_id'];folder=out/'samples'/sid;folder.mkdir(parents=True)
            image,normalization=normalize((core/row['image_path']).read_bytes());normalized=png(image)
            (folder/'normalized.png').write_bytes(normalized)
            request={'op':'infer','image_png':base64.b64encode(normalized).decode(),'models':['A','C'],'device':'cpu','condition':'original','regions':[]}
            save(folder/'request.json',request)
            result=call_worker(ROOT,request)
            save(folder/'worker_result.json',result)
            if list(result['models'])!=['A','C'] or result['process']['actual_exit']!=0 or result['shared_pixel_sha256']!=pixel_sha(image):raise ValueError('Model/source/exit mixup')
            gt=ground_truth(row,image.size);model_scores={}
            for model,value in result['models'].items():
                if value['checkpoint_sha256']!=p['checkpoint_hashes'][model] or value['shared_pixel_sha256']!=result['shared_pixel_sha256'] or value['actual_tensor_shape']!=[1,3,448,448] or value['device'].split(':')[0]!='cpu' or not value['NMS_complete']:raise ValueError('Actual model/input/device/shape mismatch')
                model_scores[model]=match_image(value['raw_predictions'],gt)
            record={'source_id':sid,'original_image_sha256':row['image_sha256'],'label_sha256':row['label_sha256'],
                    'normalized_pixel_sha256':pixel_sha(image),'normalization':normalization,'ground_truth':gt,'models':model_scores,
                    'process':result['process'],'worker_result_sha256':sha(folder/'worker_result.json')}
            save(folder/'diagnostic.json',record);records.append(record)
            print(json.dumps({'sample':index+1,'source_id':sid,'exit':result['process']['actual_exit'],'models':{k:{n:v[n] for n in ['TP','FP','FN']} for k,v in model_scores.items()}}),flush=True)
            save(out/f'progress_{index+1:02d}.json',{'completed':index+1,'expected':45,'actual_exits':[r['process']['actual_exit'] for r in records]})
        summary={'scope':p['scope'],'protocol_sha256':sha(protocol),'images':len(records),'boxes':sum(len(r['ground_truth']) for r in records),
                 'model_source_outputs':2*len(records),'actual_worker_exits':[r['process']['actual_exit'] for r in records],
                 'elapsed_seconds_including_worker_cleanup':time.monotonic()-start,'models':{model:aggregate([r['models'][model] for r in records]) for model in ['A','C']}}
        selected=[]
        for model in ['A','C']:
            for kind in ['FP','FN']:
                chosen=sorted([r for r in records if r['models'][model][kind]>0],key=lambda r:(-r['models'][model][kind],r['source_id']))[:3]
                selected.extend({'model':model,'kind':kind,'source_id':r['source_id'],'count':r['models'][model][kind]} for r in chosen)
        for sid in sorted({r['source_id'] for r in selected}):
            folder=out/'samples'/sid;result=read(folder/'worker_result.json');record=read(folder/'diagnostic.json');normalized=(folder/'normalized.png').read_bytes()
            files,archive=exports(normalized,normalized,result,.25,record['normalization'])
            target=out/'selected_failures'/sid;target.mkdir(parents=True)
            for name,data in files.items():(target/name).write_bytes(data)
            (target/'results.zip').write_bytes(archive)
        save(out/'selected_failures.json',selected);save(out/'summary.json',summary)
        save(out/'status.json',{'status':'passed complete diagnostic execution','images':45,'model_outputs':90,'actual_exits':[r['process']['actual_exit'] for r in records]})
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        if isinstance(error,WorkerFailure):save(out/'failed_worker_receipt.json',error.receipt)
        save(out/'status.json',{'status':'failed','error':str(error),'completed_images':len(records),'elapsed_seconds':time.monotonic()-start})
        raise

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['freeze','run']);a.add_argument('--protocol',default='docs/app_diagnostic_v1/protocol.json');a.add_argument('--output',default='results/w8d_v1/validation');args=a.parse_args()
    protocol=ROOT/args.protocol
    if args.mode=='freeze':freeze(protocol)
    else:execute(protocol,ROOT/args.output)
