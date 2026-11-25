"""W8A portable evaluation layer; frozen W4/W5/W6 implementations remain intact."""
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, ap_per_class, compute_ap
from p57.assets import file_sha256
from p57.complete_validation import PINNED_NMS
from p57.dataset import ROOT, parse_labels
from p57.degradation import degrade
from p57.durability import atomic_json
from p57.environment import seed_everything, validate_pins
from p57.evaluation_data import AuthDataset, pixels_sha
from p57.evaluation_engine import RecordingValidator, recompute, require_local_font

CONDITIONS = ['clean','F1','F2','F3','O1','O2','O3']
METRICS = ['precision','recall','mAP50','mAP50_95']


def read(path):
    return json.loads(Path(path).read_text())


def new_directory(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def asset_path(root, value):
    """Directories/local files only: no archives, URLs, traversal or linked ancestors."""
    if not isinstance(value,str) or '://' in value or '\\' in value or '..' in Path(value).parts:
        raise ValueError('Unsafe local asset path; URLs/archives/traversal not accepted')
    p=Path(value).expanduser()
    p=p if p.is_absolute() else Path(root)/p
    if any(x.is_symlink() for x in [p,*p.parents]):raise ValueError('Linked asset paths rejected')
    return p.absolute()


def library_hashes():
    objects={'NMS':PINNED_NMS,'base_validator':BaseValidator.__call__,'matching':BaseValidator.match_predictions,
             'prepare_batch':DetectionValidator._prepare_batch,'process_batch':DetectionValidator._process_batch,
             'scale_preds':DetectionValidator.scale_preds,'update_metrics':DetectionValidator.update_metrics,
             'get_stats':DetectionValidator.get_stats,'DetMetrics':DetMetrics.process,'AP':ap_per_class,'compute_AP':compute_ap}
    return {k:hashlib.sha256(inspect.getsource(v).encode()).hexdigest() for k,v in objects.items()}


def authenticate(root=ROOT, assets_file=None):
    root=Path(root);cfg=read(root/'configs/reproduction_v1.json')
    manifest_path=root/cfg['manifest']
    if file_sha256(manifest_path)!=cfg['manifest_sha256']:raise ValueError('Release manifest drift')
    m=read(manifest_path)
    historic=read(root/'configs/evaluation_v1.json')
    expected_names=historic['checkpoints']
    if m['checkpoints']!=expected_names:raise ValueError('Checkpoint identity drift')
    if m['settings']!={**historic['settings'],'device':'cpu'} or m['seed']!=42:raise ValueError('Scientific setting drift')
    core_lock=read(root/'docs/dataset_v2/dataset.lock.json')
    expected_classes={str(v['id']):v['name'] for v in core_lock['class_mapping'].values()}
    if m['classes']!=expected_classes or len(expected_classes)!=6:raise ValueError('Class mapping drift')
    for name,wanted in m['frozen_files'].items():
        if file_sha256(root/name)!=wanted:raise ValueError('Frozen source/config drift: '+name)
    # Small retained provenance records are required; original training stores are not.
    for name,wanted in m['provenance_references'].items():
        if file_sha256(root/name)!=wanted:raise ValueError('Retained provenance drift: '+name)
    if library_hashes()!=m['library_hashes']:raise ValueError('Installed evaluator/library identity drift')
    versions=validate_pins()
    ap=Path(assets_file) if assets_file else root/'configs/reproduction_assets.json'
    assets=read(ap);core=asset_path(root,assets['core']);models={}
    for key in ['A','C']:
        path=asset_path(root,assets['checkpoints'][key])
        if not path.is_file() or file_sha256(path)!=m['checkpoints'][key]['sha256']:
            raise ValueError('Missing/corrupt canonical checkpoint '+key)
        models[key]=path
    for split in ['val','test']:
        rows=m['splits'][split]
        expected={name:sha for name,sha in core_lock['files'].items() if name.startswith(split+'/')}
        actual={}
        if len(rows)!=45 or len({r['source_id'] for r in rows})!=45:raise ValueError('Source membership/count drift')
        for row in rows:
            if not Path(row['image_path']).stem.startswith(row['source_id']+'_'):raise ValueError('Source ID/filename membership drift')
            for field,sha_field in [('image_path','image_sha256'),('label_path','label_sha256')]:
                name=row[field];path=asset_path(core,name)
                if not name.startswith(split+'/') or expected.get(name)!=row[sha_field]:raise ValueError('Wrong source/split membership')
                if not path.is_file() or file_sha256(path)!=row[sha_field]:raise ValueError('Missing/corrupt input '+name)
                actual[name]=row[sha_field]
            if (core/row['label_path']).read_text()!=row['label_text']:raise ValueError('Label text drift')
            parse_labels(row['label_text'],6)
        if actual!=expected:raise ValueError('Incomplete locked membership')
    if len(m['condition_rows'])!=315 or len({(r['condition'],r['source_id']) for r in m['condition_rows']})!=315:
        raise ValueError('Invalid paired condition membership')
    return cfg,m,core,models,versions


def build_view(root,rows,names,source_core,condition,manifest_rows=None,split='test'):
    """Generate authenticated lossless views into a new directory, never raw assets."""
    root=new_directory(root);images=root/split/'images';labels=root/split/'labels'
    images.mkdir(parents=True);labels.mkdir()
    expected={(r['condition'],r['source_id']):r for r in (manifest_rows or [])}
    generation=[]
    for row in rows:
        source=source_core/row['image_path'];label=source_core/row['label_path']
        image=cv2.imread(str(source));boxes=parse_labels(label.read_text(),6)
        if image is None:raise ValueError('Undecodable authenticated input')
        if condition=='development':
            target=images/source.name;shutil.copy2(source,target)
        else:
            detail={'condition':'clean'}
            if condition!='clean':
                box32=np.asarray(boxes,dtype=np.float32).reshape(-1,5).tolist()
                image,_,detail=degrade(image,box32,row['source_id'],condition,read(ROOT/'configs/degradation_dev_v1.json'))
            target=images/(row['source_id']+'.png')
            if not cv2.imwrite(str(target),image,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise RuntimeError('PNG write failed')
            wanted=expected[(condition,row['source_id'])]
            if file_sha256(target)!=wanted['derived_file_sha256'] or pixels_sha(image)!=wanted['derived_decoded_pixels_sha256']:
                raise ValueError('Generated condition differs from W6 frozen view')
            if not np.array_equal(cv2.imread(str(target)),image):raise ValueError('PNG pixel mismatch')
        destlabel=labels/(target.stem+'.txt');shutil.copy2(label,destlabel)
        if file_sha256(destlabel)!=row['label_sha256']:raise ValueError('Annotation changed')
        generation.append({'source_id':row['source_id'],'image':str(target.relative_to(root)),
                           'image_sha256':file_sha256(target),'label_sha256':file_sha256(destlabel),
                           'boxes':len(boxes),'decoded_pixel_sha256':pixels_sha(image)})
    data={'train':None,'val':None,'test':None,'nc':6,'names':names};data[split]=str(images.resolve())
    path=root/'dataset.yaml';path.write_text(yaml.safe_dump(data,sort_keys=False))
    atomic_json(root/'generation.json',generation)
    return root


class PortableInputs:
    def __init__(self,view,names,rows,condition,split='test',expected_conditions=None):
        self.view=Path(view);self.images=self.view/split/'images';self.split=split
        self.yaml=self.view/'dataset.yaml';self.names=names;self.condition=condition
        self.mode='portable_development' if split=='val' else 'fresh_reproduction'
        self.metadata={'train':None,'val':None,'test':None,'nc':6,'names':names}
        self.metadata[split]=str(self.images.resolve())
        if yaml.safe_load(self.yaml.read_text())!=self.metadata:raise ValueError('Local YAML redirection rejected')
        generation=read(self.view/'generation.json')
        if len(generation)!=45 or {g['source_id'] for g in generation}!={r['source_id'] for r in rows}:raise ValueError('Foreign/missing generation source')
        self.files={};self.rows={}
        originals={r['source_id']:r for r in rows}
        expected={(r['condition'],r['source_id']):r for r in (expected_conditions or [])}
        for g in generation:
            path=asset_path(self.view,g['image']);label=self.view/split/'labels'/(path.stem+'.txt')
            original=originals[g['source_id']]
            wanted_name=Path(original['image_path']).name if split=='val' else g['source_id']+'.png'
            if g['image']!=f'{split}/images/{wanted_name}' or g['label_sha256']!=original['label_sha256']:
                raise ValueError('Wrong source/path/annotation pairing')
            if split=='test':
                wanted=expected[(condition,g['source_id'])]
                if g['image_sha256']!=wanted['derived_file_sha256']:raise ValueError('Condition image authentication drift')
            if file_sha256(path)!=g['image_sha256'] or file_sha256(label)!=g['label_sha256']:raise ValueError('View changed')
            self.files[str(path.relative_to(self.view))]=g['image_sha256'];self.files[str(label.relative_to(self.view))]=g['label_sha256']
            self.rows[str(path.resolve())]={'source_id':g['source_id']}
        self.expected=sorted(self.rows)
        actual={str(p.resolve()) for p in self.images.iterdir() if p.is_file()}
        if actual!=set(self.expected) or len(self.rows)!=45:raise ValueError('Missing/foreign image membership')
        if {p.name for p in (self.view/split/'labels').iterdir()}!={Path(x).stem+'.txt' for x in self.expected}:
            raise ValueError('Missing/foreign label membership')


def execute(auth,checkpoint,output,device,manifest,cfg):
    """Use unchanged RecordingValidator/complete-NMS/AP methods with portable auth."""
    require_local_font();versions=validate_pins();seed_everything(42,True);torch.set_num_threads(2)
    if device not in ['cpu','mps'] or (device=='mps' and not torch.backends.mps.is_available()):raise ValueError('Explicit device unavailable')
    output=new_directory(output)
    settings={**manifest['settings'],'device':device,'data':str(auth.yaml),'split':auth.split}
    dataset=AuthDataset(auth,imgsz=416,batch_size=4,hyp=get_cfg(overrides=settings),rect=True,stride=32)
    loader=build_dataloader(dataset,batch=4,workers=0,shuffle=False,rank=-1)
    validator=RecordingValidator(auth,output,loader,save_dir=output,args=settings)
    started=time.monotonic();validator(model=str(checkpoint))
    if str(validator.device).split(':')[0]!=device or validator.args.half:raise ValueError('Actual device/precision drift')
    if len(validator.records)!=45 or len({r['source_id'] for r in validator.records})!=45:raise ValueError('Incomplete metric stream')
    predictions={'complete':True,'scope':auth.mode,'condition':auth.condition,'checkpoint_sha256':file_sha256(checkpoint),'records':validator.records}
    atomic_json(output/'predictions.json',predictions)
    detail=recompute(read(output/'predictions.json')['records'],auth.names,cfg['serialization_tolerance'],True)
    original=validator.metric_result
    if len(detail['per_class'])!=len(original['per_class']):raise ValueError('Class metric incompleteness')
    for a,b in zip(detail['per_class'],original['per_class']):
        if a['id']!=b['id'] or any(abs(a[k]-b[k])>cfg['serialization_tolerance'] for k in METRICS):raise ValueError('Serialized per-class metric drift')
    if any(abs(detail['aggregate'][k]-original['aggregate'][k])>cfg['serialization_tolerance'] for k in METRICS):raise ValueError('Serialized aggregate drift')
    result={**original,'images':45,'boxes':sum(len(r['ground_truth']['classes']) for r in validator.records),
            'completion':validator.last_completion,'condition':auth.condition,'device':str(validator.device),'half':validator.args.half,
            'settings':settings,'seed':42,'versions':versions,'checkpoint_sha256':file_sha256(checkpoint),
            'predictions_sha256':file_sha256(output/'predictions.json'),'source_order':[r['source_id'] for r in validator.records],
            'input_shapes':sorted({tuple(r['input_shape']) for r in validator.records}),
            'runtime_seconds_before_cleanup':time.monotonic()-started,'serialized_metrics_recomputed':True}
    if not all(math.isfinite(v) for v in result['aggregate'].values()):raise ValueError('Invalid metrics')
    atomic_json(output/'metrics.json',result)
    return result


def aggregate_run(root,manifest,cfg,device):
    root=Path(root);rows=[];source_order=None
    historical=read(ROOT/'docs/evaluation_v1/measured/metrics_14.json')
    for condition in CONDITIONS:
        for model in ['A','C']:
            key=f'{model}_{condition}';run=root/'runs'/key
            event=read(root/'logs'/key/'exit.json');m=read(run/'metrics.json')
            if event['returncode']!=0 or not m['completion']['complete'] or m['images']!=45 or m['boxes']!=45:raise ValueError('Incomplete run; no table allowed')
            if file_sha256(run/'predictions.json')!=m['predictions_sha256']:raise ValueError('Prediction hash drift')
            if m['checkpoint_sha256']!=manifest['checkpoints'][model]['sha256'] or m['device'].split(':')[0]!=device:raise ValueError('Model/device drift')
            if source_order is None:source_order=m['source_order']
            if m['source_order']!=source_order:raise ValueError('Unequal source order')
            prior=next(x for x in historical if x['model']==model and x['condition']==condition)
            differences={k:m['aggregate'][k]-prior[k] for k in METRICS}
            tolerances={k:cfg['same_device_tolerance'] if device=='mps' else cfg['cross_device_tolerances_declared_before_development'][k] for k in METRICS}
            rows.append({'model':model,'condition':condition,**m['aggregate'],'per_class':m['per_class'],
                         'source_count':45,'box_count':45,'checkpoint_sha256':m['checkpoint_sha256'],
                         'metrics_sha256':file_sha256(run/'metrics.json'),'prediction_sha256':m['predictions_sha256'],
                         'device':device,'actual_exit':event['returncode'],'elapsed_including_cleanup':event['elapsed_seconds_including_cleanup'],
                         'difference_from_W6':differences,'within_declared_tolerance':all(abs(differences[k])<=tolerances[k] for k in METRICS)})
    output=new_directory(root/'artifacts');atomic_json(output/'metrics_14.json',rows)
    keys=['model','condition',*METRICS,'source_count','box_count','device','actual_exit','elapsed_including_cleanup']
    with (output/'metrics_14.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
    table=[]
    for condition in CONDITIONS:
        a=next(x for x in rows if x['model']=='A' and x['condition']==condition)
        c=next(x for x in rows if x['model']=='C' and x['condition']==condition)
        table.append({'condition':condition,**{'A_'+k:a[k] for k in METRICS},**{'C_'+k:c[k] for k in METRICS},**{'delta_'+k+'_fraction':c[k]-a[k] for k in METRICS}})
    atomic_json(output/'paired_table.json',table)
    text='| Condition | A mAP50 % | C mAP50 % | C-A pp | A P % | C P % | A R % | C R % |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in table:text+=f"| {r['condition']} | {100*r['A_mAP50']:.3f} | {100*r['C_mAP50']:.3f} | {100*r['delta_mAP50_fraction']:+.3f} | {100*r['A_precision']:.3f} | {100*r['C_precision']:.3f} | {100*r['A_recall']:.3f} | {100*r['C_recall']:.3f} |\n"
    (output/'main_table.md').write_text(text+'\nFresh inference; machine values fractions, displayed percentages; P/R common smoothed best-F1.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots={}
    for family,levels in [('fading',['clean','F1','F2','F3']),('occlusion',['clean','O1','O2','O3'])]:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5));plots[family]={}
        for ax,k in zip(axes,['mAP50','precision','recall']):
            plots[family][k]={}
            for model in ['A','C']:
                values=[next(x for x in rows if x['model']==model and x['condition']==c)[k] for c in levels]
                plots[family][k][model]=values;ax.plot(levels,np.asarray(values)*100,'o-',label=model)
            ax.set_ylim(0,100);ax.set_ylabel(k+' (%)');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Fresh reproduction: '+family+'; same45sources; '+device);fig.tight_layout();fig.savefig(output/(family+'.png'),dpi=180);plt.close(fig)
    atomic_json(output/'plot_data.json',plots)
    report={'complete':True,'evaluations':14,'model_source_evaluations':630,'all_actual_exits_zero':True,
            'all_within_predeclared_tolerances':all(r['within_declared_tolerance'] for r in rows),
            'max_absolute_differences':{k:max(abs(r['difference_from_W6'][k]) for r in rows) for k in METRICS}}
    atomic_json(output/'comparison.json',report);print(text,flush=True)
    return report
