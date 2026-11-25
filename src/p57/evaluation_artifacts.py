"""Canonical executed tables/curves and deterministic stored-prediction failures."""
import csv,json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.evaluation_data import config,verify_conditions,CONDITIONS
from p57.evaluation_protocol import verify_freeze
from p57.evaluation_engine import recompute
from p57.durability import atomic_json

METRICS=('mAP50','precision','recall','mAP50_95')


def display_matching(row,confidence=.25,iou_threshold=.5):
    chosen=np.flatnonzero(np.asarray(row['confidence'])>=confidence)
    true=np.asarray(row['ground_truth']['classes']);pred=np.asarray(row['classes'])[chosen]
    matrix=np.asarray(row['matching']['iou_matrix'],dtype=np.float32).reshape(len(true),len(row['classes']))[:,chosen]
    matrix=matrix*(true[:,None]==pred[None,:])
    matches=np.array(np.nonzero(matrix>=iou_threshold)).T
    if len(matches)>1:
        matches=matches[matrix[matches[:,0],matches[:,1]].argsort()[::-1]]
        matches=matches[np.unique(matches[:,1],return_index=True)[1]]
        matches=matches[np.unique(matches[:,0],return_index=True)[1]]
    tp=set(int(chosen[i]) for i in matches[:,1]) if len(matches) else set()
    gt=set(int(i) for i in matches[:,0]) if len(matches) else set()
    return {'TP':sorted(tp),'FP':[int(i) for i in chosen if int(i) not in tp],'FN':[i for i in range(len(true)) if i not in gt],'display_indices':chosen.tolist()}


def paired_table(runs):
    rows=[]
    for condition in CONDITIONS:
        a=runs[('A',condition)]['aggregate'];c=runs[('C',condition)]['aggregate']
        row={'condition':condition}
        for metric in METRICS:
            row['A_'+metric]=a[metric];row['C_'+metric]=c[metric];row['delta_'+metric+'_fraction']=c[metric]-a[metric]
        rows.append(row)
    return rows


def render_pair(a,c,path,names):
    panels=[]
    for model,row in [('A',a),('C',c)]:
        img=cv2.imread(row['image_path']);scale=640/max(img.shape[:2]);shown=cv2.resize(img,None,fx=scale,fy=scale)
        matches=display_matching(row);legend=[f"{model} {row['condition']} source {row['source_id']}",f"display conf>=0.25; TP {len(matches['TP'])} FP {len(matches['FP'])} FN {len(matches['FN'])}"]
        for i,(box,cls) in enumerate(zip(row['ground_truth']['xyxy'],row['ground_truth']['classes'])):
            x0,y0,x1,y1=(np.array(box)*scale).astype(int);color=(0,0,230) if i in matches['FN'] else (0,150,0)
            cv2.rectangle(shown,(x0,y0),(x1,y1),color,2);cv2.putText(shown,f'G{i+1}',(max(0,x0),max(18,y0+18)),0,.5,color,1)
            legend.append(f"GT G{i+1} {names[int(cls)]}"+(' FN' if i in matches['FN'] else ''))
        ranked=sorted(matches['display_indices'],key=lambda i:-row['confidence'][i])[:8]
        for n,i in enumerate(ranked):
            box=row['xyxy'][i];x0,y0,x1,y1=(np.array(box)*scale).astype(int);color=(0,140,250) if i in matches['FP'] else (220,90,0)
            cv2.rectangle(shown,(x0,y0),(x1,y1),color,1);cv2.putText(shown,str(n+1),(max(0,x0),min(shown.shape[0]-4,max(32,y0+32))),0,.5,color,1)
            legend.append(f"{n+1} {names[int(row['classes'][i])]} {row['confidence'][i]:.3f} "+('FP' if i in matches['FP'] else 'TP'))
        canvas=cv2.copyMakeBorder(shown,0,24*(len(legend)+1),0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        for n,text in enumerate(legend):cv2.putText(canvas,text,(8,shown.shape[0]+24*(n+1)),0,.45,(0,0,0),1)
        panels.append(canvas)
    height=max(p.shape[0] for p in panels);panels=[cv2.copyMakeBorder(p,0,height-p.shape[0],0,0,cv2.BORDER_CONSTANT,value=(255,255,255)) for p in panels]
    path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),np.hstack(panels))


def make_artifacts():
    receipt=verify_freeze();cfg=config();root=ROOT/cfg['output_root'];out=root/'artifacts'
    if out.exists():raise ValueError('Artifact version exists; verify it or create a documented new output version')
    manifest=verify_conditions();generation_sha=file_sha256(ROOT/cfg['derived_root']/'manifest.json')
    names={int(v['id']):v['name'] for v in json.loads((ROOT/cfg['core']/'dataset.lock.json').read_text())['class_mapping'].values()}
    runs={};predictions={};canonical=[];source_order=None
    for condition in CONDITIONS:
        for model in ['A','C']:
            key=f'{model}_{condition}';run=root/'runs'/key
            status=json.loads((run/'status.json').read_text());exit_data=json.loads((root/'supervision'/key/'exit.json').read_text())
            m=json.loads((run/'metrics.json').read_text());p=json.loads((run/'predictions.json').read_text())
            if status['status']!='passed' or exit_data['returncode']!=0 or not m['completion']['complete'] or m['images']!=45 or len(p['records'])!=45:raise ValueError('Incomplete evaluation; no missing-cell substitution allowed')
            if m['checkpoint_sha256']!=cfg['checkpoints'][model]['sha256'] or file_sha256(run/'predictions.json')!=m['predictions_sha256']:raise ValueError('Prediction/checkpoint hash drift')
            recomputed=recompute(p['records'],names,cfg['tolerances']['serialized_metric_absolute'])
            if any(abs(recomputed[k]-m['aggregate'][k])>cfg['tolerances']['serialized_metric_absolute'] for k in METRICS):raise ValueError('Metric recomputation mismatch')
            expected={r['source_id'] for r in manifest['rows'] if r['condition']==condition}
            if set(m['source_order'])!=expected or len(set(m['source_order']))!=45:raise ValueError('Pairing coverage mismatch')
            if source_order is None:source_order=m['source_order']
            if m['source_order']!=source_order:raise ValueError('Unequal model/condition source order')
            runs[(model,condition)]=m;predictions[(model,condition)]={r['source_id']:r for r in p['records']}
            canonical.append({'model':model,'condition':condition,'sources':45,'boxes':m['boxes'],**m['aggregate'],'complete':True,'checkpoint_sha256':m['checkpoint_sha256'],'metrics_sha256':file_sha256(run/'metrics.json'),'predictions_sha256':m['predictions_sha256'],'condition_manifest_sha256':generation_sha,'evaluation_config_sha256':receipt['config_sha256'],'per_class':m['per_class']})
    if len(canonical)!=14:raise ValueError('Expected14 actual metric rows')
    out.mkdir();atomic_json(out/'metrics_14.json',canonical)
    with (out/'metrics_14.csv').open('w',newline='') as f:
        keys=[k for k in canonical[0] if k!='per_class'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in canonical])
    paired=paired_table(runs);atomic_json(out/'paired_table.json',paired)
    with (out/'paired_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    header='| Condition | A mAP50 % | C mAP50 % | C-A pp | A P % | C P % | A R % | C R % |\n|---|---:|---:|---:|---:|---:|---:|---:|'
    lines=[header]+[f"| {r['condition']} | {100*r['A_mAP50']:.3f} | {100*r['C_mAP50']:.3f} | {100*r['delta_mAP50_fraction']:+.3f} | {100*r['A_precision']:.3f} | {100*r['C_precision']:.3f} | {100*r['A_recall']:.3f} | {100*r['C_recall']:.3f} |" for r in paired]
    (out/'main_table.md').write_text('\n'.join(lines)+'\n\nP/R: evaluator best-F1 convention, not fixed0.25. Machine metrics are fractions.\n')
    plot_data={}
    for family,levels,values in [('fading',['clean','F1','F2','F3'],[0,.15,.35,.60]),('occlusion',['clean','O1','O2','O3'],[0,.15,.30,.50])]:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5));data={}
        for ax,metric in zip(axes,['mAP50','precision','recall']):
            data[metric]={}
            for model,color in [('A','#2459a6'),('C','#ce6b19')]:
                fractions=[runs[(model,c)]['aggregate'][metric] for c in levels];data[metric][model]=fractions
                ax.plot(values,np.asarray(fractions)*100,marker='o',label=model,color=color)
            ax.set_ylim(0,100);ax.set_xticks(values,levels);ax.set_ylabel(metric+' (%)');ax.grid(alpha=.25);ax.legend();ax.set_xlabel('Fading alpha' if family=='fading' else 'Nominal occluded box-area fraction')
        fig.suptitle(f'P57 held-out {family}: same45sources; no test-driven tuning');fig.tight_layout();fig.savefig(out/f'{family}_robustness.png',dpi=200);fig.savefig(out/f'{family}_robustness.pdf');plt.close(fig)
        plot_data[family]={'conditions':levels,'severity':values,'metrics_in_fraction_units':data,'display_multiplier':100}
    atomic_json(out/'plot_data.json',plot_data)
    inventory=[];paired_cases=[]
    for condition in CONDITIONS:
        for sid in source_order:
            a=predictions[('A',condition)][sid];c=predictions[('C',condition)][sid]
            ma=display_matching(a);mc=display_matching(c)
            for model,m in [('A',ma),('C',mc)]:inventory.append({'source_id':sid,'condition':condition,'model':model,'FP':len(m['FP']),'FN':len(m['FN']),'TP':len(m['TP'])})
            ea=len(ma['FP'])+len(ma['FN']);ec=len(mc['FP'])+len(mc['FN']);paired_cases.append({'source_id':sid,'condition':condition,'A_errors':ea,'C_errors':ec,'delta_errors_C_minus_A':ec-ea})
    selection=[]
    for kind in ['FP','FN']:
        candidates=sorted([r for r in inventory if r[kind]>0],key=lambda r:(-r[kind],r['source_id'],CONDITIONS.index(r['condition']),r['model']))
        used=set()
        for row in candidates:
            if row['source_id'] in used:continue
            used.add(row['source_id']);selection.append({'category':kind,**row})
            if len(used)>=3:break
    choices={'negative_for_C':sorted([r for r in paired_cases if r['delta_errors_C_minus_A']>0],key=lambda r:(-r['delta_errors_C_minus_A'],r['source_id'],CONDITIONS.index(r['condition']))),
      'positive_for_C':sorted([r for r in paired_cases if r['delta_errors_C_minus_A']<0],key=lambda r:(r['delta_errors_C_minus_A'],r['source_id'],CONDITIONS.index(r['condition']))),
      'neutral':sorted([r for r in paired_cases if r['delta_errors_C_minus_A']==0],key=lambda r:(-r['A_errors'],r['source_id'],CONDITIONS.index(r['condition']))),
      'difficult':sorted(paired_cases,key=lambda r:(-(r['A_errors']+r['C_errors']),r['source_id'],CONDITIONS.index(r['condition'])))}
    absent=[]
    for category,items in choices.items():
        if items:selection.append({'category':category,**items[0]})
        else:absent.append(category)
    for i,row in enumerate(selection):
        condition=row['condition'];sid=row['source_id'];path=out/'failures'/f"{i:02d}_{row['category']}_{condition}_{sid}.png"
        render_pair(predictions[('A',condition)][sid],predictions[('C',condition)][sid],path,names)
        row['file']=str(path.relative_to(root));row['prediction_sources']={m:runs[(m,condition)]['predictions_sha256'] for m in ['A','C']}
    atomic_json(out/'failure_inventory.json',{'matching_confidence':.25,'matching_IoU':.5,'space':'stored metric-space IoUs; original-space boxes drawn','inventory':inventory,'selected':selection,'absent_categories':absent,'requested_FP_absent':not any(r['FP'] for r in inventory),'requested_FN_absent':not any(r['FN'] for r in inventory)})
    atomic_json(out/'artifact_manifest.json',{'canonical_metrics_sha256':file_sha256(out/'metrics_14.json'),'files':{str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file()},'evaluations':14,'model_source_evaluations':630,'all_recomputed':True,'all_source_orders_identical':True})
    print((out/'main_table.md').read_text());return canonical
