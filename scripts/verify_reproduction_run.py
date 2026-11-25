"""Read-only evidence verification; recomputes metrics from the NEW inference stream."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p57.assets import file_sha256
from p57.reproduction import read,METRICS,CONDITIONS
from p57.evaluation_engine import recompute
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    if args.report.exists():raise SystemExit('Existing audit receipt is preserved; choose a new report path')
    cfg=read(ROOT/'configs/reproduction_v1.json');manifest=read(ROOT/cfg['manifest']);names={int(k):v for k,v in manifest['classes'].items()}
    status=read(run/'status.json');assert status['status']=='passed' and len(status['actual_child_exits'])==14 and set(status['actual_child_exits'])=={0}
    canonical=read(run/'artifacts/metrics_14.json');plots=read(run/'artifacts/plot_data.json');paired=read(run/'artifacts/paired_table.json');order=None;batches=None
    expected_conditions={(r['condition'],r['source_id']):r for r in manifest['condition_rows']};differences=[];total=0
    for condition in CONDITIONS:
        for model in ['A','C']:
            key=model+'_'+condition;directory=run/'runs'/key;m=read(directory/'metrics.json');pred=read(directory/'predictions.json');event=read(run/'logs'/key/'exit.json')
            assert pred['complete'] and event['returncode']==0 and len(pred['records'])==45
            assert file_sha256(directory/'predictions.json')==m['predictions_sha256']
            assert m['checkpoint_sha256']==manifest['checkpoints'][model]['sha256'] and m['images']==m['boxes']==45
            assert m['seed']==42 and m['device']=='cpu' and m['half'] is False and m['input_shapes']==[[448,448]]
            for k,v in manifest['settings'].items():assert m['settings'][k]==v,k
            assert m['completion']['nms_images']==m['completion']['metric_images']==45 and m['completion']['complete']
            this_batches=[[Path(x).stem for x in b['source_ids']] for b in m['completion']['batches']]
            if order is None:order=m['source_order'];batches=this_batches
            assert m['source_order']==order and batches==this_batches
            assert set(order)=={r['source_id'] for r in manifest['splits']['test']} and len(set(order))==45
            for r in pred['records']:
                wanted=expected_conditions[(condition,r['source_id'])]
                assert r['condition']==condition and r['image_sha256']==wanted['derived_file_sha256']
                assert file_sha256(Path(r['image_path']))==wanted['derived_file_sha256']
                label=run/'views'/condition/'test/labels'/(r['source_id']+'.txt')
                assert file_sha256(label)==wanted['label_sha256'];total+=1
            detail=recompute(pred['records'],names,1e-10,True)
            assert all(abs(detail['aggregate'][k]-m['aggregate'][k])<=1e-10 for k in METRICS)
            assert len(detail['per_class'])==6
            for a,b in zip(detail['per_class'],m['per_class']):assert a['id']==b['id'] and all(abs(a[k]-b[k])<=1e-10 for k in METRICS)
            row=next(x for x in canonical if x['model']==model and x['condition']==condition)
            assert all(row[k]==m['aggregate'][k] for k in METRICS) and row['metrics_sha256']==file_sha256(directory/'metrics.json')
            differences.append(row['difference_from_W6'])
    assert total==630
    for row in paired:
        a=next(x for x in canonical if x['model']=='A' and x['condition']==row['condition']);c=next(x for x in canonical if x['model']=='C' and x['condition']==row['condition'])
        for k in METRICS:assert row['A_'+k]==a[k] and row['C_'+k]==c[k] and row['delta_'+k+'_fraction']==c[k]-a[k]
    for family,levels in [('fading',['clean','F1','F2','F3']),('occlusion',['clean','O1','O2','O3'])]:
        for k in ['mAP50','precision','recall']:
            for model in ['A','C']:assert plots[family][k][model]==[next(x for x in canonical if x['model']==model and x['condition']==c)[k] for c in levels]
    assert len(list((run/'views').glob('*/test/images/*.png')))==315
    report={'status':'passed','new_stream_records_recomputed':total,'aggregate_and_all_per_class_tolerance':1e-10,'identical45source_and12batch_order':True,'condition_file_label_hashes_match_frozen_manifest':True,'settings_devices_shapes_and_actual_exits':True,'table_delta_and_curve_data_consistent':True,'max_abs_difference_from_W6':{k:max(abs(d[k]) for d in differences) for k in METRICS},'no_inference_performed_by_auditor':True,'run':str(run)}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
