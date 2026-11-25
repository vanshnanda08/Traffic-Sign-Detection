"""Read-only recomputation of W8D retained outputs/exports, never inference."""
import argparse,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'app'))
from p57_ui.diagnostics import match_image,aggregate,iou
from p57_ui.core import displayed,normalize,pixel_sha


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def verify(directory):
    p=read(directory/'protocol.json');summary=read(directory/'summary.json');records=[];manifest=[];flags=[];nondrawable={'A':0,'C':0}
    authoritative=read(ROOT/'docs/reproduction_v1/release_manifest.json')
    if p['val_sources']!=sorted(authoritative['splits']['val'],key=lambda r:r['source_id']):raise ValueError('Non-validation membership')
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(directory/'protocol.json')==sha(ROOT/'docs/app_diagnostic_v1/protocol.json')
    for source in p['val_sources']:
        sid=source['source_id'];folder=directory/'samples'/sid;r=read(folder/'diagnostic.json');out=read(folder/'worker_result.json');request=read(folder/'request.json')
        assert r['original_image_sha256']==source['image_sha256'] and r['label_sha256']==source['label_sha256']
        assert sha(folder/'worker_result.json')==r['worker_result_sha256']
        assert r['process']['actual_exit']==out['process']['actual_exit']==0
        image,_=normalize((folder/'normalized.png').read_bytes());assert pixel_sha(image)==out['shared_pixel_sha256']==r['normalized_pixel_sha256']
        assert request['condition']=='original' and request['device']=='cpu' and request['models']==['A','C'] and not request['regions']
        for model in ['A','C']:
            value=out['models'][model];raw=value['raw_predictions']
            assert value['checkpoint_sha256']==p['checkpoint_hashes'][model]
            assert value['actual_tensor_shape']==[1,3,448,448] and value['NMS_complete']
            assert match_image(raw,r['ground_truth'])==r['models'][model]
            zero=[x for x in raw if not x['drawable']];nondrawable[model]+=len(zero)
            assert len(zero)==value['non_drawable_clipped_count']
            assert all(x['geometry_status']=='clipped_to_zero_area' for x in zero)
            assert not any(x in displayed(raw,.25) for x in zero)
        for a,ga in enumerate(r['ground_truth']):
            for b,gb in enumerate(r['ground_truth'][a+1:],a+1):
                overlap=iou(ga['xyxy'],gb['xyxy'])
                if ga['class_id']==gb['class_id'] and overlap>=.9:flags.append({'source_id':sid,'class_id':ga['class_id'],'annotation_indices':[a,b],'iou':overlap,'disposition':'Possible duplicate annotation; post-run inspection flag only. Original46GT and all diagnostic counts retained, no relabeling.'})
        records.append(r)
        manifest.append({'source_id':sid,'image_sha256':source['image_sha256'],'label_sha256':source['label_sha256'],'worker_result_sha256':sha(folder/'worker_result.json'),'diagnostic_sha256':sha(folder/'diagnostic.json'),'actual_exit':r['process']['actual_exit'],'worker_elapsed_seconds':r['process']['elapsed_seconds_including_cleanup'],
                         'normalization':r['normalization'],'predictions_per_model':{k:len(out['models'][k]['raw_predictions']) for k in ['A','C']},'scores':{k:{n:r['models'][k][n] for n in ['TP','FP','FN']} for k in ['A','C']}})
    assert len(records)==45 and sum(len(r['ground_truth']) for r in records)==46
    for model in ['A','C']:assert aggregate([r['models'][model] for r in records])==summary['models'][model]
    expected=[]
    for model in ['A','C']:
        for kind in ['FP','FN']:
            chosen=sorted([r for r in records if r['models'][model][kind]>0],key=lambda r:(-r['models'][model][kind],r['source_id']))[:3]
            expected.extend({'model':model,'kind':kind,'source_id':r['source_id'],'count':r['models'][model][kind]} for r in chosen)
    assert read(directory/'selected_failures.json')==expected
    archives=0
    for sid in sorted({r['source_id'] for r in expected}):
        folder=directory/'selected_failures'/sid
        with zipfile.ZipFile(folder/'results.zip') as z:
            assert z.testzip() is None and all('/' not in n and '\\' not in n for n in z.namelist())
            checks=json.loads(z.read('checksums.json'))
            assert all(hashlib.sha256(z.read(n)).hexdigest()==digest for n,digest in checks.items())
            exported=json.loads(z.read('predictions.json'));original=read(directory/'samples'/sid/'worker_result.json')
            csv_rows=list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
            for model in ['A','C']:
                raw=original['models'][model]['raw_predictions'];shown=displayed(raw,.25)
                assert exported['models'][model]['raw_predictions']==raw and exported['models'][model]['displayed_predictions']==shown
                for kind,rows in [('raw',raw),('displayed',shown)]:
                    actual=[v for v in csv_rows if v['model']==model and v['kind']==kind]
                    assert len(actual)==len(rows)
                    for a,b in zip(actual,rows):
                        assert int(a['class_id'])==b['class_id'] and float(a['confidence'])==b['confidence']
                        assert [float(a[k]) for k in ['x0','y0','x1','y1']]==b['xyxy']
            archives+=1
    report={'status':'passed recomputation and actual app-export verification','images':45,'boxes':46,'worker_exits':summary['actual_worker_exits'],'model_outputs':90,'per_image_matching_recomputed':True,'aggregate_per_class_recomputed':True,'deterministic_selection_verified':True,'selected_export_archives_verified':archives,'non_drawable_raw_retained':nondrawable,'annotation_review_flags':flags,'annotation_flag_scope':'Post-run inspection, not a changed/predeclared metric or revised GT','no_new_inference':True,'models':summary['models']}
    return report,manifest

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',default='results/w8d_v1/validation');a.add_argument('--report',default='docs/app_diagnostic_v1/recomputation.json');args=a.parse_args()
    report,manifest=verify(ROOT/args.run)
    for path,value in [(ROOT/args.report,report),(ROOT/'docs/app_diagnostic_v1/sample_manifest.json',manifest)]:
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))
