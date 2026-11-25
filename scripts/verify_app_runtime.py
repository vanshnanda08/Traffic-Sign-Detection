"""Explicit application checks on development upload fixtures; never benchmarks."""
import base64
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from p57_ui.core import normalize,png,exports,pixel_sha,displayed
from p57_ui.client import call_worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/w7_verification_v1/runtime')
    args=parser.parse_args()
    out=ROOT/args.output
    out.mkdir(exist_ok=False)
    fixture=ROOT/'results/w7_verification_v1/fixtures'
    records=[]
    for name,device,condition,models in [('upload_wide.png','cpu','O3',['A','C']),
                                          ('upload_exif.jpg','cpu','F1',['A']),
                                          ('upload_exif.jpg','cpu','O2',['C']),
                                          ('upload_wide.png','mps','O3',['A','C'])]:
        image,normalization=normalize((fixture/name).read_bytes())
        request={'op':'infer','image_png':base64.b64encode(png(image)).decode(),
                 'regions':[[0,0,image.width,image.height],[10,10,min(100,image.width),min(100,image.height)]],
                 'condition':condition,'models':models,'device':device}
        result=call_worker(ROOT,request)
        assert all(m['shared_pixel_sha256']==result['shared_pixel_sha256'] for m in result['models'].values())
        assert all(m['actual_tensor_shape']==[1,3,448,448] and m['NMS_complete'] for m in result['models'].values())
        assert all(m['device'].split(':')[0]==device for m in result['models'].values())
        for model in result['models'].values():
            assert all(r['drawable'] and r['xyxy'][2]>r['xyxy'][0] and r['xyxy'][3]>r['xyxy'][1] for r in displayed(model['raw_predictions'],0))
            assert model['non_drawable_clipped_count']==sum(not r['drawable'] for r in model['raw_predictions'])
        result['fixture']={'filename':name,'normalized_size':list(image.size),'source_kind':'Development-source layout/orientation fixture; not a new road photograph or held-out evaluation.'}
        (out/f'{len(records):02d}_{device}_{condition}.json').write_text(json.dumps(result,indent=2)+'\n')
        shared=call_worker(ROOT,{**request,'op':'transform'})
        shared_bytes=base64.b64decode(shared['shared_png'])
        assert shared['shared_pixel_sha256']==result['shared_pixel_sha256']
        files,archive=exports(png(image),shared_bytes,result,.25,normalization)
        (out/f'{len(records):02d}_results.zip').write_bytes(archive)
        records.append({'device':device,'condition':condition,'models':models,'actual_exit':result['process']['actual_exit'],
                        'elapsed_seconds_including_cleanup':result['process']['elapsed_seconds_including_cleanup'],
                        'normalized_size':list(image.size),'shared_pixel_sha256':result['shared_pixel_sha256']})
        print(records[-1],flush=True)
    with tempfile.TemporaryDirectory(prefix='p57-minimal-') as temp:
        layout=Path(temp)
        shutil.copytree(ROOT/'app',layout/'app',ignore=shutil.ignore_patterns('__pycache__'))
        for relative in ['requirements.txt','configs/degradation_dev_v1.json',
                         'src/p57/__init__.py','src/p57/environment.py',
                         'src/p57/degradation.py','src/p57/complete_validation.py']:
            target=layout/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
        manifest=json.loads((layout/'app/models.json').read_text())
        for key,entry in manifest['models'].items():
            target=layout/'models'/f'{key}.pt';target.parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/entry['path'],target);entry['path']=str(target.relative_to(layout))
        (layout/'app/models.json').write_text(json.dumps(manifest,indent=2)+'\n')
        assert not any((layout/p).exists() for p in ['data','results','docs','checkpoints'])
        image,_=normalize((fixture/'upload_wide.png').read_bytes())
        os.environ['P57_INFERENCE_PYTHON']=str(ROOT/'.venv/bin/python')
        minimal=call_worker(layout,{'op':'infer','image_png':base64.b64encode(png(image)).decode(),
                                   'regions':[],'condition':'original','models':['A','C'],'device':'cpu'})
        (out/'minimal_inference.json').write_text(json.dumps(minimal,indent=2)+'\n')
        inventory=[str(p.relative_to(layout)) for p in layout.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    report={'status':'passed','scope':'Application fixture inference only; not benchmark rerun or second-PC verification.',
            'actual_runs':records,'minimal_layout_inference':{'status':'passed','actual_exit':minimal['process']['actual_exit'],
                'files':inventory,'no_dataset_or_training_log_or_audit_store':True},
            'network_scope':'Run under parent macOS deny-external/allow-loopback sandbox; inherited by all workers.'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CPU/MPS, EXIF/non-square, exported archives and minimal-layout inference passed',flush=True)


if __name__=='__main__':main()
