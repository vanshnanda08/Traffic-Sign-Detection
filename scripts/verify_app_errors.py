"""Execute real rejected upload requests; save exits/cleanup times, no inference scores."""
import base64
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from p57_ui.core import normalize,png
from p57_ui.client import call_worker,WorkerFailure

out=ROOT/'results/w7_verification_v1/error_paths_final.json'
if out.exists():raise SystemExit('Preserve the existing error-path receipt; no overwrite.')
fixtures=ROOT/'results/w7_verification_v1/fixtures'
image,_=normalize((fixtures/'upload_sign.jpg').read_bytes())
raw=base64.b64encode(png(image)).decode()
manifest=json.loads((ROOT/'app/models.json').read_text())
wrong=fixtures/'wrong_model.pt';wrong.write_bytes(b'not a trained checkpoint')
manifest['models']['A']['path']=str(wrong)
path=fixtures/'models_wrong.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
requests=[{'name':'corrupt','op':'transform','image_png':base64.b64encode(b'not an image').decode(),'condition':'original','regions':[]},
          {'name':'missing_regions','op':'transform','image_png':raw,'condition':'F1','regions':[]},
          {'name':'missing_model','op':'infer','image_png':raw,'condition':'original','regions':[],'models':['A','C'],'device':'cpu','manifest_path':str(fixtures/'models_missing.json')},
          {'name':'wrong_model_bytes','op':'infer','image_png':raw,'condition':'original','regions':[],'models':['A'],'device':'cpu','manifest_path':str(path)},
          {'name':'invalid_device','op':'infer','image_png':raw,'condition':'original','regions':[],'models':['A'],'device':'cuda'}]
records=[]
for request in requests:
    name=request.pop('name')
    try:call_worker(ROOT,request)
    except WorkerFailure as error:
        assert error.receipt['actual_exit']==1
        records.append({'name':name,'error':str(error),'process':error.receipt})
        print(name,str(error),flush=True)
    else:raise AssertionError('Invalid request was accepted')
out.write_text(json.dumps(records,indent=2)+'\n')
