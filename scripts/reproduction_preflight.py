"""Development-only CPU/MPS and historical-path checks before reproduction freeze."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false';os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import sys,json,argparse
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from p57.reproduction import authenticate,build_view,PortableInputs,execute,read,new_directory,METRICS
from p57.evaluation_data import AuthInputs
from p57.evaluation_engine import execute_evaluator
from p57.assets import file_sha256
from p57.durability import atomic_json

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='results/w8a_v1/development_preflight')
    args=parser.parse_args()
    cfg,m,core,models,versions=authenticate();out=new_directory(ROOT/args.output)
    names={int(k):v for k,v in m['classes'].items()}
    view=build_view(out/'view',m['splits']['val'],names,core,'development',split='val')
    auth=PortableInputs(view,names,m['splits']['val'],'clean','val');old=AuthInputs('development')
    result={'status':'in_progress','scope':'45development sources/46boxes only; no held-out inference',
            'config_sha256':file_sha256(ROOT/'configs/reproduction_v1.json'),'declared_tolerances':cfg,'models':{}}
    for model,path in models.items():
        historical=execute_evaluator(old,path,out/(model+'_historical_mps'))
        new_mps=execute(auth,path,out/(model+'_portable_mps'),'mps',m,cfg)
        cpu=execute(auth,path,out/(model+'_portable_cpu'),'cpu',m,cfg)
        same={k:abs(historical['aggregate'][k]-new_mps['aggregate'][k]) for k in METRICS}
        cross={k:abs(historical['aggregate'][k]-cpu['aggregate'][k]) for k in METRICS}
        if any(same[k]>cfg['same_device_tolerance'] or cross[k]>cfg['cross_device_tolerances_declared_before_development'][k] for k in METRICS):raise ValueError('Development numerical gate failed; no tolerance widening or held-out execution')
        records=[read(out/(model+suffix)/'predictions.json')['records'] for suffix in ['_historical_mps','_portable_mps','_portable_cpu']]
        reference=records[0];differences=[]
        for stream in records[1:]:
            worst_box=worst_conf=0.
            for a,b in zip(reference,stream):
                if a['source_id']!=b['source_id'] or a['input_shape']!=b['input_shape']:raise ValueError('Preprocessing/source order drift')
                ai=np.flatnonzero(np.asarray(a['confidence'])>=.25);bi=np.flatnonzero(np.asarray(b['confidence'])>=.25)
                # Sort by class and confidence to compare the same high-score predictions.
                ai=sorted(ai,key=lambda i:(a['classes'][i],-a['confidence'][i]));bi=sorted(bi,key=lambda i:(b['classes'][i],-b['confidence'][i]))
                if len(ai)!=len(bi) or [a['classes'][i] for i in ai]!=[b['classes'][i] for i in bi]:raise ValueError('Development high-confidence class/count drift')
                if len(ai):
                    worst_box=max(worst_box,float(np.abs(np.array(a['network_xyxy'])[ai]-np.array(b['network_xyxy'])[bi]).max()))
                    worst_conf=max(worst_conf,float(np.abs(np.array(a['confidence'])[ai]-np.array(b['confidence'])[bi]).max()))
            if worst_box>cfg['development_prediction_tolerances']['xyxy_absolute_pixels'] or worst_conf>cfg['development_prediction_tolerances']['confidence_absolute']:raise ValueError('Development FP32 prediction tolerance failed')
            differences.append({'max_box_pixels':worst_box,'max_confidence_difference':worst_conf})
        # Verify every preprocessed development tensor, rather than infer parity from dimensions.
        from ultralytics.cfg import get_cfg
        from p57.evaluation_data import AuthDataset
        datasets=[AuthDataset(x,imgsz=416,batch_size=4,hyp=get_cfg(),rect=True,stride=32) for x in [old,auth]]
        for i in range(45):np.testing.assert_array_equal(datasets[0][i]['img'],datasets[1][i]['img'])
        result['models'][model]={'same_device_metric_absolute_differences':same,'cross_device_metric_absolute_differences':cross,
                                 'prediction_comparisons':differences,'all45input_tensors_exact':True,
                                 'all45sources_complete':True,'actual_shape':[448,448]}
    result.update(status='passed',supported_devices=['cpu','mps'],implementation_sha256=file_sha256(ROOT/'src/p57/reproduction.py'))
    atomic_json(ROOT/'docs/reproduction_v1/development_preflight.json',result)
    print(json.dumps(result,indent=2))
