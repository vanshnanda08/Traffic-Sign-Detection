"""Versioned v2 contract and candidates; never alters old pins or trained runs."""
import hashlib,json,shutil
from pathlib import Path
from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.main_training import load_contract
from p57.recovery_protocol import extra_library_sources
from p57.durability import atomic_json

FILES=['src/p57/complete_validation.py','src/p57/durability.py','src/p57/recovery_protocol.py',
 'src/p57/recovery_training.py','src/p57/recovery_verification.py','scripts/train_recovery.py',
 'scripts/supervise_recovery.py','scripts/recovery_preflight.py','scripts/check_recovery.py',
 'scripts/create_recovery_contract.py','tests/test_recovery.py']
if __name__=='__main__':
    if any((ROOT/'results/main_ac_recovery_v2'/name).exists() for name in ['A','C','A.supervision','C.supervision']):
        raise SystemExit('Replacement run exists; no repinning permitted')
    parent,parent_receipt,_,_,_,_=load_contract('configs/main_ac_frozen_v1.json','A')
    protocol={**parent,'version':'main_ac_recovery_v2','scope':'Replacement detector protocol; clean validation complete, final-test still closed',
      'parent_protocol':'configs/main_ac_frozen_v1.json','parent_protocol_sha256':file_sha256(ROOT/'configs/main_ac_frozen_v1.json'),
      'change_description':'Disable only pinned NMS wall-clock break for all validation/prediction; validate source coverage before metrics/selection. Add durable actual-event journal, atomic epoch/checkpoint commits and foreground supervised actual exit. Both models restart original initialization.',
      'validation_completeness':{'time_budget':'unlimited_wall_clock','max_time_img_argument':'positive_infinity','standard_JSON_no_Infinity_literal':True,'scope':['per_epoch_clean_validation','final_best_clean_validation','validation_galleries','future_shared_evaluator'],'all_other_NMS_math_and_limits':'unchanged pinned implementation','missing_or_duplicate_processing':'fail before metrics','empty_detections':'legitimate if processed'},
      'execution_durability':{'journal':'append-only SHA256-chain JSONL, fsync each observed delivery/batch/step/epoch/validation event','snapshot':'atomic files + immutable epoch commit + content-addressed checkpoint copies','process':'foreground inherited process group; supervised actual exit including native cleanup; no restart/detach/automation'}}
    protocol['evaluation']={**parent['evaluation'],'NMS_time_policy':'unlimited_wall_clock; same frozen thresholds, max_det, math and matching/aggregation'}
    docs=ROOT/'docs/main_ac_recovery_v2';docs.mkdir(exist_ok=True)
    for name in ['freeze_receipt.json','execution_overlay.json']:
        path=docs/name
        if path.exists():
            history=ROOT/'results/recovery_v2_verification/contract_candidates'/file_sha256(path)
            history.mkdir(parents=True,exist_ok=True);shutil.copy2(path,history/name)
    atomic_json(ROOT/'configs/main_ac_recovery_v2.json',protocol)
    files={name:file_sha256(ROOT/name) for name in FILES};files['configs/main_ac_recovery_v2.json']=file_sha256(ROOT/'configs/main_ac_recovery_v2.json')
    version=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    snapshot=ROOT/'results/main_ac_recovery_v2/execution_snapshots'/version
    if not snapshot.exists():
        for name,sha in files.items():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target);assert file_sha256(target)==sha
    overlay={'version':'recovery-v2','files':files,'snapshot':str(snapshot.relative_to(ROOT)),'git_commit':None,'parent_overlay_sha256':file_sha256(ROOT/'docs/main_ac_v1/execution_overlay.json')}
    atomic_json(docs/'execution_overlay.json',overlay)
    parent_files={name:file_sha256(ROOT/name) for name in ['configs/main_ac_frozen_v1.json','docs/protocol_w4_v1/freeze_receipt.json','docs/main_ac_v1/execution_overlay.json','docs/main_ac_v1/interruption.json']}
    receipt={'version':'recovery-v2','protocol_sha256':file_sha256(ROOT/'configs/main_ac_recovery_v2.json'),'overlay_sha256':file_sha256(docs/'execution_overlay.json'),'parent_files':parent_files,'exposure_manifest_sha256':parent_receipt['exposure_manifest_sha256'],'parent_library_hashes':parent_receipt['installed_library_source_hashes'],'extra_library_source_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in extra_library_sources().items()},'budget':parent['budget_summary'],'commands':[f'PYTHONPATH=src PYTHONHASHSEED=42 PYTORCH_ENABLE_MPS_FALLBACK=0 YOLO_AUTOINSTALL=false .venv/bin/python scripts/supervise_recovery.py --model {model} --run-dir results/main_ac_recovery_v2/{model}' for model in ['A','C']],'status':'frozen candidate; run only after tests and all45-source preflight pass'}
    atomic_json(docs/'freeze_receipt.json',receipt);atomic_json(docs/'extra_library_sources.json',extra_library_sources())
    print('Protocol',receipt['protocol_sha256'],'overlay',receipt['overlay_sha256'])
