"""Frozen W6 inputs/code, with no test execution during development checks."""
import hashlib,inspect,json
from pathlib import Path
from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.evaluation_data import CONFIG,config,source_rows
from p57.recovery_protocol import load_recovery

RECEIPT=ROOT/'docs/evaluation_v1/evaluator_receipt.json'
FILES=['src/p57/evaluation_data.py','src/p57/evaluation_engine.py','src/p57/evaluation_protocol.py','src/p57/evaluation_artifacts.py','scripts/evaluation_preflight.py','scripts/freeze_evaluator.py','scripts/evaluate_heldout.py','scripts/supervise_evaluations.py','scripts/evaluation_artifacts.py','tests/test_evaluation.py','configs/evaluation_v1.json']


def library_sources():
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics,ap_per_class,compute_ap
    from p57.complete_validation import PINNED_NMS
    objects={'NMS':PINNED_NMS,'base_validator':BaseValidator.__call__,'matching':BaseValidator.match_predictions,'prepare_batch':DetectionValidator._prepare_batch,'process_batch':DetectionValidator._process_batch,'scale_preds':DetectionValidator.scale_preds,'update_metrics':DetectionValidator.update_metrics,'get_stats':DetectionValidator.get_stats,'DetMetrics':DetMetrics.process,'AP':ap_per_class,'compute_AP':compute_ap}
    return {k:inspect.getsource(v) for k,v in objects.items()}


def authenticate_models():
    cfg=config();load_recovery(ROOT/'configs/main_ac_recovery_v2.json','A')
    if file_sha256(ROOT/'configs/main_ac_recovery_v2.json')!=cfg['replacement_protocol_sha256']:raise ValueError('Replacement protocol drift')
    verdict=json.loads((ROOT/'docs/main_ac_recovery_v2/pair_verification.json').read_text())
    if verdict['status']!='passed':raise ValueError('Replacement pair not verified')
    for model,record in cfg['checkpoints'].items():
        if file_sha256(ROOT/record['path'])!=record['sha256']:raise ValueError('Selected best checkpoint drift')
        run=ROOT/'results/main_ac_recovery_v2'/model
        state=json.loads((run/'provenance.json').read_text());exit_record=json.loads((run.parent/(model+'.supervision')/'exit.json').read_text())
        if state['status']!='passed' or exit_record['returncode']!=0 or state['checkpoint_hashes']['best.pt']!=record['sha256']:raise ValueError('Invalid trained-model lineage/exit')
    source_rows('test')  # metadata authentication only; no test decoding/inference


def verify_freeze():
    authenticate_models();r=json.loads(RECEIPT.read_text())
    if file_sha256(CONFIG)!=r['config_sha256']:raise ValueError('Evaluation config drift')
    for name,wanted in r['files'].items():
        if file_sha256(ROOT/name)!=wanted or file_sha256(ROOT/r['snapshot']/name)!=wanted:raise ValueError('Evaluator source/snapshot drift')
    actual={k:hashlib.sha256(v.encode()).hexdigest() for k,v in library_sources().items()}
    if actual!=r['library_hashes']:raise ValueError('Pinned evaluator library drift')
    preflight=ROOT/r['preflight_path']
    if file_sha256(preflight)!=r['preflight_sha256'] or json.loads(preflight.read_text())['status']!='passed':raise ValueError('Preflight gate not verified')
    return r
