"""Parent-authenticated v2 replacement contract; v1 is preserved, not repinned."""
import hashlib
import inspect
import json
from pathlib import Path
from p57.assets import file_sha256
from p57.dataset import ROOT
from p57.main_training import load_contract as load_parent
from p57.complete_validation import PINNED_NMS

PROTOCOL=ROOT/'configs/main_ac_recovery_v2.json'
RECEIPT=ROOT/'docs/main_ac_recovery_v2/freeze_receipt.json'
OVERLAY=ROOT/'docs/main_ac_recovery_v2/execution_overlay.json'


def extra_library_sources():
    from ultralytics.models.yolo.detect import DetectionValidator,DetectionPredictor
    return {name:inspect.getsource(obj) for name,obj in {
        'non_max_suppression':PINNED_NMS,'validator_postprocess':DetectionValidator.postprocess,
        'validator_update_metrics':DetectionValidator.update_metrics,'validator_get_stats':DetectionValidator.get_stats,
        'predictor_postprocess':DetectionPredictor.postprocess}.items()}


def load_recovery(protocol_path,variant):
    parent,parent_receipt,_,rows,schedule,expected=load_parent('configs/main_ac_frozen_v1.json',variant)
    receipt=json.loads(RECEIPT.read_text());protocol=json.loads(Path(protocol_path).read_text());overlay=json.loads(OVERLAY.read_text())
    if file_sha256(protocol_path)!=receipt['protocol_sha256'] or receipt['overlay_sha256']!=file_sha256(OVERLAY):
        raise ValueError('Recovery protocol/overlay drift')
    for name,wanted in receipt['parent_files'].items():
        if file_sha256(ROOT/name)!=wanted:raise ValueError('Parent artifact drift')
    if protocol['training']!=parent['training'] or protocol['budget_summary']!=parent['budget_summary']:
        raise ValueError('Undeclared scientific training change')
    if protocol['validation_completeness']['time_budget']!='unlimited_wall_clock':raise ValueError('Incomplete validation policy')
    for name,wanted in overlay['files'].items():
        if file_sha256(ROOT/name)!=wanted or file_sha256(ROOT/overlay['snapshot']/name)!=wanted:
            raise ValueError(f'Recovery execution source drift: {name}')
    actual={name:hashlib.sha256(text.encode()).hexdigest() for name,text in extra_library_sources().items()}
    if actual!=receipt['extra_library_source_hashes']:raise ValueError('NMS/evaluator source drift')
    return protocol,receipt,overlay,rows,schedule,expected
