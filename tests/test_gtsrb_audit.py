"""Official-source audit evidence and conservative full-track exclusions."""

import csv
import json
from pathlib import Path

import pytest

from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.feasibility import validation_gallery


def test_missing_or_changed_checkpoint_rejected_before_inference(tmp_path):
    (tmp_path/'provenance.json').write_text(json.dumps({'checkpoint_hashes':{'best.pt':'not-a-real-hash'}}))
    path=tmp_path/'best.pt'
    with pytest.raises(ValueError,match='Missing, changed or unrecorded'):
        validation_gallery(tmp_path,path)
    path.write_bytes(b'synthetic corrupt checkpoint fixture')
    with pytest.raises(ValueError,match='Missing, changed or unrecorded'):
        validation_gallery(tmp_path,path)


def test_executed_trial_provenance_and_checkpoints_are_traceable():
    run=ROOT/'results/feasibility/trial_v1_batch4'
    if not (run/'provenance.json').exists():pytest.skip('Executed optional feasibility assets absent')
    p=json.loads((run/'provenance.json').read_text())
    assert p['status']=='passed' and len(p['epochs'])==3 and p['actual_device'].startswith('mps')
    assert p['strict_deterministic_algorithms'] is True
    assert len(p['optimizer_steps'])==p['optimizer_update_count']>0
    assert len(p['microbatch_sizes'])==159 and sum(p['microbatch_sizes'])==630
    assert p['resolved_config']['epochs']==3 and p['resolved_config']['batch']==4
    assert 'development' in p['resolved_config']['data']
    assert file_sha256(ROOT/p['initial_weights']['path'])==p['initial_weights']['sha256']
    for name,h in p['checkpoint_hashes'].items():assert file_sha256(run/'fit/weights'/name)==h
    for name,h in p['code_fingerprints'].items():assert file_sha256(run/'executed_source_snapshot'/name)==h


def test_official_overlap_exclusions_cover_complete_available_tracks():
    root=ROOT/'results/development_w2_v1'
    if not (root/'reference_training_inventory.csv').exists():pytest.skip('Optional official data audit absent')
    inventory=list(csv.DictReader((root/'reference_training_inventory.csv').open()))
    evidence=json.loads((root/'gtsrb_exact_and_inventory.json').read_text())
    excluded=set(evidence['excluded_training_tracks'])
    assert len(inventory)==39209
    for row in inventory:
        assert (row['excluded_for_potential_overlap']=='True')==(row['track'] in excluded)
        if row['class_id']=='40':assert row['compatible_transfer_class']=='False'
    assert sum(row['excluded_for_potential_overlap']=='True' for row in inventory)==evidence['excluded_training_images']
    near=list(csv.DictReader((root/'gtsrb_audit/accepted_overlaps.csv').open()))
    assert all(row['official_track'] in excluded for row in near if row['official_split']=='train')


def test_audit_does_not_claim_license_or_independence_clearance():
    path=ROOT/'results/development_w2_v1/gtsrb_audit/summary.json'
    if not path.exists():pytest.skip('Optional official data audit absent')
    report=json.loads(path.read_text())
    assert report['status']=='blocked' and report['blockers']
    assert report['physical_identity_or_independence_guaranteed'] is False
    assert report['official_images']==51839 and report['official_classes']==43
