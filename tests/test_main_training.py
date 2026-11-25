"""Additive W5 execution tests; no optimization experiment is performed."""
import json
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader

from p57.dataset import ROOT
from p57.development import build_development_view
from p57.online import OnlineAdapter, ScheduledYOLODataset
from p57.main_training import (canonical_state_hash,load_contract,ExposureLedger,
    LoggedScheduledDataset,check_exposures,record_failure,run_main,strict_runtime)
from p57.protocol import optimizer_budget,require_equal_budgets,training_job_lock


def test_initialization_hash_is_tensor_canonical_and_detects_changed_head():
    torch.manual_seed(42);a=torch.nn.Linear(4,6)
    torch.manual_seed(42);b=torch.nn.Linear(4,6)
    assert canonical_state_hash(a)==canonical_state_hash(b)
    with torch.no_grad():b.bias[0]+=1
    assert canonical_state_hash(a)!=canonical_state_hash(b)
    state=a.state_dict();reverse=dict(reversed(list(state.items())))
    assert canonical_state_hash(SimpleNamespace(state_dict=lambda:reverse))==canonical_state_hash(a)


def test_contract_guard_invalid_variant_and_new_overlay():
    p,r,o,rows,schedule,expected=load_contract('configs/main_ac_frozen_v1.json','A')
    assert len(rows)==210 and len(schedule)==10500 and expected['optimizer_update_count']==185
    with pytest.raises(ValueError,match='Only frozen'):load_contract('configs/main_ac_frozen_v1.json','B')


def test_changed_protocol_rejected_in_fixture(tmp_path):
    p=tmp_path/'protocol.json';p.write_text('{}')
    with pytest.raises(ValueError,match='protocol hash drift'):load_contract(p,'A')


def test_existing_run_rejected_before_trainer(tmp_path):
    with pytest.raises(ValueError,match='directory exists'):run_main('configs/main_ac_frozen_v1.json','A',tmp_path)


@pytest.mark.parametrize('override',[['--resume'],['--epochs','1'],['--device','cpu'],['--model','B']])
def test_cli_forbids_resume_and_training_overrides(override):
    cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/train_main.py'),'--protocol','configs/main_ac_frozen_v1.json','--model','A','--run-dir','unused-fixture-path',*override]
    result=subprocess.run(cmd,cwd=ROOT,env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
    assert result.returncode==2
    assert not (ROOT/'unused-fixture-path').exists()


def test_ledger_observes_actual_step_events_with_tail():
    ledger=ExposureLedger()
    ledger.batch([{'epoch':1,'source_id':str(i),'condition':'clean'} for i in range(4)],1)
    ledger.optimizer_step(1,1,1)
    ledger.batch([{'epoch':1,'source_id':str(i),'condition':'F1'} for i in range(2)],1)
    ledger.epoch_end(1)
    assert ledger.steps==[{'epoch':1,'microbatches_seen':1,'accumulation_setting':1,'actual_images_since_previous_step':4}]
    assert ledger.pending==[5,6] and ledger.ema_updates==[1]
    assert ledger.budget()['final_unstepped_images']==2
    with pytest.raises(ValueError):ledger.batch([{'epoch':2}],1)


def test_actual_delivered_logging_consumes_no_randomness(monkeypatch,tmp_path):
    ds=object.__new__(LoggedScheduledDataset);ds.mode='train';ds.view=tmp_path;ds.delivery_queue=[]
    ds.last_exposure={'epoch':2,'source_id':'fixture','condition':'O2'}
    path=tmp_path/'train/images/fixture.jpg'
    monkeypatch.setattr(ScheduledYOLODataset,'__getitem__',lambda self,index:{'im_file':str(path)})
    before=torch.random.get_rng_state().clone();ds[0]
    assert torch.equal(before,torch.random.get_rng_state())
    records=ds.delivered_batch([str(path)])
    assert records[0]['condition']=='O2' and records[0]['epoch']==2 and ds.delivery_queue==[]
    with pytest.raises(ValueError):ds.delivered_batch([str(path)])


def test_dataset_epoch_callback_wiring_with_real_dataloader_only():
    _,_,_,rows,schedule,_=load_contract('configs/main_ac_frozen_v1.json','C')
    degradation=json.loads((ROOT/'configs/degradation_dev_v1.json').read_text())
    adapter=OnlineAdapter('C',rows,schedule,degradation)
    ds=LoggedScheduledDataset(view_yaml=build_development_view(),mode='train',adapter=adapter,imgsz=416,batch_size=4,hyp=get_cfg(),rect=False)
    loader=build_dataloader(ds,batch=4,workers=0,shuffle=True,rank=-1)
    for epoch in [1,2]:
        ds.set_epoch(epoch);records=[]
        for batch in loader:records.extend(ds.delivered_batch(batch['im_file']))
        assert len(records)==210 and len({r['source_id'] for r in records})==210
        assert all(r['epoch']==epoch for r in records)
        expected={(r['epoch'],r['source_id']):r['condition'] for r in schedule}
        assert all(r['condition']==expected[(epoch,r['source_id'])] for r in records)
    with pytest.raises(ValueError,match='Final-test'):LoggedScheduledDataset(view_yaml=build_development_view(),mode='test',imgsz=416,hyp=get_cfg())


def test_exposure_membership_and_condition_rejection():
    rows=[{'source_id':'a','split':'train','image_path':'train/images/a.jpg'}]
    schedule=[{'epoch':1,'source_id':'a','condition':'F1'}]
    exposure=[{'epoch':1,'source_id':'a','condition':'clean','image_path':'train/images/a.jpg'}]
    check_exposures(exposure,rows,schedule,'A',epochs=1)
    with pytest.raises(ValueError,match='schedule'):check_exposures(exposure,rows,schedule,'C',epochs=1)
    with pytest.raises(ValueError):check_exposures(exposure*2,rows,schedule,'A',epochs=1)


def test_failure_record_retains_traceback_without_retry():
    state={}
    try:raise RuntimeError('synthetic fixture failure')
    except RuntimeError as error:record_failure(state,error)
    assert state['status']=='failed' and 'synthetic fixture failure' in state['traceback']


def test_shared_lock_and_runtime_environment_guards(tmp_path,monkeypatch):
    with training_job_lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError):
            with training_job_lock(tmp_path/'lock'):pass
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(RuntimeError,match='fallback'):strict_runtime()
