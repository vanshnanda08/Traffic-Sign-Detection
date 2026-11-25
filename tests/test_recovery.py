"""Recovery regressions: no extra optimization experiment or final-test IO."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics.utils import nms

from p57.dataset import ROOT
from p57.complete_validation import complete_nms,PINNED_NMS,Coverage,CompleteValidator
from p57.durability import Journal,read_journal,atomic_json,epoch_commit,sha,atomic_checkpoint_targets
from p57.recovery_protocol import load_recovery
from p57.recovery_training import run_recovery
from p57.main_verification import verify_single as original_verify


def positive_batch():
    p=torch.zeros((3,5,8));p[:,0,0]=20;p[:,1,0]=20;p[:,2:4,0]=10;p[:,4,0]=.9
    return p


def test_default_timeout_loses_positive_images_complete_policy_keeps_all(monkeypatch):
    class SlowClock:
        def __init__(self):self.index=-1
        def time(self):self.index+=1;return self.index*10.
    monkeypatch.setattr(nms,'time',SlowClock())
    ordinary=PINNED_NMS(positive_batch(),conf_thres=.1,iou_thres=.7)
    monkeypatch.setattr(nms,'time',SlowClock())
    complete=complete_nms(positive_batch(),conf_thres=.1,iou_thres=.7)
    assert [len(x) for x in ordinary]==[1,0,0]
    assert [len(x) for x in complete]==[1,1,1]
    assert nms.non_max_suppression is PINNED_NMS


@pytest.mark.parametrize('multi_label',[True,False])
@pytest.mark.parametrize('options',[{}, {'classes':[1]}, {'max_det':2,'max_nms':3}, {'agnostic':True}, {'return_idxs':True}])
def test_ordinary_case_exact_equivalence_without_timeout(multi_label,options):
    torch.manual_seed(5);pred=torch.rand((3,7,8));pred[:,:4]*=30
    a=PINNED_NMS(pred.clone(),conf_thres=.2,iou_thres=.7,multi_label=multi_label,**options)
    b=complete_nms(pred.clone(),conf_thres=.2,iou_thres=.7,multi_label=multi_label,**options)
    groups=zip(a,b) if options.get('return_idxs') else [(a,b)]
    for aa,bb in groups:
        for x,y in zip(aa,bb):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_legitimate_empty_is_complete_but_missing_metrics_is_not():
    c=Coverage(['a','b']);c.nms_completed(['a','b'],[torch.zeros((0,6)),torch.zeros((1,6))])
    with pytest.raises(ValueError,match='incomplete'):c.finish()
    c.metrics_completed(['a','b']);assert c.finish()['complete']
    with pytest.raises(ValueError,match='Duplicate'):c.nms_completed(['a'],[torch.zeros((0,6))])


def test_incomplete_stats_are_rejected_before_aggregation():
    validator=object.__new__(CompleteValidator);validator.coverage=Coverage(['a','b']);validator.seen=1
    validator.coverage.nms_completed(['a'],[torch.ones((1,6))]);validator.coverage.metrics_completed(['a'])
    with pytest.raises(ValueError,match='metrics forbidden'):validator.get_stats()


def test_journal_durable_prefix_and_truncated_tail_preserved(tmp_path):
    path=tmp_path/'events.jsonl';journal=Journal(path);journal.append('actual_delivery',{'source_id':'fixture'});journal.close()
    with path.open('ab') as stream:stream.write(b'{"partial":')
    before=path.read_bytes();read=read_journal(path)
    assert len(read['events'])==1 and read['truncated_tail']['raw_utf8']=='{"partial":'
    assert path.read_bytes()==before
    with pytest.raises(FileExistsError):Journal(path)


def test_journal_cannot_forge_or_reorder_observed_events(tmp_path):
    path=tmp_path/'j';j=Journal(path);j.append('sample',{'id':1});j.append('sample',{'id':2});j.close()
    lines=path.read_bytes().splitlines(keepends=True);path.write_bytes(b''.join(reversed(lines)))
    with pytest.raises(ValueError,match='integrity'):read_journal(path)


def test_atomic_epoch_commit_retains_consistent_checkpoint_copy(tmp_path):
    weights=tmp_path/'fit/weights';weights.mkdir(parents=True);(weights/'best.pt').write_bytes(b'checkpoint-fixture');(weights/'last.pt').write_bytes(b'checkpoint-fixture')
    j=Journal(tmp_path/'events.jsonl');j.append('actual_step',{'step':1})
    manifest=epoch_commit(tmp_path,'epoch-001',{'status':'in_progress'},{'steps':[1]},[{'id':'actual'}],j);j.close()
    obj=tmp_path/manifest['checkpoints']['last.pt']['object'];before=obj.read_bytes()
    (weights/'last.pt').write_bytes(b'newer-current-checkpoint')
    assert obj.read_bytes()==before and sha(obj)==manifest['checkpoints']['last.pt']['sha256']
    assert json.loads((tmp_path/'latest_commit.json').read_text())['label']=='epoch-001'
    assert json.loads((tmp_path/'commits/epoch-001/exposures.json').read_text())==[{'id':'actual'}]


def test_atomic_checkpoint_stage_failure_keeps_original(tmp_path):
    w=tmp_path/'fit/weights';w.mkdir(parents=True);(w/'last.pt').write_bytes(b'old');(w/'best.pt').write_bytes(b'best')
    trainer=SimpleNamespace(wdir=w,last=w/'last.pt',best=w/'best.pt')
    with pytest.raises(RuntimeError):
        with atomic_checkpoint_targets(trainer,tmp_path,'fixture'):
            trainer.last.write_bytes(b'partial');raise RuntimeError('fixture interruption')
    assert trainer.last==w/'last.pt' and trainer.last.read_bytes()==b'old'
    assert (tmp_path/'checkpoint_staging/fixture/last.pt').read_bytes()==b'partial'


def test_standard_JSON_and_recovery_freeze(tmp_path):
    protocol,receipt,overlay,*_=load_recovery(ROOT/'configs/main_ac_recovery_v2.json','A')
    assert protocol['validation_completeness']['time_budget']=='unlimited_wall_clock'
    p=tmp_path/'bad.json';p.write_text('{}')
    with pytest.raises(ValueError,match='drift'):load_recovery(p,'A')
    with pytest.raises(ValueError):atomic_json(tmp_path/'nan.json',{'bad':float('inf')})


def test_existing_run_and_original_incomplete_C_remain_rejected(tmp_path):
    with pytest.raises(ValueError,match='directory exists'):run_recovery(ROOT/'configs/main_ac_recovery_v2.json','A',tmp_path)
    with pytest.raises(ValueError,match='every main training epoch'):original_verify(ROOT/'results/main_ac_v1/C',load_weights=False)


def test_coverage_rejects_omitted_source_even_with_valid_empty_outputs():
    coverage=Coverage(['a','b','c'])
    coverage.nms_completed(['a','b'],[torch.zeros((0,6)),torch.zeros((0,6))])
    coverage.metrics_completed(['a','b'])
    with pytest.raises(ValueError,match='incomplete'):
        coverage.finish()
