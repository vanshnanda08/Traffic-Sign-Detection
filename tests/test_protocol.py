"""W4 checks use temporary copies/synthetic pixels; no optimization or test IO."""
import csv
import inspect
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from p57.assets import file_sha256
from p57.dataset import ROOT
from p57.development import build_development_view, validate_development_view, development_records
from p57.degradation import degrade
from p57.online import OnlineAdapter, ScheduledYOLODataset
from p57.protocol import exposure_schedule, optimizer_budget, require_equal_budgets, validate_freeze


@pytest.fixture
def copied_view(tmp_path):
    source=build_development_view().parent
    target=tmp_path/'view';shutil.copytree(source,target)
    return target


def test_valid_authenticated_view(copied_view):
    assert validate_development_view(copied_view/'dataset.yaml')['nc']==6


def test_label_and_editable_lock_cotampering_rejected(copied_view):
    label=next((copied_view/'train/labels').glob('*.txt'))
    label.write_text('0 0.5 0.5 0.2 0.2\n')
    lock=json.loads((copied_view/'view.lock.json').read_text())
    lock['files'][label.relative_to(copied_view).as_posix()]=file_sha256(label)
    (copied_view/'view.lock.json').write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='authenticated frozen core'):
        validate_development_view(copied_view/'dataset.yaml')


def test_membership_and_lock_cotampering_rejected(copied_view):
    image=next((copied_view/'train/images').glob('*.jpg'));old=image.relative_to(copied_view).as_posix()
    new=image.with_name('unexpected.jpg');image.rename(new)
    lock=json.loads((copied_view/'view.lock.json').read_text());lock['files'][new.relative_to(copied_view).as_posix()]=lock['files'].pop(old)
    (copied_view/'view.lock.json').write_text(json.dumps(lock))
    with pytest.raises(ValueError,match='authenticated frozen core'):
        validate_development_view(copied_view/'dataset.yaml')


@pytest.mark.parametrize('kind',['image','label','extra'])
def test_links_and_unexpected_files_rejected(copied_view,tmp_path,kind):
    if kind=='extra':
        (copied_view/'train/labels/hidden.txt').write_text('0 .5 .5 .2 .2')
    else:
        path=next((copied_view/('train/images' if kind=='image' else 'train/labels')).glob('*'))
        outside=tmp_path/'outside';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    with pytest.raises(ValueError,match='External|Unexpected'):
        validate_development_view(copied_view/'dataset.yaml')


def training_rows():
    return [r for r in development_records() if r['split']=='train']


def test_schedule_exact_once_independent_of_rng_and_input_order():
    rows=training_rows();a=exposure_schedule(rows)
    random.seed(100);np.random.seed(7);torch.manual_seed(99)
    b=exposure_schedule(list(reversed(rows)))
    assert a==b and len(a)==10500
    for epoch in range(1,51):
        selected=[r['source_id'] for r in a if r['epoch']==epoch]
        assert len(set(selected))==len(selected)==210
    path=ROOT/'docs/protocol_w4_v1/exposure_manifest.csv'
    stored=list(csv.DictReader(path.open()))
    assert [{**r,'epoch':int(r['epoch'])} for r in stored]==a
    with pytest.raises(ValueError):exposure_schedule([{'source_id':'bad','split':'test'}])


def test_budget_matches_trial_every_record_and_partial_batches():
    trial=json.loads((ROOT/'results/feasibility/trial_v1_batch4/provenance.json').read_text())
    sim=optimizer_budget(3)
    assert sim['steps']==trial['optimizer_steps']
    assert sim['microbatch_sizes']==trial['microbatch_sizes'] and sim['microbatch_sizes'].count(2)==3
    assert sim['final_unstepped_images']==trial['pending_unstepped_images']==0


def test_main_budget_and_unequal_actual_rejection():
    budget=optimizer_budget(50)
    assert (budget['optimizer_update_count'],budget['microbatches'],budget['image_presentations'],budget['final_unstepped_images'])==(185,2650,10500,42)
    assert sum(s['actual_images_since_previous_step'] for s in budget['steps'])==10458
    require_equal_budgets(budget,budget,budget)
    bad={**budget,'optimizer_update_count':184}
    with pytest.raises(ValueError):require_equal_budgets(budget,bad,budget)


@pytest.mark.parametrize('condition',['clean','F1','F2','F3','O1','O2','O3'])
def test_actual_adapter_exact_transform_and_source_preservation(condition):
    row={'source_id':'fixture','split':'train','image_path':'train/images/fixture.jpg'}
    cfg=json.loads((ROOT/'configs/degradation_dev_v1.json').read_text())
    schedule=[{'epoch':1,'source_id':'fixture','condition':condition}]
    image=np.full((100,100,3),20,np.uint8);before=image.copy();boxes=[[0,.5,.5,.4,.4]]
    adapter=OnlineAdapter('C',[row],schedule,cfg)
    output,report=adapter.apply(image,boxes,row,1)
    expected=image if condition=='clean' else degrade(image,boxes,'fixture',condition,cfg)[0]
    np.testing.assert_array_equal(output,expected);np.testing.assert_array_equal(image,before)
    a=OnlineAdapter('A',[row],schedule,cfg);np.testing.assert_array_equal(a.apply(image,boxes,row,1)[0],image)
    with pytest.raises(ValueError):adapter.apply(image,boxes,{**row,'split':'test'},1)
    with pytest.raises(ValueError):adapter.apply(image,boxes,row,51)


def test_real_ultralytics_dataset_path_without_optimization_or_test_access(monkeypatch):
    from ultralytics.cfg import get_cfg
    rows=training_rows();schedule=exposure_schedule(rows)
    cfg=json.loads((ROOT/'configs/degradation_dev_v1.json').read_text())
    adapter=OnlineAdapter('C',rows,schedule,cfg)
    read=cv2.imread
    def guard(path,*args,**kwargs):
        assert '/test/' not in str(path)
        return read(path,*args,**kwargs)
    monkeypatch.setattr(cv2,'imread',guard)
    view=build_development_view();hyp=get_cfg();hyp.imgsz=416
    ds=ScheduledYOLODataset(view_yaml=view,mode='train',adapter=adapter,imgsz=416,batch_size=4,hyp=hyp,rect=False)
    index=next(i for i,f in enumerate(ds.im_files) if '57025_' in f) if any('57025_' in f for f in ds.im_files) else 0
    original=ds.labels[index]['bboxes'].copy()
    for epoch in [1,2,50]:
        ds.set_epoch(epoch);sample=ds[index]
        assert sample['img'].shape==(3,416,416) and torch.isfinite(sample['img']).all()
        assert ds.last_exposure['epoch']==epoch
        np.testing.assert_array_equal(ds.labels[index]['bboxes'],original)
    assert not any(type(t).__name__=='Albumentations' for t in ds.transforms.transforms)
    val=ScheduledYOLODataset(view_yaml=view,mode='val',imgsz=416,batch_size=4,hyp=get_cfg(),rect=False)
    assert val[0]['img'].shape==(3,416,416) and val.last_exposure=={}
    with pytest.raises(ValueError,match='Final-test'):
        ScheduledYOLODataset(view_yaml=view,mode='test',imgsz=416,hyp=get_cfg())


def test_installed_fitness_and_tie_conventions():
    from ultralytics.utils.metrics import Metric
    from ultralytics.engine.trainer import BaseTrainer
    metric=Metric();metric.all_ap=np.array([[.2]*10,[.4]*10]);assert metric.fitness()==pytest.approx(.3)
    dummy=SimpleNamespace(ema=None,world_size=1,loss=torch.tensor(0.),best_fitness=.3,validator=lambda _: {'fitness':.3})
    _,fitness=BaseTrainer.validate(dummy)
    assert dummy.best_fitness==fitness==.3
    assert 'if self.best_fitness == self.fitness:' in inspect.getsource(BaseTrainer.save_model)
    assert 'smooth(f1_curve.mean(0), 0.1).argmax()' in inspect.getsource(__import__('ultralytics.utils.metrics',fromlist=['ap_per_class']).ap_per_class)


def test_frozen_protocol_and_pinned_drift_rejection(tmp_path):
    _,receipt=validate_freeze()
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    with pytest.raises(ValueError,match='protocol hash drift'):
        validate_freeze(protocol)
    pinned=tmp_path/'config.json';pinned.write_text('changed')
    receipt['pinned_files']={str(pinned):'0'*64}
    rp=tmp_path/'receipt.json';rp.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='hash drift'):
        validate_freeze(receipt_path=rp)


def test_only_one_training_job_allowed_without_starting_training(tmp_path):
    from p57.protocol import training_job_lock
    with training_job_lock(tmp_path/'training.lock'):
        with pytest.raises(RuntimeError,match='Another training job'):
            with training_job_lock(tmp_path/'training.lock'):
                pytest.fail('Second lock unexpectedly acquired')
    with training_job_lock(tmp_path/'training.lock'):
        pass
