"""W4 protocol, deterministic exposure allocation and pinned-trainer accounting.

No optimization or model inference is performed by this module.
"""
import csv
import contextlib
import fcntl
import hashlib
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from p57.assets import file_sha256
from p57.dataset import ROOT
from p57.degradation import stable_seed
from p57.development import EXPECTED_LOCK, validate_development_view

INITIAL_SHA = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
CONDITIONS = ('clean', 'F1', 'F2', 'F3', 'O1', 'O2', 'O3')


def exposure_schedule(rows, epochs=50):
    """One independent SHA-derived categorical draw per original/epoch.

    Six of twelve buckets are clean, one bucket per degraded condition. Source
    seed is domain-separated from initialization/shuffle and degradation masks.
    """
    rows = sorted(rows, key=lambda r: r['source_id'])
    if len({r['source_id'] for r in rows}) != len(rows) or any(r['split'] != 'train' for r in rows):
        raise ValueError('Exposure schedule accepts unique training originals only')
    schedule = []
    for epoch in range(1, epochs+1):
        for row in rows:
            # Rejection sampling avoids modulo bias in the declared 12 buckets.
            rng = np.random.Generator(np.random.PCG64(stable_seed(42, row['source_id'], ['W4-exposure-v1', epoch])))
            bucket = int(rng.integers(0, 12))
            condition = 'clean' if bucket < 6 else CONDITIONS[bucket-5]
            schedule.append({'epoch':epoch, 'source_id':row['source_id'], 'image_path':row['image_path'],
                             'label_path':row['label_path'], 'classes':row['remapped_class_ids'], 'condition':condition})
    return schedule


def exposure_counts(schedule):
    totals = Counter(); by_class = defaultdict(Counter)
    for row in schedule:
        totals[row['condition']] += 1
        for cls in set(row['classes'].split(';')):
            by_class[cls][row['condition']] += 1
    return {'total':dict(totals), 'by_class_image_exposure':{k:dict(v) for k,v in sorted(by_class.items())}}


def optimizer_budget(epochs, images=210, batch=4, nbs=64, warmup_epochs=3.0):
    """Pinned BaseTrainer loop semantics; no optimizer is constructed.

    zero_grad runs at training start and after each optimizer step. Gradients
    carry across epoch boundaries. The final partial batch is two images.
    """
    nb = math.ceil(images/batch)
    nw = max(round(warmup_epochs*nb), 100) if warmup_epochs>0 else -1
    last_opt_step = -1; accumulate = max(round(nbs/batch),1)
    steps=[]; tails=[]; micro=[]; pending=0
    for epoch in range(epochs):
        for i in range(nb):
            ni=i+nb*epoch
            if ni<=nw:
                accumulate=max(1,int(np.interp(ni,[0,nw],[1,nbs/batch]).round()))
            count=min(batch,images-i*batch);pending+=count;micro.append(count)
            if ni-last_opt_step>=accumulate:
                steps.append({'epoch':epoch+1,'microbatches_seen':ni+1,'accumulation_setting':accumulate,
                              'actual_images_since_previous_step':pending})
                pending=0;last_opt_step=ni
        tails.append({'epoch':epoch+1,'pending_unstepped_images':pending})
    return {'epochs':epochs,'microbatches':len(micro),'image_presentations':sum(micro),
            'optimizer_update_count':len(steps),'steps':steps,'epoch_tails':tails,
            'final_unstepped_images':pending,
            'microbatch_sizes':micro,'nominal_steady_accumulation':accumulate}


def require_equal_budgets(a,b,expected):
    """Future run gate: both runs must match the complete frozen accounting."""
    for actual in (a,b):
        for key in ('optimizer_update_count','image_presentations','microbatches','steps','epoch_tails'):
            if actual.get(key)!=expected[key]:
                raise ValueError(f'Unequal or drifted actual training budget: {key}')


def library_sources():
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.metrics import Metric, ap_per_class, compute_ap
    from ultralytics.utils.torch_utils import ModelEMA
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.models.yolo.detect import DetectionValidator
    names = {'train_loop':BaseTrainer._do_train,'setup':BaseTrainer._setup_train,
             'scheduler':BaseTrainer._setup_scheduler,'optimizer':BaseTrainer.build_optimizer,
             'optimizer_step':BaseTrainer.optimizer_step,'validate':BaseTrainer.validate,
             'save_model':BaseTrainer.save_model,'fitness':Metric.fitness,
             'ap_per_class':ap_per_class,'compute_ap':compute_ap,'EMA':ModelEMA,
             'validator_call':BaseValidator.__call__,'detector_validator_init':DetectionValidator.__init__}
    return {name:inspect.getsource(obj) for name,obj in names.items()}


def validate_freeze(protocol_path='configs/main_ac_frozen_v1.json', receipt_path='docs/protocol_w4_v1/freeze_receipt.json'):
    protocol_path=ROOT/protocol_path;receipt_path=ROOT/receipt_path
    receipt=json.loads(receipt_path.read_text())
    if file_sha256(protocol_path)!=receipt['protocol_sha256']:
        raise ValueError('Frozen protocol hash drift')
    for name,sha in receipt['pinned_files'].items():
        if file_sha256(ROOT/name)!=sha:
            raise ValueError(f'Frozen configuration/source/input hash drift: {name}')
    actual={name:hashlib.sha256(text.encode()).hexdigest() for name,text in library_sources().items()}
    if actual!=receipt['installed_library_source_hashes']:
        raise ValueError('Pinned trainer/evaluator implementation drift')
    validate_development_view(ROOT/'data/processed/p57_development_v1/dataset.yaml')
    return json.loads(protocol_path.read_text()),receipt


@contextlib.contextmanager
def training_job_lock(path=None):
    """Shared with feasibility; a second detector job must fail, not race."""
    path=Path(path) if path else ROOT/'.cache/p57_training.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another training job holds the shared lock') from exc
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)
