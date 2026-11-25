"""W5-only execution overlay for the unmodified W4 detector contract."""
import csv
import hashlib
import json
import math
import os
import shutil
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

from p57.assets import file_sha256
from p57.dataset import ROOT, save_json
from p57.development import development_records, validate_development_view
from p57.environment import seed_everything, validate_pins
from p57.feasibility import memory_sample
from p57.online import ScheduledYOLODataset, OnlineAdapter
from p57.protocol import validate_freeze, training_job_lock, require_equal_budgets

OVERLAY = ROOT/'docs/main_ac_v1/execution_overlay.json'


def canonical_state_hash(model):
    """Sorted tensor names, dtype, shape, and contiguous CPU bytes (not pickle)."""
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        tensor=tensor.detach().cpu().contiguous()
        header=json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()
        raw=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8,'big'));digest.update(header)
        digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()


def load_contract(protocol_path, variant):
    if variant not in {'A','C'}:
        raise ValueError('Only frozen variants A and C are permitted')
    protocol,receipt=validate_freeze(protocol_path)
    overlay=json.loads(OVERLAY.read_text())
    if overlay['protocol_sha256']!=receipt['protocol_sha256'] or overlay['W4_receipt_sha256']!=file_sha256(ROOT/'docs/protocol_w4_v1/freeze_receipt.json'):
        raise ValueError('Execution overlay does not match W4')
    for name,sha in overlay['files'].items():
        if file_sha256(ROOT/name)!=sha or file_sha256(ROOT/overlay['snapshot']/name)!=sha:
            raise ValueError(f'Execution overlay source drift: {name}')
    cfg=dict(protocol['training'])
    if cfg['resume'] or cfg['epochs']!=50 or cfg['device']!='mps':
        raise ValueError('Resume/overrides outside frozen protocol')
    view=ROOT/cfg['data'];validate_development_view(view)
    rows=[r for r in development_records() if r['split']=='train']
    with (ROOT/protocol['exposure']['manifest']).open() as stream:
        schedule=[{**r,'epoch':int(r['epoch'])} for r in csv.DictReader(stream)]
    expected=json.loads((ROOT/'docs/protocol_w4_v1/optimizer_budget.json').read_text())
    return protocol,receipt,overlay,rows,schedule,expected


class ExposureLedger:
    """Observe dataset deliveries and optimizer events, independent of RNG."""
    def __init__(self):
        self.exposures=[];self.microbatch_sizes=[];self.steps=[];self.epoch_tails=[]
        self.pending=[];self.ema_updates=[]

    def batch(self, records, epoch):
        if not records or any(r['epoch']!=epoch for r in records):
            raise ValueError('Exposure epoch wiring mismatch')
        self.microbatch_sizes.append(len(records))
        for row in records:
            record={**row,'presentation':len(self.exposures)+1,'microbatch':len(self.microbatch_sizes)}
            self.exposures.append(record);self.pending.append(record['presentation'])

    def optimizer_step(self,epoch,accumulation,ema_updates):
        self.steps.append({'epoch':epoch,'microbatches_seen':len(self.microbatch_sizes),
                           'accumulation_setting':int(accumulation),'actual_images_since_previous_step':len(self.pending)})
        self.pending=[];self.ema_updates.append(int(ema_updates))

    def epoch_end(self,epoch):
        self.epoch_tails.append({'epoch':epoch,'pending_unstepped_images':len(self.pending)})

    def budget(self):
        return {'epochs':len(self.epoch_tails),'microbatches':len(self.microbatch_sizes),
                'image_presentations':len(self.exposures),'optimizer_update_count':len(self.steps),
                'steps':self.steps,'epoch_tails':self.epoch_tails,'microbatch_sizes':self.microbatch_sizes,
                'final_unstepped_images':len(self.pending)}


def check_exposures(exposures,rows,schedule,variant,epochs=50):
    by_id={r['source_id']:r for r in rows}
    conditions={(int(r['epoch']),r['source_id']):r['condition'] for r in schedule}
    if len(exposures)!=len(rows)*epochs:
        raise ValueError('Incomplete or extra source presentations')
    for epoch in range(1,epochs+1):
        actual=[r for r in exposures if r['epoch']==epoch]
        if Counter(r['source_id'] for r in actual)!=Counter(by_id.keys()):
            raise ValueError('Source membership/duplicate/missing exposure')
        for row in actual:
            wanted='clean' if variant=='A' else conditions[(epoch,row['source_id'])]
            if row['condition']!=wanted or row['image_path']!=by_id[row['source_id']]['image_path']:
                raise ValueError('Actual exposure differs from frozen schedule')


class LoggedScheduledDataset(ScheduledYOLODataset):
    def __init__(self,**kwargs):
        self.delivery_queue=[]
        super().__init__(**kwargs)

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        if self.mode=='train':
            row=self.last_exposure
            self.delivery_queue.append({'epoch':row['epoch'],'source_id':row['source_id'],
                'condition':row['condition'],'image_path':Path(sample['im_file']).relative_to(self.view).as_posix()})
        return sample

    def delivered_batch(self,paths):
        records=self.delivery_queue[:len(paths)]
        if len(records)!=len(paths) or [r['image_path'] for r in records]!=[Path(p).relative_to(self.view).as_posix() for p in paths]:
            raise ValueError('Unlogged/extra/out-of-order dataset delivery')
        del self.delivery_queue[:len(paths)]
        return records


def strict_runtime():
    if os.environ.get('PYTHONHASHSEED')!='42' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':
        raise RuntimeError('Require PYTHONHASHSEED=42 and MPS fallback=0')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Frozen MPS device unavailable')
    seed_everything(42,True)
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError('Strict determinism required')


def record_failure(state,error):
    state.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())


def run_main(protocol_path,variant,run_dir):
    # All input/source checks happen before creating any optimizer.
    protocol,receipt,overlay,rows,schedule,expected=load_contract(protocol_path,variant)
    run_dir=Path(run_dir).absolute()
    if run_dir.exists():
        raise ValueError('Run directory exists; preserve evidence and choose a matched revision')
    with training_job_lock():
        run_dir.mkdir(parents=True,exist_ok=False)
        started=time.monotonic();stop=threading.Event();sampler=None
        ledger=ExposureLedger()
        state={'status':'in_progress','variant':variant,'scope':'Main detector training + clean validation only; no final-test evaluation',
               'started_utc':datetime.now(timezone.utc).isoformat(),'requested_config':protocol['training'],
               'protocol_sha256':receipt['protocol_sha256'],'overlay_sha256':file_sha256(OVERLAY),
               'exposure_manifest_sha256':receipt['exposure_manifest_sha256'],
               'dataset_lock_sha256':protocol['dataset_lock_sha256'],'initial_weights_sha256':protocol['initial_weights_sha256'],
               'epochs':[],'checkpoint_writes':[],'validation_passes':[],'memory_samples':[],
               'memory_method':'One-second psutil RSS/system available/global swap + torch.mps tensor/driver allocations. Observed samples, not guaranteed peaks; unified counters not additive; global swap not solely attributable to training.',
               'frozen_receipt_sha256':file_sha256(ROOT/'docs/protocol_w4_v1/freeze_receipt.json')}
        trainer=None
        def persist():
            save_json(run_dir/'provenance.json',state)
            save_json(run_dir/'actual_accounting.json',ledger.budget())
            save_json(run_dir/'exposures.json',ledger.exposures)
        persist()
        try:
            strict_runtime();state['packages']=validate_pins();torch.set_num_threads(2)
            disk=shutil.disk_usage(ROOT);state['preflight']={'disk_free_bytes':disk.free,**memory_sample()}
            if disk.free<2_000_000_000:
                raise RuntimeError('Less than 2GB disk headroom; no training started')
            state['environment']={k:os.environ.get(k) for k in ['PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','YOLO_AUTOINSTALL','YOLO_CONFIG_DIR','MPLCONFIGDIR']}
            state['hardware']={'machine':os.uname().machine,'ram_bytes':__import__('psutil').virtual_memory().total,'cpu':'Apple M3'}
            state['source_fingerprints']=overlay['files'];state['library_fingerprints']=receipt['installed_library_source_hashes']
            view=ROOT/protocol['training']['data']
            degradation=json.loads((ROOT/protocol['degradation_config']).read_text())
            adapter=OnlineAdapter(variant,rows,schedule,degradation)
            epoch_clock={};validation_clock={}
            class MainTrainer(DetectionTrainer):
                def get_dataset(self):
                    data=super().get_dataset()
                    if data.get('test') or Path(data['train']).resolve()!=view.parent/'train/images' or Path(data['val']).resolve()!=view.parent/'val/images':
                        raise ValueError('Forbidden model input outside train/clean-validation')
                    return data

                def build_dataset(self,img_path,mode='train',batch=None):
                    if mode not in {'train','val'} or Path(img_path).resolve()!=view.parent/mode/'images':
                        raise ValueError('Forbidden dataset access')
                    return LoggedScheduledDataset(view_yaml=view,mode=mode,adapter=adapter if mode=='train' else None,
                        imgsz=self.args.imgsz,batch_size=batch,hyp=self.args,rect=mode=='val',stride=32)

                def preprocess_batch(self,batch):
                    records=self.train_loader.dataset.delivered_batch(batch['im_file'])
                    ledger.batch(records,self.epoch+1)
                    return super().preprocess_batch(batch)

                def optimizer_step(self):
                    if not torch.isfinite(self.loss).all():
                        raise RuntimeError('Nonfinite training loss; recovery forbidden')
                    super().optimizer_step()
                    ledger.optimizer_step(self.epoch+1,self.accumulate,self.ema.updates)

                def _handle_nan_recovery(self,epoch):
                    if not torch.isfinite(self.loss).all() or not math.isfinite(float(self.fitness)):
                        raise RuntimeError('Nonfinite loss/fitness; no replay or resume permitted')
                    return False

                def save_model(self):
                    super().save_model()
                    state['checkpoint_writes'].append({'epoch':self.epoch+1,'fitness':float(self.fitness),
                        'best_fitness':float(self.best_fitness),'best_written':self.fitness==self.best_fitness,
                        'EMA_updates':self.ema.updates,'best_sha256':file_sha256(self.best),'last_sha256':file_sha256(self.last)})

            def train_start(t):
                strict_runtime()  # Library setup enables warn_only; restore strict mode.
                state['effective_initialization_sha256']=canonical_state_hash(t.model)
                if variant=='C':
                    a=json.loads((run_dir.parent/'A/provenance.json').read_text())
                    if a['status']!='passed' or a['effective_initialization_sha256']!=state['effective_initialization_sha256'] or a['overlay_sha256']!=state['overlay_sha256']:
                        raise RuntimeError('C must match completed A initialization and overlay before any update')
                state['actual_device']=str(next(t.model.parameters()).device)
                if not state['actual_device'].startswith('mps'):
                    raise RuntimeError('Device changed from MPS')
                state['strict_determinism']=True;state['resolved_config']=vars(t.args).copy()
                state['optimizer']={'type':type(t.optimizer).__name__,'groups':[{k:v for k,v in g.items() if k!='params'} for g in t.optimizer.param_groups]}
                state['setup_seconds']=time.monotonic()-started
                for k,v in protocol['training'].items():
                    wanted=str((ROOT/v).resolve()) if k in {'model','data'} else v
                    if getattr(t.args,k)!=wanted:
                        raise ValueError(f'Resolved scientific configuration drift: {k}')
                persist()

            def epoch_start(t):
                if ledger.epoch_tails and t.epoch+1!=len(ledger.epoch_tails)+1:
                    raise RuntimeError('Epoch replay/skip forbidden')
                if t.train_loader.dataset.delivery_queue:
                    raise RuntimeError('Cached/prefetched exposures crossed epoch boundary')
                t.train_loader.dataset.set_epoch(t.epoch+1);epoch_clock['start']=time.monotonic()

            def batch_end(t):
                if not torch.isfinite(t.loss).all():
                    raise RuntimeError('Nonfinite loss; no retry permitted')
                if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
                    raise RuntimeError('Determinism drift during training')

            def train_epoch_end(t):
                epoch_clock['train_end']=time.monotonic()
                ledger.epoch_end(t.epoch+1)
                check_exposures(ledger.exposures,rows,schedule,variant,epochs=t.epoch+1)
                if t.train_loader.dataset.delivery_queue:
                    raise RuntimeError('Unconsumed training samples')

            def fit_epoch_end(t):
                if len(state['epochs'])>=t.epoch+1:
                    return  # final best-checkpoint validation is not another training epoch
                metrics={k:float(v) for k,v in t.metrics.items()}
                if not all(math.isfinite(v) for v in metrics.values()):
                    raise RuntimeError('Nonfinite clean-validation metric')
                state['epochs'].append({'epoch':t.epoch+1,'train_seconds':epoch_clock['train_end']-epoch_clock['start'],
                    'train_validation_save_seconds':time.monotonic()-epoch_clock['start'],
                    'clean_validation_metrics':metrics,'training_losses':[float(v) for v in t.tloss.detach().cpu()],
                    'optimizer_updates':len(ledger.steps),'pending_images':len(ledger.pending),**memory_sample()})
                persist()

            def val_start(v):validation_clock['start']=time.monotonic()
            def val_end(v):state['validation_passes'].append({'seconds':time.monotonic()-validation_clock['start'],'scope':'clean validation only'})
            def sample_loop():
                while not stop.is_set():
                    state['memory_samples'].append({'elapsed':time.monotonic()-started,**memory_sample()});stop.wait(1)
            sampler=threading.Thread(target=sample_loop,daemon=True);sampler.start()
            cfg=dict(protocol['training']);cfg.update(model=str(ROOT/cfg['model']),data=str(view),project=str(run_dir),name='fit',exist_ok=False)
            trainer=MainTrainer(overrides=cfg)
            for event,callback in [('on_train_start',train_start),('on_train_epoch_start',epoch_start),('on_train_batch_end',batch_end),
                                   ('on_train_epoch_end',train_epoch_end),('on_fit_epoch_end',fit_epoch_end),('on_val_start',val_start),('on_val_end',val_end)]:
                trainer.add_callback(event,callback)
            trainer.train()
            actual=ledger.budget();require_equal_budgets(actual,actual,expected)
            if actual['microbatch_sizes']!=expected['microbatch_sizes'] or actual['final_unstepped_images']!=42 or ledger.ema_updates!=list(range(1,186)) or len(state['epochs'])!=50:
                raise RuntimeError('Actual main budget/EMA/tail contract failed')
            check_exposures(ledger.exposures,rows,schedule,variant)
            state['EMA_updates']=ledger.ema_updates
            state['pending_source_exposures']=[ledger.exposures[i-1] for i in ledger.pending]
            state['optimizer_consumed_presentations']=sum(s['actual_images_since_previous_step'] for s in ledger.steps)
            state['checkpoint_hashes']={p.name:file_sha256(p) for p in (run_dir/'fit/weights').glob('*.pt')}
            if set(state['checkpoint_hashes'])!={'best.pt','last.pt'}:
                raise RuntimeError('Expected best/last checkpoints missing or unexpected checkpoint')
            best=[w for w in state['checkpoint_writes'] if w['best_written']][-1]
            state['selected_epoch']=best['epoch'];state['selected_fitness']=best['fitness']
            state['final_clean_validation_metrics']={k:float(v) for k,v in trainer.metrics.items()}
            state['status']='passed';persist()
            return state
        except BaseException as error:
            record_failure(state,error);persist();raise
        finally:
            stop.set()
            if sampler:sampler.join(timeout=3)
            state['finished_utc']=datetime.now(timezone.utc).isoformat();state['total_elapsed_seconds']=time.monotonic()-started
            if state['memory_samples']:
                state['observed_memory_maxima']={k:max(r[k] for r in state['memory_samples']) for k in state['memory_samples'][0] if k.endswith('bytes')}
                state['minimum_available_bytes']=min(r['system_available_bytes'] for r in state['memory_samples'])
            state['EMA_updates']=ledger.ema_updates
            persist()
