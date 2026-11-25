"""Recovery-v2 replacement trainer, derived from preserved W5-v1 implementation."""
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

from p57.main_training import canonical_state_hash, ExposureLedger, check_exposures, LoggedScheduledDataset, strict_runtime, record_failure
from p57.recovery_protocol import load_recovery, OVERLAY
from p57.complete_validation import CompleteValidator
from p57.durability import Journal, atomic_json, epoch_commit, atomic_checkpoint_targets



def run_recovery(protocol_path,variant,run_dir):
    # All input/source checks happen before creating any optimizer.
    protocol,receipt,overlay,rows,schedule,expected=load_recovery(protocol_path,variant)
    run_dir=Path(run_dir).absolute()
    if run_dir.exists():
        raise ValueError('Run directory exists; preserve evidence and choose a matched revision')
    with training_job_lock():
        run_dir.mkdir(parents=True,exist_ok=False)
        started=time.monotonic();stop=threading.Event();sampler=None
        journal=Journal(run_dir/'events.jsonl')
        journal.append('run_start',{'pid':os.getpid(),'parent_pid':os.getppid(),'variant':variant,'run_id':str(run_dir),
                                  'command':__import__('sys').argv,'protocol_sha256':receipt['protocol_sha256']})
        class DurableLedger(ExposureLedger):
            def batch(self,records,epoch):
                super().batch(records,epoch)
                journal.append('training_batch',{'epoch':epoch,'microbatch':len(self.microbatch_sizes),
                    'presentations':self.exposures[-len(records):]})
            def optimizer_step(self,epoch,accumulation,ema_updates):
                consumed=list(self.pending)
                super().optimizer_step(epoch,accumulation,ema_updates)
                journal.append('optimizer_step',{'step':self.steps[-1],'EMA_updates':ema_updates,'consumed_presentations':consumed})
            def epoch_end(self,epoch):
                super().epoch_end(epoch)
                journal.append('epoch_train_end',self.epoch_tails[-1])
        ledger=DurableLedger()
        state={'status':'in_progress','variant':variant,'scope':'Main detector training + clean validation only; no final-test evaluation',
               'started_utc':datetime.now(timezone.utc).isoformat(),'requested_config':protocol['training'],
               'protocol_sha256':receipt['protocol_sha256'],'overlay_sha256':file_sha256(OVERLAY),
               'exposure_manifest_sha256':receipt['exposure_manifest_sha256'],
               'dataset_lock_sha256':protocol['dataset_lock_sha256'],'initial_weights_sha256':protocol['initial_weights_sha256'],
               'epochs':[],'checkpoint_writes':[],'validation_passes':[],'memory_samples':[],
               'memory_method':'One-second psutil RSS/system available/global swap + torch.mps tensor/driver allocations. Observed samples, not guaranteed peaks; unified counters not additive; global swap not solely attributable to training.',
               'frozen_receipt_sha256':file_sha256(ROOT/'docs/main_ac_recovery_v2/freeze_receipt.json'),'pid':os.getpid(),'run_id':str(run_dir),'parent_protocol_sha256':protocol['parent_protocol_sha256']}
        trainer=None
        def persist():
            atomic_json(run_dir/'provenance.json',state)
            atomic_json(run_dir/'actual_accounting.json',ledger.budget())
            atomic_json(run_dir/'exposures.json',ledger.exposures)
        persist()
        try:
            strict_runtime();state['packages']=validate_pins();torch.set_num_threads(2)
            disk=shutil.disk_usage(ROOT);state['preflight']={'disk_free_bytes':disk.free,**memory_sample()}
            if disk.free<6_000_000_000 or state['preflight']['system_available_bytes']<536870912:
                raise RuntimeError('Require 6GB disk and512MiB available memory before optimization')
            state['environment']={k:os.environ.get(k) for k in ['PYTHONHASHSEED','PYTORCH_ENABLE_MPS_FALLBACK','YOLO_AUTOINSTALL','YOLO_CONFIG_DIR','MPLCONFIGDIR']}
            state['hardware']={'machine':os.uname().machine,'ram_bytes':__import__('psutil').virtual_memory().total,'cpu':'Apple M3'}
            state['source_fingerprints']=overlay['files'];state['library_fingerprints']={**receipt['parent_library_hashes'],**receipt['extra_library_source_hashes']}
            view=ROOT/protocol['training']['data']
            degradation=json.loads((ROOT/protocol['degradation_config']).read_text())
            adapter=OnlineAdapter(variant,rows,schedule,degradation)
            epoch_clock={};validation_clock={}
            class DurableDataset(LoggedScheduledDataset):
                def __getitem__(self,index):
                    sample=super().__getitem__(index)
                    if self.mode=='train':journal.append('sample_delivery',self.delivery_queue[-1])
                    return sample
            class MainTrainer(DetectionTrainer):
                def get_dataset(self):
                    data=super().get_dataset()
                    if data.get('test') or Path(data['train']).resolve()!=view.parent/'train/images' or Path(data['val']).resolve()!=view.parent/'val/images':
                        raise ValueError('Forbidden model input outside train/clean-validation')
                    return data

                def build_dataset(self,img_path,mode='train',batch=None):
                    if mode not in {'train','val'} or Path(img_path).resolve()!=view.parent/mode/'images':
                        raise ValueError('Forbidden dataset access')
                    return DurableDataset(view_yaml=view,mode=mode,adapter=adapter if mode=='train' else None,
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

                def get_validator(self):
                    from copy import copy
                    self.loss_names='box_loss','cls_loss','dfl_loss'
                    validator=CompleteValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
                    validator.journal=journal
                    return validator

                def save_model(self):
                    complete=getattr(self.validator,'last_completion',None)
                    if not complete or not complete['complete'] or complete['expected_images']!=45 or complete['scope']!={'kind':'epoch','epoch':self.epoch+1}:
                        raise RuntimeError('Checkpoint selection inputs not completely validated')
                    with atomic_checkpoint_targets(self,run_dir,f'epoch-{self.epoch+1:03d}'):
                        super().save_model()
                    record={'epoch':self.epoch+1,'fitness':float(self.fitness),
                        'best_fitness':float(self.best_fitness),'best_written':self.fitness==self.best_fitness,
                        'EMA_updates':self.ema.updates,'best_sha256':file_sha256(self.best),'last_sha256':file_sha256(self.last),
                        'complete_validation':True}
                    state['checkpoint_writes'].append(record);journal.append('checkpoint_write',record)

                def final_eval(self):
                    with atomic_checkpoint_targets(self,run_dir,'final-best-validation',copy_existing=True):
                        super().final_eval()

            def train_start(t):
                strict_runtime()  # Library setup enables warn_only; restore strict mode.
                state['effective_initialization_sha256']=canonical_state_hash(t.model)
                if variant=='C':
                    a=json.loads((run_dir.parent/'A/provenance.json').read_text())
                    exit_record=json.loads((run_dir.parent/'A.supervision/exit.json').read_text())
                    if exit_record['returncode']!=0:raise RuntimeError('A process has not exited successfully')
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
                journal.append('effective_initialization',{'sha256':state['effective_initialization_sha256'],'device':state['actual_device']})
                persist()

            def epoch_start(t):
                if ledger.epoch_tails and t.epoch+1!=len(ledger.epoch_tails)+1:
                    raise RuntimeError('Epoch replay/skip forbidden')
                if t.train_loader.dataset.delivery_queue:
                    raise RuntimeError('Cached/prefetched exposures crossed epoch boundary')
                t.train_loader.dataset.set_epoch(t.epoch+1);epoch_clock['start']=time.monotonic()
                journal.append('epoch_start',{'epoch':t.epoch+1})

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
                epoch_commit(run_dir,f'epoch-{t.epoch+1:03d}',state,ledger.budget(),ledger.exposures,journal)

            def val_start(v):validation_clock['start']=time.monotonic()
            def val_end(v):
                if not v.last_completion or not v.last_completion['complete']:raise RuntimeError('Unverified validation metrics')
                state['validation_passes'].append({'seconds':time.monotonic()-validation_clock['start'],'scope':'clean validation only','completion':v.last_completion})
            def sample_loop():
                while not stop.is_set():
                    state['memory_samples'].append({'elapsed':time.monotonic()-started,**memory_sample()})
                    atomic_json(run_dir/'heartbeat.json',{'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),'elapsed':time.monotonic()-started,'epochs_committed':len(state['epochs'])})
                    stop.wait(1)
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
            epoch_commit(run_dir,'final',state,ledger.budget(),ledger.exposures,journal)
            journal.append('training_finished',{'epochs':len(state['epochs']),'updates':len(ledger.steps)})
            return state
        except BaseException as error:
            record_failure(state,error);journal.append('run_failed',{'type':type(error).__name__,'error':str(error)});persist();raise
        finally:
            stop.set()
            if sampler:sampler.join(timeout=3)
            state['finished_utc']=datetime.now(timezone.utc).isoformat();state['total_elapsed_seconds']=time.monotonic()-started
            if state['memory_samples']:
                state['observed_memory_maxima']={k:max(r[k] for r in state['memory_samples']) for k in state['memory_samples'][0] if k.endswith('bytes')}
                state['minimum_available_bytes']=min(r['system_available_bytes'] for r in state['memory_samples'])
            state['EMA_updates']=ledger.ema_updates
            persist()
            journal.append('python_run_finally',{'status':state['status'],'elapsed':state['total_elapsed_seconds']})
            journal.close()
