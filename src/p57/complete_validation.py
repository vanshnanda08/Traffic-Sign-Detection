"""Recovery-v2: complete pinned NMS, with no global/installed-code patches."""
import json
from pathlib import Path
from collections import Counter

import torch
from ultralytics.models.yolo.detect import DetectionValidator, DetectionPredictor
from ultralytics.utils import nms, ops


PINNED_NMS = nms.non_max_suppression


def complete_nms(prediction, *args, **kwargs):
    """Only the wall-clock break changes; ordinary NMS math/limits stay pinned."""
    if 'max_time_img' in kwargs:
        raise ValueError('Time policy is fixed by complete-validation adapter')
    raw=prediction[0] if isinstance(prediction,(tuple,list)) else prediction
    batch_size=int(raw.shape[0])
    result=PINNED_NMS(prediction,*args,max_time_img=float('inf'),**kwargs)
    outputs=result[0] if kwargs.get('return_idxs',False) else result
    if len(outputs)!=batch_size:
        raise ValueError('Incomplete NMS batch output')
    return result


class Coverage:
    """A zero-detection image is valid; an unprocessed image is not."""
    def __init__(self,expected):
        self.expected=list(map(str,expected));self.batches=[];self.metric_ids=[]
        if len(set(self.expected))!=len(self.expected):raise ValueError('Duplicate expected source IDs')

    def nms_completed(self,source_ids,outputs):
        ids=list(map(str,source_ids))
        if len(ids)!=len(outputs) or len(set(ids))!=len(ids):raise ValueError('Incomplete/duplicate validation batch')
        if any(i not in self.expected for i in ids):raise ValueError('Forbidden validation input')
        prior=[i for b in self.batches for i in b['source_ids']]
        if set(prior)&set(ids):raise ValueError('Duplicate validation source')
        self.batches.append({'batch':len(self.batches)+1,'source_ids':ids,'nms_completed':True,
                             'detection_counts':[int(len(o)) for o in outputs]})

    def metrics_completed(self,source_ids):
        ids=list(map(str,source_ids))
        if not self.batches or ids!=self.batches[-1]['source_ids']:raise ValueError('Unverified metric inputs')
        self.metric_ids.extend(ids)

    def finish(self):
        ids=[i for b in self.batches for i in b['source_ids']]
        if Counter(ids)!=Counter(self.expected) or Counter(self.metric_ids)!=Counter(self.expected):
            raise ValueError('Missing/duplicate/incomplete validation processing; metrics forbidden')
        return {'complete':True,'expected_images':len(self.expected),'nms_images':len(ids),
                'metric_images':len(self.metric_ids),'batches':self.batches,
                'time_policy':'unlimited_wall_clock; pinned max_time_img argument is positive infinity',
                'empty_detections_are_valid':True}


class CompleteValidator(DetectionValidator):
    """Checks every expected source before any stats/fitness can be returned."""
    journal=None

    def __call__(self,trainer=None,model=None):
        self.last_completion=None
        self.validation_scope={'kind':'epoch' if trainer is not None else 'checkpoint',
                               'epoch':trainer.epoch+1 if trainer is not None else None}
        if self.journal:self.journal.append('validation_start',self.validation_scope)
        result=super().__call__(trainer=trainer,model=model)
        if not self.last_completion or not self.last_completion['complete']:
            raise ValueError('Validator returned unverified/incomplete metrics')
        return result

    def init_metrics(self,model):
        super().init_metrics(model)
        # Expected membership comes from the authenticated view, not a possibly
        # shortened loader. Source hashes/metadata are rechecked without test IO.
        from p57.development import validate_development_view
        view_yaml=Path(self.args.data)
        validate_development_view(view_yaml)
        lock=json.loads((view_yaml.parent/'view.lock.json').read_text())
        expected=[str((view_yaml.parent/name).resolve()) for name in lock['files'] if name.startswith('val/images/')]
        actual=[str(Path(p).resolve()) for p in self.dataloader.dataset.im_files]
        if len(expected)!=45 or Counter(actual)!=Counter(expected):
            raise ValueError('Validation loader missing/duplicating authenticated sources')
        self.coverage=Coverage(expected)

    def preprocess(self,batch):
        self.current_source_ids=[str(Path(p).resolve()) for p in batch['im_file']]
        return super().preprocess(batch)

    def postprocess(self,preds):
        outputs=complete_nms(preds,self.args.conf,self.args.iou,
            nc=0 if self.args.task=='detect' else self.nc,multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,max_det=self.args.max_det,
            end2end=self.end2end,rotated=self.args.task=='obb')
        self.coverage.nms_completed(self.current_source_ids,outputs)
        if self.journal:self.journal.append('validation_nms_batch',self.coverage.batches[-1])
        return [{'bboxes':x[:,:4],'conf':x[:,4],'cls':x[:,5],'extra':x[:,6:]} for x in outputs]

    def update_metrics(self,preds,batch):
        if len(preds)!=len(batch['im_file']):raise ValueError('Missing metric predictions')
        super().update_metrics(preds,batch)
        self.coverage.metrics_completed([str(Path(p).resolve()) for p in batch['im_file']])
        if self.journal:self.journal.append('validation_metric_batch',{'source_ids':self.current_source_ids})

    def get_stats(self):
        self.last_completion={**self.coverage.finish(),'scope':self.validation_scope}
        if self.seen!=len(self.coverage.expected):raise ValueError('Validator seen count mismatch')
        result=super().get_stats()
        if self.journal:self.journal.append('validation_complete',self.last_completion)
        return result


class CompletePredictor(DetectionPredictor):
    """Gallery NMS mirrors pinned DetectionPredictor defaults, without timeout."""
    def postprocess(self,preds,img,orig_imgs,**kwargs):
        save_feats=getattr(self,'_feats',None) is not None
        outputs=complete_nms(preds,self.args.conf,self.args.iou,self.args.classes,self.args.agnostic_nms,
            max_det=self.args.max_det,nc=0 if self.args.task=='detect' else len(self.model.names),
            end2end=getattr(self.model,'end2end',False),rotated=self.args.task=='obb',return_idxs=save_feats)
        ids=list(map(str,self.batch[0]));dets=outputs[0] if save_feats else outputs
        coverage=Coverage(ids);coverage.nms_completed(ids,dets);coverage.metrics_completed(ids)
        self.last_completion=coverage.finish()
        if not isinstance(orig_imgs,list):orig_imgs=ops.convert_torch2numpy_batch(orig_imgs)
        if save_feats:
            feats=self.get_obj_feats(self._feats,outputs[1]);outputs=outputs[0]
        results=self.construct_results(outputs,img,orig_imgs,**kwargs)
        if save_feats:
            for result,feat in zip(results,feats):result.feats=feat
        return results
