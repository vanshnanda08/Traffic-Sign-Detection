"""Exact evaluator-stream capture and pinned metric recomputation; no predict API."""
import json,time,math
from pathlib import Path
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.build import build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.utils import ops,USER_CONFIG_DIR
from ultralytics.utils.metrics import box_iou,ap_per_class,smooth
from ultralytics.engine.validator import BaseValidator
from p57.dataset import ROOT
from p57.assets import file_sha256
from p57.complete_validation import CompleteValidator,Coverage
from p57.evaluation_data import config,AuthDataset
from p57.durability import atomic_json
from p57.feasibility import validate_detections
from p57.environment import validate_pins


def require_local_font():
    from matplotlib import font_manager
    candidates=[USER_CONFIG_DIR/'Arial.ttf']+[Path(s) for s in font_manager.findSystemFonts() if 'Arial.ttf' in s]
    for path in candidates:
        if path.is_file():return {'path':str(path),'sha256':file_sha256(path)}
    raise RuntimeError('Local Arial font missing; no network downloads allowed')


def aggregate_stats(stats,names,detailed=False):
    arrays={key:np.concatenate(values,axis=0) for key,values in stats.items()}
    result=ap_per_class(arrays['tp'],arrays['conf'],arrays['pred_cls'],arrays['target_cls'],plot=False,names=names)
    p,r,ap,ids=result[2],result[3],result[5],result[6]
    aggregate={'precision':float(np.mean(p)),'recall':float(np.mean(r)),'mAP50':float(ap[:,0].mean()),'mAP50_95':float(ap.mean())}
    per_class=[{'id':int(c),'precision':float(p[i]),'recall':float(r[i]),'mAP50':float(ap[i,0]),'mAP50_95':float(ap[i].mean())} for i,c in enumerate(ids)]
    return {'aggregate':aggregate,'per_class':per_class} if detailed else aggregate


def recompute(records,names,tolerance,detailed=False):
    stats={key:[] for key in ['tp','conf','pred_cls','target_cls']}
    matcher=object.__new__(BaseValidator)
    for row in records:
        match=row['matching'];matcher.iouv=torch.tensor(match['iou_thresholds'],dtype=torch.float32)
        pred=torch.tensor(row['classes'],dtype=torch.float32);target=torch.tensor(row['ground_truth']['classes'],dtype=torch.float32)
        matrix=torch.tensor(match['iou_matrix'],dtype=torch.float32).reshape(len(target),len(pred))
        actual=BaseValidator.match_predictions(matcher,pred,target,matrix).numpy()
        expected=np.asarray(match['stats']['tp'],dtype=bool).reshape(len(pred),10)
        if not np.array_equal(actual,expected):raise ValueError('Stored matching information disagrees with pinned matcher')
        for key in stats:
            shape=(-1,10) if key=='tp' else (-1,)
            stats[key].append(np.asarray(match['stats'][key],dtype=match['dtypes'][key]).reshape(shape))
    return aggregate_stats(stats,names,detailed)


class RecordingValidator(CompleteValidator):
    def __init__(self,auth,output,*args,**kwargs):
        self.auth=auth;self.output=Path(output);self.records=[];self.metric_result=None
        super().__init__(*args,**kwargs)

    def init_metrics(self,model):
        DetectionValidator.init_metrics(self,model)
        if self.args.split!=self.auth.split or Path(self.args.data)!=self.auth.yaml:
            raise ValueError('Wrong evaluator split/YAML')
        if self.data.get(self.auth.split)!=str(self.auth.images.resolve()):raise ValueError('Resolved dataset path redirection')
        actual=sorted(str(Path(p).resolve()) for p in self.dataloader.dataset.im_files)
        if actual!=self.auth.expected:raise ValueError('Missing/duplicate/foreign evaluator input')
        if self.data['names']!=self.auth.names or self.data['nc']!=6:raise ValueError('Class metadata drift')
        self.coverage=Coverage(self.auth.expected)

    def update_metrics(self,preds,batch):
        before={k:len(v) for k,v in self.metrics.stats.items()}
        super().update_metrics(preds,batch)
        for i,pred in enumerate(preds):
            pb=self._prepare_batch(i,batch);p=self._prepare_pred(pred)
            native=self.scale_preds(p,pb)
            gt_native=ops.scale_boxes(pb['imgsz'],pb['bboxes'].clone(),pb['ori_shape'],ratio_pad=pb['ratio_pad'])
            stats={key:self.metrics.stats[key][before[key]+i] for key in ['tp','conf','pred_cls','target_cls']}
            matrix=box_iou(pb['bboxes'],p['bboxes']).cpu().numpy()
            path=str(Path(batch['im_file'][i]).resolve());source=self.auth.rows[path]
            boxes=native['bboxes'].cpu().numpy();classes=p['cls'].cpu().numpy();scores=p['conf'].cpu().numpy()
            validate_detections(boxes,classes,scores,(*pb['ori_shape'],3),6)
            self.records.append({'source_id':source['source_id'],'condition':self.auth.condition,'image_path':path,'image_sha256':file_sha256(path),
                'shape':list(pb['ori_shape']),'input_shape':list(pb['imgsz']),'xyxy':boxes.tolist(),'network_xyxy':p['bboxes'].cpu().tolist(),'classes':classes.tolist(),'confidence':scores.tolist(),
                'ground_truth':{'classes':pb['cls'].cpu().tolist(),'xyxy':gt_native.cpu().tolist(),'network_xyxy':pb['bboxes'].cpu().tolist()},
                'matching':{'space':'letterboxed metric/input coordinates','iou_thresholds':self.iouv.cpu().tolist(),'iou_matrix':matrix.tolist(),
                    'stats':{k:v.tolist() for k,v in stats.items()},'dtypes':{k:str(v.dtype) for k,v in stats.items()}}})
        atomic_json(self.output/'predictions.partial.json',{'scope':self.auth.mode,'records':self.records,'complete':False})

    def get_stats(self):
        result=super().get_stats()
        b=self.metrics.box
        idx=int(smooth(b.f1_curve.mean(0),.1).argmax())
        self.metric_result={'aggregate':{'precision':float(result['metrics/precision(B)']),'recall':float(result['metrics/recall(B)']),'mAP50':float(result['metrics/mAP50(B)']),'mAP50_95':float(result['metrics/mAP50-95(B)'])},
          'operating_point':{'convention':'common smoothed mean-class best-F1 at IoU0.5; not fixed0.25','confidence':float(b.px[idx]),'grid_index':idx},
          'per_class':[{'id':int(c),'name':self.auth.names[int(c)],'precision':float(b.p[i]),'recall':float(b.r[i]),'mAP50':float(b.all_ap[i,0]),'mAP50_95':float(b.all_ap[i].mean()),'ground_truth_boxes':int(self.metrics.nt_per_class[c]),'ground_truth_images':int(self.metrics.nt_per_image[c])} for i,c in enumerate(b.ap_class_index)]}
        return result


def execute_evaluator(auth,checkpoint,output):
    require_local_font();versions=validate_pins();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    settings={**config()['settings'],'data':str(auth.yaml),'split':auth.split}
    hyp=get_cfg(overrides=settings)
    dataset=AuthDataset(auth,imgsz=416,batch_size=4,hyp=hyp,rect=True,stride=32)
    loader=build_dataloader(dataset,batch=4,workers=0,shuffle=False,rank=-1)
    validator=RecordingValidator(auth,output,loader,save_dir=output,args=settings)
    started=time.monotonic();validator(model=str(checkpoint))
    if not str(validator.device).startswith('mps') or validator.args.half:raise ValueError('Actual evaluator device/precision drift')
    if len(validator.records)!=45 or len({r['source_id'] for r in validator.records})!=45:raise ValueError('Prediction record coverage incomplete')
    predictions={'scope':auth.mode,'condition':auth.condition,'checkpoint_sha256':file_sha256(checkpoint),'config_sha256':file_sha256(ROOT/'configs/evaluation_v1.json'),'records':validator.records,'complete':True}
    atomic_json(output/'predictions.json',predictions)
    decoded=json.loads((output/'predictions.json').read_text())
    detail=recompute(decoded['records'],auth.names,config()['tolerances']['serialized_metric_absolute'],detailed=True)
    recomputed=detail['aggregate']
    for observed,original in zip(detail['per_class'],validator.metric_result['per_class']):
        if observed['id']!=original['id'] or any(abs(observed[k]-original[k])>config()['tolerances']['serialized_metric_absolute'] for k in ['precision','recall','mAP50','mAP50_95']):raise ValueError('Per-class serialized recomputation mismatch')
    for key,value in recomputed.items():
        if abs(value-validator.metric_result['aggregate'][key])>config()['tolerances']['serialized_metric_absolute']:
            raise ValueError('Serialized prediction metric recomputation failed')
    result={**validator.metric_result,'scope':auth.mode,'condition':auth.condition,'images':45,'boxes':sum(len(r['ground_truth']['classes']) for r in validator.records),
      'completion':validator.last_completion,'checkpoint_sha256':file_sha256(checkpoint),'predictions_sha256':file_sha256(output/'predictions.json'),
      'config_sha256':file_sha256(ROOT/'configs/evaluation_v1.json'),'recomputed':recomputed,'recomputed_per_class':detail['per_class'],'runtime_seconds':time.monotonic()-started,
      'settings':settings,'actual_device':str(validator.device),'half':validator.args.half,'software_versions':versions,'source_order':[r['source_id'] for r in validator.records]}
    if not all(math.isfinite(v) for v in result['aggregate'].values()):raise ValueError('Nonfinite metrics')
    atomic_json(output/'metrics.json',result)
    return result
