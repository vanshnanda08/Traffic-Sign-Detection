"""Train-only frozen online adapter and real Ultralytics dataset integration.

The load/resize convention follows Ultralytics8.3.228 BaseDataset.load_image
(AGPL-3.0); source decoding is uncached so conditions cannot leak across epochs.
Generic augmentation is replaced with deterministic letterboxing/formatting for
both A and C, avoiding optional Albumentations defaults.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, LetterBox, Format

from p57.assets import file_sha256
from p57.dataset import parse_labels
from p57.degradation import degrade
from p57.development import require_development_row, validate_development_view


class OnlineAdapter:
    def __init__(self, variant, rows, schedule, degradation):
        if variant not in {'A','C'}:
            raise ValueError('Only A and C are defined')
        self.variant=variant
        self.rows={r['source_id']:r for r in rows}
        if len(self.rows)!=len(rows) or any(r['split']!='train' for r in rows):
            raise ValueError('Adapter accepts unique train originals only')
        self.schedule={(int(r['epoch']),r['source_id']):r['condition'] for r in schedule}
        if len(self.schedule)!=len(schedule):
            raise ValueError('Duplicate exposure entries')
        self.degradation=degradation

    def apply(self, image, boxes, row, epoch):
        require_development_row(row)
        if row['split']!='train' or self.rows.get(row['source_id'])!=row:
            raise ValueError('Online augmentation requires authenticated training membership')
        key=(epoch,row['source_id'])
        if key not in self.schedule:
            raise ValueError('Missing frozen epoch/source exposure')
        condition='clean' if self.variant=='A' else self.schedule[key]
        if condition=='clean':
            return image.copy(), {'condition':'clean'}
        result,_,record=degrade(image,boxes,row['source_id'],condition,self.degradation)
        return result,record


class ScheduledYOLODataset(YOLODataset):
    """Actual Dataset.__getitem__ path, without optimization or final-test inputs."""
    def __init__(self, *, view_yaml, mode, adapter=None, **kwargs):
        if mode not in {'train','val'}:
            raise ValueError('Final-test dataset access forbidden')
        self.view=Path(view_yaml).absolute().parent
        self.metadata=validate_development_view(view_yaml)
        self.mode=mode;self.adapter=adapter;self.epoch=1;self.last_exposure={}
        if mode=='train' and adapter is None:
            raise ValueError('Training requires declared A/C adapter')
        self.auth=json.loads((self.view/'view.lock.json').read_text())['files']
        super().__init__(img_path=str(self.view/mode/'images'),data=self.metadata,
                         augment=(mode=='train'),cache=False,task='detect',**kwargs)

    def set_epoch(self, epoch):
        if epoch not in range(1,51):
            raise ValueError('Epoch outside frozen budget')
        self.epoch=epoch

    def get_labels(self):
        # Do not consume Ultralytics labels.cache, even when historically present.
        labels=[]
        for name in self.im_files:
            path=Path(name); relative=path.relative_to(self.view).as_posix()
            label=self.view/self.mode/'labels'/(path.stem+'.txt')
            for p in (path,label):
                if p.is_symlink() or file_sha256(p)!=self.auth[p.relative_to(self.view).as_posix()]:
                    raise ValueError('Unauthenticated online input')
            image=cv2.imread(str(path)); boxes=np.asarray(parse_labels(label.read_text(),self.metadata['nc']),dtype=np.float32).reshape(-1,5)
            labels.append({'im_file':str(path),'shape':image.shape[:2], 'cls':boxes[:,:1],
                           'bboxes':boxes[:,1:],'segments':[],'keypoints':None,
                           'normalized':True,'bbox_format':'xywh'})
        return labels

    def build_transforms(self, hyp=None):
        return Compose([LetterBox(new_shape=(self.imgsz,self.imgsz),scaleup=self.mode=='train'),
                        Format(bbox_format='xywh',normalize=True,batch_idx=True,bgr=0.)])

    def load_image(self, i, rect_mode=True):
        path=Path(self.im_files[i])
        if path.is_symlink() or file_sha256(path)!=self.auth[path.relative_to(self.view).as_posix()]:
            raise ValueError('Online image changed')
        image=cv2.imread(str(path))
        if image is None:
            raise ValueError('Unreadable online image')
        if self.mode=='train':
            row=next((r for r in self.adapter.rows.values() if r['image_path']==path.relative_to(self.view).as_posix()),None)
            if row is None:
                raise ValueError('Unknown training image')
            lab=self.labels[i]
            boxes=np.concatenate([lab['cls'],lab['bboxes']],axis=1).tolist()
            image,record=self.adapter.apply(image,boxes,row,self.epoch)
            self.last_exposure={'epoch':self.epoch,'source_id':row['source_id'],**record}
        h0,w0=image.shape[:2]
        if rect_mode:
            ratio=self.imgsz/max(h0,w0)
            if ratio!=1:
                image=cv2.resize(image,(min(math.ceil(w0*ratio),self.imgsz),min(math.ceil(h0*ratio),self.imgsz)),interpolation=cv2.INTER_LINEAR)
        elif not h0==w0==self.imgsz:
            image=cv2.resize(image,(self.imgsz,self.imgsz),interpolation=cv2.INTER_LINEAR)
        return image,(h0,w0),image.shape[:2]
