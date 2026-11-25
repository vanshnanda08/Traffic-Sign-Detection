"""Authenticated development/held-out views; test pixels require explicit gate."""
import csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics.data.dataset import YOLODataset
from p57.dataset import ROOT,parse_labels
from p57.assets import file_sha256
from p57.development import validate_development_view
from p57.online import ScheduledYOLODataset
from p57.degradation import degrade
from p57.durability import atomic_json

CONFIG=ROOT/'configs/evaluation_v1.json'
CONDITIONS=('clean','F1','F2','F3','O1','O2','O3')


def config():return json.loads(CONFIG.read_text())


def pixels_sha(image):
    header=json.dumps([list(image.shape),str(image.dtype)],separators=(',',':')).encode()
    return hashlib.sha256(header+image.tobytes()).hexdigest()


def source_rows(split):
    if split not in {'val','test'}:raise ValueError('Only explicit validation or held-out membership')
    cfg=config();core=ROOT/cfg['core'];lock_path=core/'dataset.lock.json'
    if file_sha256(lock_path)!=cfg['core_lock_sha256']:raise ValueError('Core lock drift')
    lock=json.loads(lock_path.read_text())
    with (ROOT/'results/dataset_v2/manifest.csv').open() as stream:rows=[r for r in csv.DictReader(stream) if r['split']==split]
    names={v['id']:v['name'] for v in lock['class_mapping'].values()}
    if len(rows)!=45 or len({r['source_id'] for r in rows})!=45:raise ValueError('Expected45 unique frozen sources')
    expected={n:s for n,s in lock['files'].items() if n.startswith(f'{split}/')}
    actual={}
    for row in rows:
        for key in ['image_path','label_path']:
            name=row[key]
            if not name.startswith(split+'/') or name not in expected:raise ValueError('Foreign split membership')
            actual[name]=expected[name]
    if actual!=expected:raise ValueError('Source membership differs from core lock')
    return sorted(rows,key=lambda r:r['source_id']),names,lock


def render_condition(image,boxes,source_id,condition):
    if condition=='clean':return image.copy(),{'condition':'clean'}
    if condition not in CONDITIONS:raise ValueError('Unknown condition')
    labels=np.asarray(boxes,dtype=np.float32).reshape(-1,5).tolist()
    output,_,record=degrade(image,labels,source_id,condition,json.loads((ROOT/config()['degradation_config']).read_text()))
    return output,record


def evaluation_yaml(root,names):
    return {'train':None,'val':None,'test':str((Path(root)/'test/images').resolve()),'nc':len(names),'names':names}


def validate_eval_yaml(path,root,names):
    path=Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Absolute local evaluation YAML with no external links required')
    data=yaml.safe_load(path.read_text())
    if data!=evaluation_yaml(root,names):raise ValueError('YAML redirection/download/split/class metadata rejected')
    return data


class AuthInputs:
    def __init__(self,mode,condition='clean',allow_heldout=False):
        if mode=='development':
            rows,names,lock=source_rows('val');self.view=ROOT/'data/processed/p57_development_v1';self.split='val'
            self.yaml=self.view/'dataset.yaml';self.metadata=validate_development_view(self.yaml)
            self.files={n:s for n,s in lock['files'].items() if n.startswith('val/')}
            self.rows={str((self.view/r['image_path']).resolve()):{**r,'condition':'clean'} for r in rows}
        elif mode=='heldout':
            if not allow_heldout:raise ValueError('Explicit held-out execution flag required')
            manifest=verify_conditions(condition)
            rows=[r for r in manifest['rows'] if r['condition']==condition];_,names,_=source_rows('test')
            self.view=ROOT/config()['derived_root']/condition;self.split='test';self.yaml=self.view/'dataset.yaml'
            self.metadata=validate_eval_yaml(self.yaml,self.view,names)
            self.files={r[key]:r[sha] for r in rows for key,sha in [('image_path','derived_file_sha256'),('label_path','label_sha256')]}
            self.rows={str((self.view/r['image_path']).resolve()):r for r in rows}
        else:raise ValueError('Unknown evaluation mode')
        self.mode=mode;self.condition=condition;self.names=names
        self.images=self.view/self.split/'images'
        self.expected=sorted(self.rows)
        if len(self.expected)!=45:raise ValueError('Expected45 authenticated inputs')


class AuthDataset(ScheduledYOLODataset):
    def __init__(self,auth,**kwargs):
        self.view=auth.view;self.mode=auth.split;self.auth=auth.files;self.metadata=auth.metadata
        self.adapter=None;self.epoch=0;self.last_exposure={}
        YOLODataset.__init__(self,img_path=str(auth.images),data=auth.metadata,augment=False,cache=False,task='detect',**kwargs)
        if sorted(map(lambda p:str(Path(p).resolve()),self.im_files))!=auth.expected:raise ValueError('Loader membership drift')


def generate_conditions(allow_heldout=False):
    if not allow_heldout:raise ValueError('Explicit held-out execution flag required')
    from p57.evaluation_protocol import verify_freeze
    verify_freeze();cfg=config();root=ROOT/cfg['derived_root']
    if root.exists():return verify_conditions(deep=True)
    rows,names,lock=source_rows('test');core=ROOT/cfg['core'];root.mkdir(parents=True,exist_ok=False)
    records=[]
    for condition in CONDITIONS:
        dest=root/condition;(dest/'test/images').mkdir(parents=True);(dest/'test/labels').mkdir()
        (dest/'dataset.yaml').write_text(yaml.safe_dump(evaluation_yaml(dest,names),sort_keys=False))
        for row in rows:
            src=core/row['image_path'];label=core/row['label_path']
            if src.is_symlink() or label.is_symlink() or file_sha256(src)!=lock['files'][row['image_path']] or file_sha256(label)!=lock['files'][row['label_path']]:raise ValueError('Frozen source bytes changed')
            image=cv2.imread(str(src));boxes=parse_labels(label.read_text(),len(names));out,detail=render_condition(image,boxes,row['source_id'],condition)
            image_name=f"test/images/{row['source_id']}.png";label_name=f"test/labels/{row['source_id']}.txt"
            if not cv2.imwrite(str(dest/image_name),out,[cv2.IMWRITE_PNG_COMPRESSION,3]):raise RuntimeError('PNG write failed')
            (dest/label_name).write_bytes(label.read_bytes());decoded=cv2.imread(str(dest/image_name))
            if not np.array_equal(out,decoded):raise ValueError('Lossless PNG round trip failed')
            if condition=='clean' and not np.array_equal(decoded,image):raise ValueError('Clean decoded pixels changed')
            records.append({'source_id':row['source_id'],'condition':condition,'source_image_path':row['image_path'],'source_label_path':row['label_path'],'source_file_sha256':file_sha256(src),'source_decoded_pixels_sha256':pixels_sha(image),'derived_file_sha256':file_sha256(dest/image_name),'derived_decoded_pixels_sha256':pixels_sha(decoded),'label_sha256':file_sha256(label),'image_path':image_name,'label_path':label_name,'boxes':len(boxes),'degradation':detail})
    manifest={'config_sha256':file_sha256(CONFIG),'core_lock_sha256':cfg['core_lock_sha256'],'degradation_config_sha256':file_sha256(ROOT/cfg['degradation_config']),'degradation_source_sha256':file_sha256(ROOT/cfg['degradation_implementation']),'rows':records}
    atomic_json(root/'manifest.json',manifest)
    atomic_json(ROOT/'docs/evaluation_v1/condition_generation_receipt.json',{'manifest_sha256':file_sha256(root/'manifest.json'),'config_sha256':manifest['config_sha256'],'sources':45,'image_label_pairs':315,'box_counts':{c:sum(r['boxes'] for r in records if r['condition']==c) for c in CONDITIONS}})
    return verify_conditions(deep=True)


def verify_conditions(condition=None,deep=False):
    cfg=config();root=ROOT/cfg['derived_root'];path=root/'manifest.json';manifest=json.loads(path.read_text())
    receipt=json.loads((ROOT/'docs/evaluation_v1/condition_generation_receipt.json').read_text())
    if len(manifest['rows'])!=315:raise ValueError('Expected exactly315 manifest rows')
    if manifest['core_lock_sha256']!=cfg['core_lock_sha256'] or manifest['degradation_config_sha256']!=file_sha256(ROOT/cfg['degradation_config']) or manifest['degradation_source_sha256']!=file_sha256(ROOT/cfg['degradation_implementation']):raise ValueError('Derived protocol provenance drift')
    if manifest['config_sha256']!=file_sha256(CONFIG) or receipt['manifest_sha256']!=file_sha256(path):raise ValueError('Derived manifest/config drift')
    rows,names,lock=source_rows('test');original={r['source_id']:r for r in rows};seen=set()
    for c in CONDITIONS:
        subset=[r for r in manifest['rows'] if r['condition']==c]
        if len(subset)!=45 or {r['source_id'] for r in subset}!=set(original):raise ValueError('Condition pairing mismatch')
        dest=root/c;validate_eval_yaml(dest/'dataset.yaml',dest,names)
        expected={'dataset.yaml'}|{r[k] for r in subset for k in ['image_path','label_path']}
        for p in dest.rglob('*'):
            if p.is_symlink() or p.is_file() and p.relative_to(dest).as_posix() not in expected:raise ValueError('Unexpected or external derived file')
        for r in subset:
            key=(c,r['source_id'])
            if key in seen:raise ValueError('Duplicate condition source')
            seen.add(key);o=original[r['source_id']]
            if r['image_path']!=f"test/images/{r['source_id']}.png" or r['label_path']!=f"test/labels/{r['source_id']}.txt" or r['source_image_path']!=o['image_path'] or r['source_label_path']!=o['label_path']:raise ValueError('Derived source/path pairing drift')
            if r['source_file_sha256']!=lock['files'][o['image_path']] or r['label_sha256']!=lock['files'][o['label_path']]:raise ValueError('Source/annotation provenance drift')
            if condition is None or c==condition:
                for key,sha in [('image_path','derived_file_sha256'),('label_path','label_sha256')]:
                    if file_sha256(dest/r[key])!=r[sha]:raise ValueError('Derived bytes changed')
                if len(parse_labels((dest/r['label_path']).read_text(),len(names)))!=r['boxes']:raise ValueError('Derived annotation count drift')
                if deep:
                    src=ROOT/cfg['core']/o['image_path'];image=cv2.imread(str(src));actual=cv2.imread(str(dest/r['image_path']));boxes=parse_labels((ROOT/cfg['core']/o['label_path']).read_text(),len(names));expected_image,_=render_condition(image,boxes,r['source_id'],c)
                    if not np.array_equal(actual,expected_image) or pixels_sha(actual)!=r['derived_decoded_pixels_sha256']:raise ValueError('Derived pixels differ from frozen transform')
    if len(seen)!=315:raise ValueError('Expected315 image-label pairs')
    return manifest
