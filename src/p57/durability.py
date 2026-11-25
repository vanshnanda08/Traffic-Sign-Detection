"""Atomic snapshots and append-only, fsynced actual-event evidence."""
import contextlib
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from datetime import datetime,timezone


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path);fsync_dir(path.parent)


class Journal:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('x');self.sequence=0;self.previous='0'*64
    def append(self,event,data):
        value={'sequence':self.sequence+1,'previous_sha256':self.previous,'event':event,
               'utc':datetime.now(timezone.utc).isoformat(),'monotonic':time.monotonic(),'data':data}
        encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        digest=hashlib.sha256(encoded).hexdigest();value['sha256']=digest
        self.stream.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        self.stream.flush();os.fsync(self.stream.fileno());self.sequence+=1;self.previous=digest
        return self.sequence
    def close(self):self.stream.close()


def read_journal(path):
    lines=Path(path).read_bytes().splitlines(keepends=True);events=[];previous='0'*64;tail=None
    for index,line in enumerate(lines):
        if not line.endswith(b'\n'):
            if index!=len(lines)-1:raise ValueError('Nonfinal truncated journal line')
            tail={'bytes':len(line),'sha256':hashlib.sha256(line).hexdigest(),'raw_utf8':line.decode('utf8',errors='replace')};break
        value=json.loads(line);digest=value.pop('sha256')
        actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if actual!=digest or value['previous_sha256']!=previous or value['sequence']!=len(events)+1:
            raise ValueError('Journal integrity/sequence mismatch')
        value['sha256']=digest;events.append(value);previous=digest
    return {'events':events,'truncated_tail':tail,'last_sequence':len(events),'last_sha256':previous}


@contextlib.contextmanager
def atomic_checkpoint_targets(trainer,root,label,copy_existing=False):
    root=Path(root);stage=root/'checkpoint_staging'/label;stage.mkdir(parents=True,exist_ok=False)
    old=(trainer.wdir,trainer.last,trainer.best)
    if copy_existing:
        for path in old[1:]:
            if path.exists():shutil.copyfile(path,stage/path.name)
    trainer.wdir=stage;trainer.last=stage/'last.pt';trainer.best=stage/'best.pt'
    try:
        yield
        for name,target in [('last.pt',old[1]),('best.pt',old[2])]:
            source=stage/name
            if source.exists():
                with source.open('rb') as stream:os.fsync(stream.fileno())
                os.replace(source,target)
        fsync_dir(old[0])
    finally:
        trainer.wdir,trainer.last,trainer.best=old


def epoch_commit(root,label,state,budget,exposures,journal):
    root=Path(root);dest=root/'commits'/label;dest.mkdir(parents=True,exist_ok=False)
    for name,value in [('provenance.json',state),('accounting.json',budget),('exposures.json',exposures)]:atomic_json(dest/name,value)
    objects=root/'checkpoint_objects';objects.mkdir(exist_ok=True)
    checkpoints={}
    for path in sorted((root/'fit/weights').glob('*.pt')):
        digest=sha(path);obj=objects/(digest+'.pt')
        if not obj.exists():
            temp=obj.with_suffix('.tmp');shutil.copyfile(path,temp)
            with temp.open('rb') as stream:os.fsync(stream.fileno())
            os.replace(temp,obj);fsync_dir(objects)
        if sha(obj)!=digest:raise ValueError('Checkpoint object changed')
        checkpoints[path.name]={'sha256':digest,'object':str(obj.relative_to(root))}
    manifest={'label':label,'snapshots':{p.name:sha(p) for p in dest.glob('*.json')},'checkpoints':checkpoints,
              'journal_sequence_before_commit':journal.sequence,'journal_sha256_before_commit':journal.previous}
    atomic_json(dest/'manifest.json',manifest)
    journal.append('epoch_commit',{'label':label,'manifest_sha256':sha(dest/'manifest.json')})
    atomic_json(root/'latest_commit.json',{'label':label,'manifest_sha256':sha(dest/'manifest.json'),'journal_sequence':journal.sequence})
    return manifest
