"""W8D engineering/diagnostic tests. Subprocess reply fixtures are not model results."""
import json,sys,time,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'app'))
from p57_ui.client import call_worker


def fixture_worker(tmp_path,monkeypatch):
 (tmp_path/'app').mkdir();(tmp_path/'app/worker.py').write_text('import time,json;time.sleep(.45);print(json.dumps({"ok":True,"result":{"fixture":True},"worker_seconds_before_cleanup":.45}))')
 monkeypatch.setenv('P57_INFERENCE_PYTHON',sys.executable)
 return tmp_path


def test_optional_swap_failure_retains_other_counters(tmp_path,monkeypatch):
 import psutil
 root=fixture_worker(tmp_path,monkeypatch)
 monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1234)))
 monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=5678))
 def unavailable():raise OSError('controlled swap counter unavailable')
 monkeypatch.setattr(psutil,'swap_memory',unavailable)
 result=call_worker(root,{'op':'fixture'})
 assert result['process']['actual_exit']==0
 samples=result['process']['resource_observations'];assert samples
 assert all(s['rss_bytes']==1234 and s['system_available_bytes']==5678 and s['global_swap_bytes'] is None for s in samples)
 assert all(s['unavailable']['global_swap_bytes']['type']=='OSError' for s in samples)


def test_interrupted_communication_reaps_owned_child(tmp_path,monkeypatch):
 import p57_ui.client as client
 root=fixture_worker(tmp_path,monkeypatch);original=client.subprocess.Popen;children=[]
 def launch(*args,**kwargs):
  child=original(*args,**kwargs);children.append(child);communicate=child.communicate;calls=0
  def interrupted(*a,**kw):
   nonlocal calls
   calls+=1
   if calls==1:raise KeyboardInterrupt('controlled communication interruption')
   return communicate(*a,**kw)
  child.communicate=interrupted;return child
 monkeypatch.setattr(client.subprocess,'Popen',launch)
 try:
  with pytest.raises(KeyboardInterrupt):call_worker(root,{'op':'fixture'})
  assert children and children[0].poll() is not None
 finally:
  for child in children:
   if child.poll() is None:child.kill()
   child.wait()


def row(c,box,score=.9):return {'class_id':c,'class_name':__import__('p57_ui.core',fromlist=['CLASSES']).CLASSES[c],'confidence':score,'xyxy':box}


def test_matching_wrong_class_duplicates_and_fixed_threshold():
 from p57_ui.diagnostics import match_image,aggregate
 gt=[{'class_id':0,'xyxy':[0,0,10,10]}]
 matched=match_image([row(0,[0,0,10,10]),row(0,[0,0,10,10],.8),row(1,[0,0,10,10],.7),row(0,[0,0,10,10],.249)],gt)
 assert (matched['TP'],matched['FP'],matched['FN'])==(1,2,0)
 wrong=match_image([row(1,[0,0,10,10])],gt)
 assert (wrong['TP'],wrong['FP'],wrong['FN'])==(0,1,1)
 summary=aggregate([matched,wrong]);assert (summary['TP'],summary['FP'],summary['FN'],summary['GT'])==(1,3,1,2)
 assert summary['precision']==.25 and summary['recall']==.5
 assert summary['per_class'][5]['precision'] is None and summary['per_class'][5]['recall'] is None


def test_matching_iou_boundary_ties_and_non_drawable():
 from p57_ui.diagnostics import match_image
 gt=[{'class_id':0,'xyxy':[0,0,10,10]},{'class_id':0,'xyxy':[0,0,10,10]}]
 a=match_image([row(0,[0,0,5,10],.25),row(0,[0,0,5,10],.25),row(0,[0,0,10,0],.99)],gt)
 assert (a['TP'],a['FP'],a['FN'])==(2,0,0)
 assert [(r['prediction_index'],r['ground_truth_index']) for r in a['decisions']]==[(0,0),(1,1)]


def test_actual_worker_predictor_color_padding_and_model_separation(monkeypatch):
 import base64,torch,ultralytics
 import worker
 from PIL import Image
 from p57_ui.core import CLASSES,png
 captured=[]
 class FakeYOLO:
  def __init__(self,path,task):self.model_key='A' if '/A/' in str(path) else 'C';self.names=dict(enumerate(CLASSES))
  def predict(self,source,predictor,**kwargs):
   assert source.shape==(32,64,3) and source[0,0].tolist()==[0,0,255] # actual RGB->BGR handoff
   p=object.__new__(predictor);p.device=torch.device('cpu');p.model=SimpleNamespace(fp16=False)
   tensor=p.preprocess([source]) # actual AppPredictor + pinned BasePredictor preprocessing
   assert list(tensor.shape)==[1,3,448,448]
   torch.testing.assert_close(tensor[0,:,224,224],torch.tensor([1.,0.,0.]),rtol=0,atol=0)
   torch.testing.assert_close(tensor[0,:,0,0],torch.full((3,),114/255),rtol=0,atol=1e-7)
   p.last_completion={'complete':True};self.predictor=p
   c=0 if self.model_key=='A' else 1;captured.append(self.model_key)
   boxes=SimpleNamespace(xyxy=torch.tensor([[0.,0.,64.,32.]]),cls=torch.tensor([float(c)]),conf=torch.tensor([.8]))
   return [SimpleNamespace(boxes=boxes)]
 monkeypatch.setattr(ultralytics,'YOLO',FakeYOLO)
 result=worker.run({'op':'infer','image_png':base64.b64encode(png(Image.new('RGB',(64,32),'red'))).decode(),'models':['A','C'],'device':'cpu','condition':'original','regions':[]})
 assert captured==['A','C'] and result['models']['A']['raw_predictions'][0]['class_id']==0 and result['models']['C']['raw_predictions'][0]['class_id']==1
 assert result['models']['A']['shared_pixel_sha256']==result['models']['C']['shared_pixel_sha256']


def test_filter_and_repeat_exports_do_not_invoke_worker(monkeypatch):
 from p57_ui.core import exports,png,displayed
 from PIL import Image
 import p57_ui.client as client
 def forbidden(*args,**kwargs):raise AssertionError('Export/filter attempted inference')
 monkeypatch.setattr(client,'call_worker',forbidden)
 raw=[row(0,[0,0,10,10],.8),row(1,[0,0,10,10],.2)]
 image=png(Image.new('RGB',(10,10)))
 result={'models':{'A':{'name':'A','raw_predictions':raw}},'device':'cpu'}
 a=exports(image,image,result,.25,{})[1];b=exports(image,image,result,.25,{})[1]
 assert a==b and len(displayed(raw,.25))==1 and len(displayed(raw,.1))==2 and len(raw)==2
