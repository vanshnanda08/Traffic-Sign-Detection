"""W6 development-only tests. Synthetic tensors/temporary files, no test inference."""
import copy,json,subprocess
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import yaml
from ultralytics.utils import nms
from p57.dataset import ROOT
from p57.evaluation_data import AuthInputs,AuthDataset,render_condition,pixels_sha,evaluation_yaml,validate_eval_yaml,generate_conditions,source_rows,CONDITIONS
from p57.evaluation_engine import recompute
from p57.evaluation_protocol import authenticate_models
from p57.evaluation_artifacts import display_matching,paired_table
from p57.complete_validation import complete_nms,Coverage,PINNED_NMS
from p57.feasibility import validate_detections


def test_explicit_test_gate_before_any_real_test_generation():
    manifest=ROOT/'data/processed/p57_heldout_conditions_v1/manifest.json'
    before=manifest.read_bytes() if manifest.exists() else None
    with pytest.raises(ValueError,match='Explicit'):AuthInputs('heldout')
    with pytest.raises(ValueError,match='Explicit'):generate_conditions()
    assert (manifest.read_bytes() if manifest.exists() else None)==before
    with pytest.raises(ValueError):source_rows('train')


def test_cli_without_test_flag_never_runs_evaluation():
    result=subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/evaluate_heldout.py'),'--model','A','--condition','clean'],cwd=ROOT,env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
    assert result.returncode==2 and 'execute-heldout' in result.stderr


def test_absolute_null_train_val_yaml_and_no_redirection(tmp_path):
    names={0:'a',1:'b'};root=tmp_path/'view';root.mkdir();path=root/'dataset.yaml'
    data=evaluation_yaml(root,names);path.write_text(yaml.safe_dump(data));validate_eval_yaml(path,root,names)
    assert data['train'] is None and data['val'] is None and Path(data['test']).is_absolute()
    for key,value in [('val','core/val/images'),('download','unsafe'),('path','elsewhere'),('nc',3)]:
        changed={**data,key:value};path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):validate_eval_yaml(path,root,names)


def test_development_loader_isolation_and_missing_membership():
    from ultralytics.cfg import get_cfg
    auth=AuthInputs('development');assert auth.split=='val' and len(auth.expected)==45
    bad=copy.copy(auth);bad.expected=auth.expected[:-1]
    with pytest.raises(ValueError,match='membership'):AuthDataset(bad,imgsz=416,batch_size=4,hyp=get_cfg(),rect=True)


def test_changed_selected_checkpoint_hash_rejected(monkeypatch):
    import p57.evaluation_protocol as ep
    cfg=copy.deepcopy(ep.config());cfg['checkpoints']['A']['sha256']='0'*64
    monkeypatch.setattr(ep,'config',lambda:cfg)
    with pytest.raises(ValueError,match='checkpoint drift'):authenticate_models()


@pytest.mark.parametrize('condition',CONDITIONS)
def test_multibox_deterministic_transforms_preserve_geometry_and_png(tmp_path,condition):
    image=np.arange(80*100*3,dtype=np.uint8).reshape(80,100,3);before=image.copy();boxes=[[0,.4,.4,.5,.5],[1,.6,.6,.5,.5]];original=copy.deepcopy(boxes)
    a,detail=render_condition(image,boxes,'synthetic-development-fixture',condition);b,_=render_condition(image,list(reversed(boxes)),'synthetic-development-fixture',condition)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(image,before);assert boxes==original
    p=tmp_path/'view.png';cv2.imwrite(str(p),a,[cv2.IMWRITE_PNG_COMPRESSION,3]);assert pixels_sha(cv2.imread(str(p)))==pixels_sha(a)
    if condition=='clean':np.testing.assert_array_equal(a,image)


def perfect_record():
    return {'classes':[0.],'confidence':[np.float32(.9).item()],'ground_truth':{'classes':[0.]},'matching':{'iou_matrix':[[1.]],'iou_thresholds':torch.linspace(.5,.95,10).tolist(),'stats':{'tp':[[True]*10],'conf':[np.float32(.9).item()],'pred_cls':[0.],'target_cls':[0.]},'dtypes':{'tp':'bool','conf':'float32','pred_cls':'float32','target_cls':'float32'}}}


def test_pinned_metric_extraction_serialization_and_matching():
    row=json.loads(json.dumps(perfect_record()));metrics=recompute([row],{0:'fixture'},1e-10)
    assert metrics['precision']==1 and metrics['recall']==1 and metrics['mAP50']==pytest.approx(.995)
    row['matching']['stats']['tp'][0][0]=False
    with pytest.raises(ValueError,match='matching'):recompute([row],{0:'fixture'},1e-10)


def test_legitimate_empty_predictions_and_incomplete_coverage():
    c=Coverage(['a','b']);c.nms_completed(['a'],[torch.zeros((0,6))]);c.metrics_completed(['a'])
    with pytest.raises(ValueError):c.finish()
    with pytest.raises(ValueError):c.nms_completed(['a'],[torch.zeros((0,6))])
    with pytest.raises(ValueError):c.nms_completed(['foreign'],[torch.zeros((0,6))])
    c.nms_completed(['b'],[torch.zeros((0,6))]);c.metrics_completed(['b']);assert c.finish()['complete']


def test_wrong_class_overlap_is_explanatory_FP_and_FN():
    row=perfect_record();row['classes']=[1.]
    match=display_matching(row);assert match['FP']==[0] and match['FN']==[0] and not match['TP']
    row=perfect_record();row['confidence']=[.1];assert display_matching(row)['FN']==[0]


def test_invalid_class_and_metric_pair_table_units():
    with pytest.raises(ValueError):validate_detections(np.array([[0,0,5,5]]),np.array([6]),np.array([.9]),(10,10,3),6)
    runs={(m,c):{'aggregate':{k:(.6 if m=='A' else .5) for k in ['mAP50','precision','recall','mAP50_95']}} for m in ['A','C'] for c in CONDITIONS}
    rows=paired_table(runs);assert len(rows)==7 and rows[0]['delta_mAP50_fraction']==pytest.approx(-.1)
    assert 100*rows[0]['delta_mAP50_fraction']==pytest.approx(-10.)
    del runs[('A','F1')]
    with pytest.raises(KeyError):paired_table(runs)


def test_delayed_clock_complete_NMS_and_ordinary_equivalence(monkeypatch):
    prediction=torch.zeros((3,5,8));prediction[:,:2,0]=20;prediction[:,2:4,0]=10;prediction[:,4,0]=.9
    ordinary=PINNED_NMS(prediction.clone(),conf_thres=.1);complete=complete_nms(prediction.clone(),conf_thres=.1)
    for x,y in zip(ordinary,complete):torch.testing.assert_close(x,y,rtol=0,atol=0)
    class Clock:
        def __init__(self):self.n=-1
        def time(self):self.n+=1;return self.n*10.
    monkeypatch.setattr(nms,'time',Clock());partial=PINNED_NMS(prediction.clone(),conf_thres=.1)
    monkeypatch.setattr(nms,'time',Clock());full=complete_nms(prediction.clone(),conf_thres=.1)
    assert [len(x) for x in partial]==[1,0,0] and [len(x) for x in full]==[1,1,1]


def test_preflight_rect_padding_matches_existing_path():
    from ultralytics.cfg import get_cfg
    from p57.online import ScheduledYOLODataset
    auth=AuthInputs('development')
    new=AuthDataset(auth,imgsz=416,batch_size=4,hyp=get_cfg(),rect=True,stride=32)
    old=ScheduledYOLODataset(view_yaml=auth.yaml,mode='val',imgsz=416,batch_size=4,hyp=get_cfg(),rect=True,stride=32)
    np.testing.assert_array_equal(new.batch_shapes,old.batch_shapes)
    np.testing.assert_array_equal(new[0]['img'].numpy(),old[0]['img'].numpy())


def test_yaml_ancestor_link_rejected(tmp_path):
    names={0:'fixture'};real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    (real/'dataset.yaml').write_text(yaml.safe_dump(evaluation_yaml(alias,names)))
    with pytest.raises(ValueError,match='external links'):validate_eval_yaml(alias/'dataset.yaml',alias,names)
