"""Development-only access and controlled degradation behavior."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from p57.dataset import ROOT
from p57.development import build_development_view, validate_development_view, require_development_row
from p57.degradation import degrade
from p57.feasibility import check_trial_config, validate_detections


@pytest.fixture
def degradation_config():
    return json.loads((ROOT/"configs/degradation_dev_v1.json").read_text())


@pytest.mark.parametrize("condition",["F1","F2","F3","O1","O2","O3"])
def test_determinism_outside_regions_and_unchanged_sources(degradation_config,condition):
    image=np.arange(120*100*3,dtype=np.uint8).reshape(120,100,3)
    original=image.copy()
    boxes=[[0,.4,.4,.4,.4],[0,.6,.6,.4,.4]]
    before=[b[:] for b in boxes]
    a,mask,record=degrade(image,boxes,"fixture",condition,degradation_config)
    b,mask2,record2=degrade(image,list(reversed(boxes)),"fixture",condition,degradation_config)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(mask,mask2)
    np.testing.assert_array_equal(a[~mask],image[~mask])
    np.testing.assert_array_equal(image,original)
    assert boxes==before and record==record2 and a.dtype==np.uint8


def test_severity_is_nested_and_fading_increases_distance(degradation_config):
    image=np.full((100,120,3),20,np.uint8)
    boxes=[[0,.5,.5,.7,.7]]
    faded=[degrade(image,boxes,"fixture",c,degradation_config)[0] for c in ("F1","F2","F3")]
    distances=[np.abs(f.astype(float)-image).sum() for f in faded]
    assert distances[0]<distances[1]<distances[2]
    masks=[degrade(image,boxes,"fixture",c,degradation_config)[1] for c in ("O1","O2","O3")]
    assert masks[0].sum()<masks[1].sum()<masks[2].sum()
    assert not np.any(masks[0]&~masks[1]) and not np.any(masks[1]&~masks[2])


def test_overlapping_fading_regions_processed_once(degradation_config):
    image=np.zeros((100,100,3),np.uint8)
    boxes=[[0,.4,.4,.6,.6],[1,.6,.6,.6,.6]]
    faded,_,_=degrade(image,boxes,"fixture","F2",degradation_config)
    assert faded[50,50,0]==int(np.rint(np.float32(.35)*np.float32(170)))
    assert faded[25,25,0]==faded[50,50,0]


def test_final_test_row_rejected():
    with pytest.raises(ValueError,match="Final-test"):
        require_development_row({"split":"test","image_path":"test/images/a.jpg"})


def test_development_view_excludes_test_and_keeps_exact_cardinalities():
    path=build_development_view()
    data=validate_development_view(path)
    assert "test" not in data and not (path.parent/"test").exists()
    assert len(list((path.parent/"train/images").glob("*")))==210
    assert len(list((path.parent/"val/images").glob("*")))==45


def test_dataset_download_or_test_override_rejected(tmp_path):
    path=tmp_path/"dataset.yaml"
    path.write_text(yaml.safe_dump({"train":"train/images","val":"val/images","nc":6,"names":{},"test":"test/images"}))
    with pytest.raises(ValueError,match="test/download/path"):
        validate_development_view(path)


def test_changed_class_metadata_rejected(tmp_path):
    path=tmp_path/'dataset.yaml'
    data=yaml.safe_load(build_development_view().read_text())
    data['names'][0]='WRONG_CLASS'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match='Class metadata'):
        validate_development_view(path)


@pytest.mark.parametrize("changes",[{"epochs":50},{"device":"cpu"},{"deterministic":False},{"amp":True},{"mosaic":1.0}])
def test_trial_scope_is_bounded(changes):
    cfg=json.loads((ROOT/"configs/feasibility_v1.json").read_text())
    cfg["training"].update(changes)
    with pytest.raises(ValueError):check_trial_config(cfg)


@pytest.mark.parametrize("box,cls,confidence",[([0,0,float('nan'),1],0,.5),([0,0,50,50],6,.5),([5,5,1,1],0,.5),([0,0,50,50],0,1.1)])
def test_invalid_inference_outputs_fail(box,cls,confidence):
    with pytest.raises(ValueError):
        validate_detections(np.array([box]),np.array([cls]),np.array([confidence]),(100,100,3),6)


def test_valid_and_empty_detections():
    validate_detections(np.array([[0,0,50,50]]),np.array([1]),np.array([.4]),(100,100,3),6)
    validate_detections(np.empty((0,4)),np.array([]),np.array([]),(100,100,3),6)
