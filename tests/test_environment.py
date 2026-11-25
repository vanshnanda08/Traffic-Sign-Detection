import random

import cv2
import numpy as np
import pytest
import torch
from torchvision.ops import nms

from p57.environment import load_config, seed_everything, select_device, validate_pins


def test_all_installed_versions_match_lock():
    assert {"torch", "torchvision", "ultralytics", "opencv-python", "pytest"} <= validate_pins().keys()


def test_torchvision_compiled_nms_is_compatible():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
    kept = nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
    assert kept.tolist() == [0]


def test_fixed_seed_repeats_cpu_generators():
    def sample():
        image = np.zeros((4, 4), dtype=np.float32)
        cv2.randu(image, 0, 1)
        return random.random(), np.random.rand(4), torch.rand(4), image
    seed_everything(57)
    first = sample()
    seed_everything(57)
    second = sample()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    np.testing.assert_array_equal(first[3], second[3])


def test_cpu_device_and_unknown_device():
    assert select_device("cpu") == "cpu"
    with pytest.raises(ValueError):
        select_device("cuda")


def test_unavailable_mps_fails_explicitly(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device() == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"):
        select_device("mps")


def test_auto_selects_available_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device() == "mps"


def test_actual_selected_device_forward_and_backward():
    device = select_device(load_config()["device"])
    x = torch.tensor([2., 3.], device=device, requires_grad=True)
    (x.square().sum()).backward()
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([4., 6.]))
