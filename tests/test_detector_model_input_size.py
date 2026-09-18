from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np

from app.detector.bubble_detector import YoloDetector


def test_yolo_preprocess_uses_the_loaded_model_contract_size():
    detector = object.__new__(YoloDetector)
    detector.contract = SimpleNamespace(input_width=640, input_height=640)
    image = np.zeros((120, 200, 3), dtype=np.uint8)

    blob, transform = detector._preprocess(image)

    assert blob.shape == (1, 3, 640, 640)
    assert transform.input_w == 640
    assert transform.input_h == 640


def test_secondary_segmenter_specialization_plan_matches_yolo_head_geometry():
    module_name = "app.detector.segmenter_specializer"
    assert find_spec(module_name) is not None, "secondary segmenter specializer is missing"
    module = import_module(module_name)

    plan = module.specialization_plan(640)

    assert plan.candidate_count == 8400
    assert plan.prototype_side == 160
    assert plan.reshape_dfl == (1, 4, 16, 8400)
    assert plan.reshape_boxes == (1, 4, 8400)
    assert plan.stride_counts == {8: 6400, 16: 1600, 32: 400}
