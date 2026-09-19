from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "yolo26_teacher_distill_spike.py"
WORKFLOW = ROOT / ".github" / "workflows" / "yolo26-teacher-distill-spike.yml"


def _load_spike():
    assert SCRIPT.is_file(), "YOLO26 teacher-distill spike script is missing"
    spec = importlib.util.spec_from_file_location("yolo26_teacher_distill_spike", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_authority_mask_uses_only_verified_safe_masks():
    spike = _load_spike()

    class Box:
        def __init__(self, *, x1, y1, x2, y2, mask, verified_mask, safe_to_inpaint):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.mask = mask
            self.verified_mask = verified_mask
            self.safe_to_inpaint = safe_to_inpaint

    good = Box(
        x1=1, y1=1, x2=4, y2=4,
        mask=np.full((3, 3), 255, dtype=np.uint8),
        verified_mask=True,
        safe_to_inpaint=True,
    )
    unsafe = Box(
        x1=5, y1=1, x2=8, y2=4,
        mask=np.full((3, 3), 255, dtype=np.uint8),
        verified_mask=True,
        safe_to_inpaint=False,
    )
    unverified = Box(
        x1=1, y1=5, x2=4, y2=8,
        mask=np.full((3, 3), 255, dtype=np.uint8),
        verified_mask=False,
        safe_to_inpaint=True,
    )

    authority = spike.teacher_authority_mask((10, 10, 3), [good, unsafe, unverified])

    assert int(np.count_nonzero(authority)) == 9
    assert np.all(authority[1:4, 1:4] == 255)
    assert int(np.count_nonzero(authority[1:4, 5:8])) == 0
    assert int(np.count_nonzero(authority[5:8, 1:4])) == 0


def test_mask_to_yolo_polygons_round_trips_teacher_authority():
    spike = _load_spike()
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[2:8, 3:12] = 255
    mask[12:20, 18:29] = 255

    polygons = spike.mask_to_yolo_polygons(mask)
    reconstructed = spike.rasterize_yolo_polygons(polygons, mask.shape)

    metrics = spike.mask_metrics(mask, reconstructed)
    assert len(polygons) == 2
    assert metrics["pixel_recall"] >= 0.98
    assert metrics["pixel_precision"] >= 0.98
    for polygon in polygons:
        assert len(polygon) >= 6
        assert len(polygon) % 2 == 0
        assert all(0.0 <= value <= 1.0 for value in polygon)


def test_distill_split_is_deterministic_and_spreads_validation_pages():
    spike = _load_spike()

    splits = [spike.dataset_split_for_index(index, validation_stride=4) for index in range(8)]

    assert splits == ["val", "train", "train", "train", "val", "train", "train", "train"]


def test_teacher_distill_workflow_is_shadow_only_and_targets_yolo26s_768():
    assert WORKFLOW.is_file(), "YOLO26 teacher-distill workflow is missing"
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "yolo26s_manga_best.pt" in text
    assert "imgsz=768" in text
    assert "epochs=3" in text
    assert "--text-class-id 0" in text
    assert "--sizes 768" in text
    assert "teacher-dataset" in text
    assert "models/text_segmenter.onnx" not in text or "cp " not in text
    assert "benchmark-only" in text.lower() or "shadow" in text.lower()


def test_teacher_distill_workflow_verifies_one_class_segmentation_contract():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "expected raw [1,37,N]" in text
    assert "expected prototypes [1,32,H,W]" in text


def test_teacher_distill_workflow_persists_actual_ultralytics_train_save_dir():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'Path("benchmark-results/yolo26-teacher-distill").resolve()' in text
    assert "model.trainer.save_dir" in text
    assert "train-save-dir.txt" in text
    assert '.read_text(encoding="utf-8").strip()' in text
