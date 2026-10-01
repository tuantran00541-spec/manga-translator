import pytest

import app.detector.ctd_mask as ctd_mask
import app.detector.kiuyha_detector as kiuyha
import app.parameters as parameters
from app.knobs import KNOBS, knob


def test_every_knob_starts_inside_its_range_and_names_its_step():
    for entry in KNOBS.values():
        assert entry.low <= entry.default <= entry.high, entry.name
        assert entry.low < entry.high, entry.name
        assert entry.stage in {"detect", "mask", "fill"}, entry.name
        assert entry.doc, entry.name


def test_the_pipeline_reads_its_numbers_from_the_knobs():
    pairs = {
        (kiuyha, "BOX_PAD"): "detect.box_pad",
        (kiuyha, "UNION_SHARE"): "detect.union_share",
        (kiuyha, "BLOCK_GAP"): "detect.block_gap",
        (ctd_mask, "REACH"): "mask.reach",
        (ctd_mask, "EDGE"): "mask.edge",
        (ctd_mask, "THRESHOLD"): "mask.ink_threshold",
        (parameters, "INPAINT_CROP_PADDING"): "fill.crop_padding",
        (parameters, "DYNAMIC_LAMA_MAX_SINGLE_CROP_DIM"): "fill.lama_side",
        (parameters, "SMART_FILL_RING_TOLERANCE"): "fill.ring_tolerance",
    }
    for (module, constant), name in pairs.items():
        assert getattr(module, constant) == knob(name), constant
    assert parameters.DYNAMIC_LAMA_MAX_SINGLE_CROP_PIXELS == knob("fill.lama_side") ** 2


def test_an_unknown_knob_is_an_error():
    with pytest.raises(KeyError):
        knob("mask.unknown")
