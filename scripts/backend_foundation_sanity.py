# Deterministic acceptance gate for backend foundation Phases 1-2.
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
def check(c,m):
    if not c: raise AssertionError(m)
def mb(r): return json.dumps({"schema_version":3,"chapter_id":"a1b2c3d4","pages":[{"clean_revision":r}]},separators=(",",":")).encode()
def publication_safety_checks():
    import app.manifest_utils as mu
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); src=r/'n.tmp'; dst=r/'live'; src.write_bytes(b'NEW'); dst.write_bytes(b'OLD'); calls=0
        def locked(a,b):
            nonlocal calls; calls+=1; raise PermissionError('locked')
        with mock.patch.object(mu.os,'replace',side_effect=locked):
            try: mu.atomic_replace(src,dst,max_retries=3,delay=0)
            except PermissionError: pass
            else: raise AssertionError('rename unexpectedly succeeded')
        check(calls==3,'retry count'); check(dst.read_bytes()==b'OLD','old destination changed'); check(src.read_bytes()==b'NEW','temp changed')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); live=r/'clean.png'; live.write_bytes(b'OLD'); (r/'manifest.json').write_bytes(mb(0)); tx=mu.PageArtifactTransaction(r,0,[live],1); tx.__enter__(); backup=r/str(tx.records[0]['backup']); live.write_bytes(b'NEW'); check(backup.read_bytes()==b'OLD','rollback shares inode'); check(tx.rollback(),'rollback failed'); check(live.read_bytes()==b'OLD','rollback bytes wrong')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); live=r/'clean.png'; live.write_bytes(b'OLD'); (r/'manifest.json').write_bytes(mb(0)); tx=mu.PageArtifactTransaction(r,0,[live],1); tx.__enter__(); live.write_bytes(b'NEW'); check(mu.recover_page_artifact_transactions(r)==1,'recovery count'); check(live.read_bytes()==b'OLD','precommit crash not restored')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); live=r/'clean.png'; live.write_bytes(b'OLD'); (r/'manifest.json').write_bytes(mb(0)); tx=mu.PageArtifactTransaction(r,0,[live],1); tx.__enter__(); n=r/'n.tmp'; n.write_bytes(b'NEW'); mu.atomic_replace(n,live); committed=json.loads((r/'manifest.json').read_text()); committed_page=committed['pages'][0]; committed_page['clean_revision']=1; tx.mark_manifest_commit(committed_page); (r/'manifest.json').write_text(json.dumps(committed),encoding='utf-8'); check(mu.recover_page_artifact_transactions(r)==1,'commit recovery count'); check(live.read_bytes()==b'NEW','committed artifact rolled back')
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); src=r/'render.tmp'; dst=r/'page.png'; src.write_bytes(b'NEW'); dst.write_bytes(b'OLD')
        try: mu.publish_then_commit(src,dst,lambda: (_ for _ in ()).throw(RuntimeError('manifest')))
        except RuntimeError: pass
        else: raise AssertionError('metadata failure succeeded')
        check(dst.read_bytes()==b'OLD','render rollback failed')
    p=(ROOT/'app/pipeline.py').read_text(encoding='utf-8'); rr=(ROOT/'app/routers/render_commit.py').read_text(encoding='utf-8'); ex=(ROOT/'app/routers/export.py').read_text(encoding='utf-8')
    check('atomic_replace(tmp_clean_path, final_clean_path)' in p,'clean publish'); check('atomic_replace(tmp_auto_clean_path, auto_clean_path)' in p,'auto-clean publish'); check('publish_then_commit(tmp_path, final_path, commit_manifest)' in rr,'render publish'); check('atomic_replace(tmp_archive, final_archive)' in ex,'export publish')
def windows_locked_reader_check():
    if os.name!='nt': return
    import ctypes; from ctypes import wintypes; import app.manifest_utils as mu
    k=ctypes.WinDLL('kernel32',use_last_error=True); cf=k.CreateFileW; cf.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]; cf.restype=wintypes.HANDLE; ch=k.CloseHandle
    with tempfile.TemporaryDirectory() as t:
        r=Path(t); src=r/'n.tmp'; dst=r/'live'; src.write_bytes(b'NEW'); dst.write_bytes(b'OLD'); h=cf(str(dst),0x80000000,1,None,3,0,None)
        try:
            try: mu.atomic_replace(src,dst,max_retries=2,delay=0)
            except PermissionError: pass
            else: raise AssertionError('locked Windows rename succeeded')
            check(dst.read_bytes()==b'OLD','Windows old bytes changed'); check(src.read_bytes()==b'NEW','Windows temp changed')
        finally: ch(h)


def model_contract_checks():
    import numpy as np
    from app.model_contracts import (
        decode_lama_output,
        validate_detector_session,
        validate_lama_session,
    )

    class Meta:
        def __init__(self, name, shape, type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = type

    class Session:
        def __init__(self, inputs, outputs):
            self._inputs = inputs
            self._outputs = outputs
        def get_inputs(self):
            return self._inputs
        def get_outputs(self):
            return self._outputs

    bubble = Session(
        [Meta("images", [1, 3, 1024, 1024])],
        [Meta("output0", [1, 6, 21504])],
    )
    bubble_contract = validate_detector_session(
        bubble,
        role="bubble_detector",
        configured_input_size=1024,
    )
    check(
        bubble_contract.class_names == ("text_bubble", "text_free")
        and not bubble_contract.destructive_text_mask,
        "bubble role authority",
    )

    text = Session(
        [Meta("images", [1, 3, 1024, 1024])],
        [
            Meta("output0", [1, 37, 21504]),
            Meta("output1", [1, 32, 256, 256]),
        ],
    )
    text_contract = validate_detector_session(
        text,
        role="text_segmenter",
        configured_input_size=1024,
    )
    check(
        text_contract.provides_prototypes and text_contract.destructive_text_mask,
        "text role authority",
    )
    try:
        validate_detector_session(
            text,
            role="text_segmenter",
            configured_input_size=960,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible detector resolution accepted")

    dynamic = Session(
        [
            Meta("mask", ["batch", 1, "h", "w"]),
            Meta("image", ["batch", 3, "h", "w"]),
        ],
        [Meta("inpainted", ["batch", 3, "h", "w"])],
    )
    dynamic_contract = validate_lama_session(dynamic, dynamic=True, fixed_size=512)
    check(
        dynamic_contract.image_input_name == "image"
        and dynamic_contract.mask_input_name == "mask"
        and dynamic_contract.output_range == "zero_to_one",
        "dynamic names/range",
    )
    check(
        int(
            decode_lama_output(
                np.full((1, 3, 2, 2), 0.5, np.float32),
                dynamic_contract,
            )[0, 0, 0]
        )
        == 127,
        "dynamic scale contract",
    )

    fixed = Session(
        [
            Meta("mask", [1, 1, 512, 512]),
            Meta("image", [1, 3, 512, 512]),
        ],
        [Meta("output", [1, 3, 512, 512])],
    )
    fixed_contract = validate_lama_session(fixed, dynamic=False, fixed_size=512)
    check(
        fixed_contract.output_range == "zero_to_255"
        and fixed_contract.output_name == "output",
        "fixed output contract",
    )
    check(
        int(
            decode_lama_output(
                np.full((1, 3, 2, 2), 0.5, np.float32),
                fixed_contract,
            )[0, 0, 0]
        )
        == 0,
        "fixed dark output was incorrectly rescaled",
    )
    try:
        decode_lama_output(
            np.full((1, 3, 2, 2), 1.5, np.float32),
            dynamic_contract,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range dynamic output accepted")


def geometry_contract_checks():
    import math
    import types
    import numpy as np

    if "onnxruntime" not in sys.modules:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
    from app.detector.bubble_detector import (
        LetterboxTransform,
        MaskDecodeGeometry,
        YoloDetector,
    )

    def reference(src_w, src_h, box, offset_x=0, offset_y=0):
        nominal = min(1024 / src_w, 1024 / src_h)
        resized_w = max(1, min(1024, int(src_w * nominal)))
        resized_h = max(1, min(1024, int(src_h * nominal)))
        pad_x = (1024 - resized_w) // 2
        pad_y = (1024 - resized_h) // 2
        scale_x = resized_w / src_w
        scale_y = resized_h / src_h
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(src_w, (x1 - pad_x) / scale_x))
        y1 = max(0.0, min(src_h, (y1 - pad_y) / scale_y))
        x2 = max(0.0, min(src_w, (x2 - pad_x) / scale_x))
        y2 = max(0.0, min(src_h, (y2 - pad_y) / scale_y))
        ix1 = max(0, min(src_w, math.floor(x1)))
        iy1 = max(0, min(src_h, math.floor(y1)))
        ix2 = max(ix1, min(src_w, math.ceil(x2)))
        iy2 = max(iy1, min(src_h, math.ceil(y2)))
        return (
            ix1 + offset_x,
            iy1 + offset_y,
            ix2 + offset_x,
            iy2 + offset_y,
        )

    cases = [
        (800, 1338, (-50, 200, 300, 600), 0, 0),
        (517, 333, (0, -40, 1024, 600), 0, 0),
        (333, 517, (900, 900, 1100, 1200), 0, 0),
        (401, 277, (-20, -30, 1080, 1060), 37, 91),
    ]
    for src_w, src_h, box, ox, oy in cases:
        transform = LetterboxTransform.create(
            src_w,
            src_h,
            1024,
            1024,
            offset_x=ox,
            offset_y=oy,
        )
        check(
            transform.page_box_from_canvas(box)
            == reference(src_w, src_h, box, ox, oy),
            f"geometry mismatch {(src_w, src_h, box)}",
        )

    odd = LetterboxTransform.create(333, 517, 1024, 1024)
    check(
        abs(odd.scale_x - odd.scale_y) > 1e-6,
        "odd-size transform lost actual x/y resize scales",
    )

    tile = LetterboxTransform.create(
        320,
        240,
        1024,
        1024,
        offset_x=100,
        offset_y=200,
    )
    full_content = (
        tile.pad_x,
        tile.pad_y,
        tile.pad_x + tile.resized_w,
        tile.pad_y + tile.resized_h,
    )
    check(
        tile.page_box_from_canvas(full_content) == (100, 200, 420, 440),
        "tile source ownership offset",
    )

    # Independent clipped-border mask reproduction: the final source box is
    # [0,50,80,100), so prototype crop must be derived from that clipped box,
    # not from the original detector rectangle that extended into padding.
    transform = LetterboxTransform.create(400, 200, 1024, 1024)
    source_box = (0, 50, 80, 100)
    geometry = MaskDecodeGeometry(transform, source_box)
    prototypes = np.full((1, 256, 256), -10.0, np.float32)
    prototypes[0, 96:128, 0:26] = 10.0
    mask = YoloDetector._decode_mask(
        None,
        np.array([1.0], np.float32),
        prototypes,
        geometry,
        80,
        50,
    )
    xs = np.where(mask > 0)[1]
    check(
        xs.size > 0 and int(xs.min()) == 0 and 37 <= int(xs.max()) <= 42,
        f"clipped mask shifted: {xs.min() if xs.size else None}.."
        f"{xs.max() if xs.size else None}",
    )


def evidence_retention_checks():
    import types
    import numpy as np

    if "onnxruntime" not in sys.modules:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

    from app.detector.bubble_detector import BubbleBox, YoloDetector
    from app.detector.combined_detector import CombinedTextDetector
    from app.detector.mask_builder import build_mask
    from app.inpaint.lama_inpainter import Inpainter

    giant = BubbleBox(
        0, 100, 980, 120, 0.9,
        np.full((20, 980), 255, np.uint8),
        source_model="segmenter.onnx",
        class_name="text_comic",
        semantic_type="text",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
        ocr_eligible=True,
        source_role="text_segmenter",
    )
    retained = YoloDetector._filter_invalid([giant], 1000, 1000)
    check(len(retained) == 1, "giant SFX evidence disappeared")
    check(
        retained[0].deferred_reason
        and "box_width_limit" in retained[0].deferred_reason
        and retained[0].needs_review
        and not retained[0].safe_to_inpaint,
        "giant SFX was not explicitly deferred",
    )
    review_mask = build_mask((1000, 1000), retained)
    check(
        not np.any(review_mask > 0),
        "review-only region gained automatic destructive authority",
    )

    first = BubbleBox(
        200, 50, 800, 350, 0.9,
        np.full((300, 600), 255, np.uint8),
        source_model="segmenter.onnx", class_name="text_comic",
        semantic_type="text", mask_source="text_segmenter",
        safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
    )
    second = BubbleBox(
        200, 350, 800, 650, 0.8,
        np.full((300, 600), 255, np.uint8),
        source_model="segmenter.onnx", class_name="text_comic",
        semantic_type="text", mask_source="text_segmenter",
        safe_to_inpaint=True, ocr_eligible=True,
        source_role="text_segmenter",
    )

    detector = object.__new__(CombinedTextDetector)
    grouped = detector._cluster_free_text_boxes([first, second], 1000, 1000)
    check(len(grouped) == 2, "oversized free-text group dropped evidence")
    check(
        all(box.mask is not None and np.any(box.mask > 0) for box in grouped),
        "oversized free-text split lost a child mask",
    )

    split = Inpainter._split_oversized_cluster_area(
        [first, second], 1000, 1000
    )
    flat = [box for group in split for box in group]
    check(len(split) == 2 and len(flat) == 2, "two-box inpaint split lost evidence")
    combined_mask = build_mask((1000, 1000), flat)
    check(
        combined_mask[100, 300] > 0 and combined_mask[500, 300] > 0,
        "two-box reproduction did not retain both masks",
    )

    pipeline_source = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check(
        'target_page["deferred_regions"]' in pipeline_source
        and '"deferred_reason": b.deferred_reason' in pipeline_source
        and 'and not record.get("deferred_reason")' in pipeline_source,
        "saved/deferred decision contract missing",
    )


evidence_retention_checks()
model_contract_checks()
geometry_contract_checks()
publication_safety_checks()
windows_locked_reader_check()
print("backend foundation sanity: phases 1-3 PASS")
