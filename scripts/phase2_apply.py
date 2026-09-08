from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one source match, got {count}")
    return text.replace(old, new, 1)


MODEL_CONTRACTS = r'''from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOAT_TENSOR = "tensor(float)"


def _shape(meta) -> tuple:
    return tuple(meta.shape)


def _is_dynamic(value) -> bool:
    return value is None or isinstance(value, str)


def _check_float(meta, label: str) -> None:
    actual = getattr(meta, "type", None)
    if actual != FLOAT_TENSOR:
        raise ValueError(f"{label} must be {FLOAT_TENSOR}; got {actual!r}")


@dataclass(frozen=True)
class DetectorModelContract:
    role: str
    input_name: str
    input_height: int
    input_width: int
    class_names: tuple[str, ...]
    provides_prototypes: bool
    destructive_text_mask: bool


def validate_detector_session(
    session,
    *,
    role: str,
    configured_input_size: int,
) -> DetectorModelContract:
    specs = {
        "bubble_detector": (("text_bubble", "text_free"), False, False),
        "text_segmenter": (("text_comic",), True, True),
    }
    if role not in specs:
        raise ValueError(f"Unknown detector model role: {role!r}")

    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 1:
        raise ValueError(f"{role} requires exactly one input; got {len(inputs)}")

    inp = inputs[0]
    if inp.name != "images":
        raise ValueError(f"{role} input must be named 'images'; got {inp.name!r}")
    _check_float(inp, f"{role} input")
    shape = _shape(inp)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"{role} input must be NCHW [1,3,H,W]; got {shape}")
    if not all(isinstance(v, int) and v > 0 for v in shape[2:4]):
        raise ValueError(f"{role} detector spatial shape must be static; got {shape}")

    height, width = int(shape[2]), int(shape[3])
    if height != width or height != int(configured_input_size):
        raise ValueError(
            f"{role} model is {width}x{height}, incompatible with "
            f"DETECTOR_INPUT_SIZE={configured_input_size}"
        )

    output_by_name = {item.name: item for item in outputs}
    output0 = output_by_name.get("output0")
    if output0 is None:
        raise ValueError(f"{role} is missing output0")
    _check_float(output0, f"{role} output0")
    detection_shape = _shape(output0)
    if len(detection_shape) != 3 or detection_shape[0] != 1:
        raise ValueError(
            f"{role} output0 must be rank-3 with batch 1; got {detection_shape}"
        )

    class_names, needs_proto, destructive = specs[role]
    expected_features = 4 + len(class_names) + (32 if needs_proto else 0)
    if detection_shape[1] != expected_features:
        raise ValueError(
            f"{role} output0 feature count must be {expected_features}; "
            f"got {detection_shape}"
        )

    if needs_proto:
        if set(output_by_name) != {"output0", "output1"}:
            raise ValueError(
                "text_segmenter outputs must be exactly output0/output1; "
                f"got {sorted(output_by_name)}"
            )
        proto = output_by_name["output1"]
        _check_float(proto, "text_segmenter output1")
        proto_shape = _shape(proto)
        if (
            len(proto_shape) != 4
            or proto_shape[0] != 1
            or proto_shape[1] != 32
            or not all(isinstance(v, int) and v > 0 for v in proto_shape[2:4])
        ):
            raise ValueError(
                "text_segmenter prototype shape must be static [1,32,H,W]; "
                f"got {proto_shape}"
            )
    elif set(output_by_name) != {"output0"}:
        raise ValueError(
            "bubble_detector contract is detection-only and must not expose "
            "implicit mask authority"
        )

    return DetectorModelContract(
        role=role,
        input_name=inp.name,
        input_height=height,
        input_width=width,
        class_names=class_names,
        provides_prototypes=needs_proto,
        destructive_text_mask=destructive,
    )


@dataclass(frozen=True)
class LamaModelContract:
    dynamic: bool
    image_input_name: str
    mask_input_name: str
    output_name: str
    output_range: str
    mask_value_for_inpaint: float = 1.0


def _check_lama_tensor(meta, *, name: str, channels: int) -> tuple:
    if meta.name != name:
        raise ValueError(f"LaMa tensor must be named {name!r}; got {meta.name!r}")
    _check_float(meta, f"LaMa {name}")
    shape = _shape(meta)
    if len(shape) != 4 or shape[1] != channels:
        raise ValueError(
            f"LaMa {name} must be NCHW with {channels} channels; got {shape}"
        )
    if not (_is_dynamic(shape[0]) or shape[0] == 1):
        raise ValueError(f"LaMa {name} batch must be symbolic or 1; got {shape[0]!r}")
    return shape


def validate_lama_session(
    session,
    *,
    dynamic: bool,
    fixed_size: int = 512,
) -> LamaModelContract:
    inputs = {item.name: item for item in session.get_inputs()}
    if set(inputs) != {"image", "mask"}:
        raise ValueError(f"LaMa inputs must be exactly image/mask; got {sorted(inputs)}")

    image_shape = _check_lama_tensor(inputs["image"], name="image", channels=3)
    mask_shape = _check_lama_tensor(inputs["mask"], name="mask", channels=1)

    if dynamic:
        if not all(_is_dynamic(v) for v in image_shape[2:4]):
            raise ValueError(f"Dynamic LaMa image spatial axes must be symbolic: {image_shape}")
        if not all(_is_dynamic(v) for v in mask_shape[2:4]):
            raise ValueError(f"Dynamic LaMa mask spatial axes must be symbolic: {mask_shape}")
        output_name = "inpainted"
        output_range = "zero_to_one"
    else:
        expected = (int(fixed_size), int(fixed_size))
        if tuple(image_shape[2:4]) != expected or tuple(mask_shape[2:4]) != expected:
            raise ValueError(
                f"Fixed LaMa requires {fixed_size}x{fixed_size} inputs; "
                f"got image={image_shape}, mask={mask_shape}"
            )
        output_name = "output"
        output_range = "zero_to_255"

    outputs = {item.name: item for item in session.get_outputs()}
    if set(outputs) != {output_name}:
        raise ValueError(
            f"LaMa output must be exactly {output_name!r}; got {sorted(outputs)}"
        )
    out = outputs[output_name]
    _check_float(out, f"LaMa {output_name}")
    out_shape = _shape(out)
    if len(out_shape) != 4 or out_shape[1] != 3:
        raise ValueError(f"LaMa output must be NCHW RGB; got {out_shape}")
    if not (_is_dynamic(out_shape[0]) or out_shape[0] == 1):
        raise ValueError(f"LaMa output batch must be symbolic or 1; got {out_shape}")
    if dynamic:
        if not all(_is_dynamic(v) for v in out_shape[2:4]):
            raise ValueError(f"Dynamic LaMa output spatial axes must be symbolic: {out_shape}")
    else:
        if tuple(out_shape[2:4]) != (int(fixed_size), int(fixed_size)):
            raise ValueError(
                f"Fixed LaMa output must be {fixed_size}x{fixed_size}; got {out_shape}"
            )

    return LamaModelContract(
        dynamic=bool(dynamic),
        image_input_name="image",
        mask_input_name="mask",
        output_name=output_name,
        output_range=output_range,
    )


def decode_lama_output(output, contract: LamaModelContract) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] != 3:
        raise ValueError(f"LaMa runtime output must have shape [1,3,H,W]; got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("LaMa runtime output contains non-finite values")

    lo, hi = float(arr.min()), float(arr.max())
    eps = 1e-3
    if contract.output_range == "zero_to_one":
        if lo < -eps or hi > 1.0 + eps:
            raise ValueError(f"Normalized LaMa output out of range: [{lo}, {hi}]")
        arr = arr * 255.0
    elif contract.output_range == "zero_to_255":
        if lo < -eps or hi > 255.0 + eps:
            raise ValueError(f"Byte-range LaMa output out of range: [{lo}, {hi}]")
    else:
        raise ValueError(f"Unknown LaMa output range contract: {contract.output_range}")

    return np.clip(arr[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
'''


LETTERBOX_CLASSES = r'''@dataclass(frozen=True)
class LetterboxTransform:
    """Exact source-to-model transform for half-open pixel boxes [x1,y1,x2,y2)."""

    src_w: int
    src_h: int
    input_w: int
    input_h: int
    resized_w: int
    resized_h: int
    pad_x: int
    pad_y: int
    scale_x: float
    scale_y: float
    offset_x: int = 0
    offset_y: int = 0

    @classmethod
    def create(
        cls,
        src_w: int,
        src_h: int,
        input_w: int,
        input_h: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        if min(src_w, src_h, input_w, input_h) <= 0:
            raise ValueError("Letterbox dimensions must be positive")
        nominal = min(input_w / src_w, input_h / src_h)
        resized_w = max(1, min(input_w, int(src_w * nominal)))
        resized_h = max(1, min(input_h, int(src_h * nominal)))
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        return cls(
            src_w=src_w,
            src_h=src_h,
            input_w=input_w,
            input_h=input_h,
            resized_w=resized_w,
            resized_h=resized_h,
            pad_x=pad_x,
            pad_y=pad_y,
            scale_x=resized_w / src_w,
            scale_y=resized_h / src_h,
            offset_x=int(offset_x),
            offset_y=int(offset_y),
        )

    def page_box_from_canvas(self, canvas_box) -> tuple[int, int, int, int]:
        import math

        x1, y1, x2, y2 = (float(v) for v in canvas_box)
        local_x1 = max(0.0, min(float(self.src_w), (x1 - self.pad_x) / self.scale_x))
        local_y1 = max(0.0, min(float(self.src_h), (y1 - self.pad_y) / self.scale_y))
        local_x2 = max(0.0, min(float(self.src_w), (x2 - self.pad_x) / self.scale_x))
        local_y2 = max(0.0, min(float(self.src_h), (y2 - self.pad_y) / self.scale_y))
        ix1 = max(0, min(self.src_w, int(math.floor(local_x1))))
        iy1 = max(0, min(self.src_h, int(math.floor(local_y1))))
        ix2 = max(ix1, min(self.src_w, int(math.ceil(local_x2))))
        iy2 = max(iy1, min(self.src_h, int(math.ceil(local_y2))))
        return (
            ix1 + self.offset_x,
            iy1 + self.offset_y,
            ix2 + self.offset_x,
            iy2 + self.offset_y,
        )

    def canvas_box_from_page(self, page_box) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in page_box)
        x1 -= self.offset_x
        x2 -= self.offset_x
        y1 -= self.offset_y
        y2 -= self.offset_y
        return (
            x1 * self.scale_x + self.pad_x,
            y1 * self.scale_y + self.pad_y,
            x2 * self.scale_x + self.pad_x,
            y2 * self.scale_y + self.pad_y,
        )


@dataclass(frozen=True)
class MaskDecodeGeometry:
    transform: LetterboxTransform
    source_box: tuple[int, int, int, int]


'''


DETECTOR_GEOMETRY_BLOCK = r'''    def _preprocess(
        self,
        image: np.ndarray,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None, None
        transform = LetterboxTransform.create(
            w,
            h,
            INPUT_SIZE,
            INPUT_SIZE,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        img_rgb = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3 and image.shape[2] == 3
            else image
        )
        resized = cv2.resize(img_rgb, (transform.resized_w, transform.resized_h))
        canvas = np.full(
            (transform.input_h, transform.input_w, 3),
            DETECTOR_LETTERBOX_VALUE,
            dtype=np.uint8,
        )
        y1, y2 = transform.pad_y, transform.pad_y + transform.resized_h
        x1, x2 = transform.pad_x, transform.pad_x + transform.resized_w
        canvas[y1:y2, x1:x2] = resized
        blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return blob, transform

    def _postprocess(
        self,
        outputs,
        transform: LetterboxTransform,
    ) -> list[BubbleBox]:
        out_arr = np.squeeze(outputs[0])
        if out_arr.ndim == 1:
            out_arr = out_arr[np.newaxis, :]
        if out_arr.ndim == 2 and out_arr.shape[0] < out_arr.shape[1]:
            out_arr = out_arr.T
        if out_arr.ndim != 2 or out_arr.shape[0] == 0:
            return []

        has_proto = len(outputs) > 1 and outputs[1].ndim == 4
        if has_proto:
            num_mask_coeffs = outputs[1].shape[1]
            num_classes = max(1, out_arr.shape[1] - 4 - num_mask_coeffs)
            prototypes = outputs[1][0]
        else:
            num_mask_coeffs = 0
            num_classes = max(1, out_arr.shape[1] - 4)
            prototypes = None

        class_end = 4 + num_classes
        class_scores = out_arr[:, 4:class_end].astype(np.float32, copy=False)
        if num_classes == 1:
            class_ids = np.zeros(out_arr.shape[0], dtype=np.int32)
            confidences = class_scores[:, 0]
        else:
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32, copy=False)
            confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep = np.flatnonzero(confidences >= self.conf_threshold)
        if keep.size == 0:
            return []

        selected = out_arr[keep]
        conf_selected = confidences[keep]
        class_selected = class_ids[keep]
        coeff_start = 4 + num_classes
        coeff_end = coeff_start + num_mask_coeffs
        candidates = []
        for j in range(selected.shape[0]):
            cx, cy, bw, bh = (float(v) for v in selected[j, :4])
            raw_canvas_box = (
                cx - bw / 2.0,
                cy - bh / 2.0,
                cx + bw / 2.0,
                cy + bh / 2.0,
            )
            source_box = transform.page_box_from_canvas(raw_canvas_box)
            x1, y1, x2, y2 = source_box
            if (
                (x2 - x1) < DETECTOR_MIN_BOX_SIDE
                or (y2 - y1) < DETECTOR_MIN_BOX_SIDE
            ):
                continue
            geometry = None
            mask_coeffs = None
            if has_proto and num_mask_coeffs > 0:
                geometry = MaskDecodeGeometry(transform, source_box)
                mask_coeffs = selected[j, coeff_start:coeff_end].copy()
            candidates.append(
                (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(conf_selected[j]),
                    int(class_selected[j]),
                    num_classes,
                    geometry,
                    mask_coeffs,
                )
            )
        return self._nms(candidates, prototypes)

    def _decode_mask(
        self,
        mask_coeffs,
        prototypes,
        geometry,
        box_w: int,
        box_h: int,
    ) -> np.ndarray | None:
        if mask_coeffs is None or prototypes is None or box_w < 1 or box_h < 1:
            return None

        num_proto, mh, mw = prototypes.shape
        logits = np.clip(
            mask_coeffs @ prototypes.reshape(num_proto, -1),
            -88.0,
            88.0,
        )
        probability_map = 1 / (1 + np.exp(-logits.reshape(mh, mw)))

        if isinstance(geometry, MaskDecodeGeometry):
            canvas_box = geometry.transform.canvas_box_from_page(geometry.source_box)
            input_w = geometry.transform.input_w
            input_h = geometry.transform.input_h
        else:
            # Legacy seven-field test/plugin candidates remain supported, but
            # production candidates always carry the explicit transform above.
            canvas_box = geometry
            input_w = INPUT_SIZE
            input_h = INPUT_SIZE
        if canvas_box is None:
            return None

        cx1, cy1, cx2, cy2 = (float(v) for v in canvas_box)
        proto_scale_x = mw / float(input_w)
        proto_scale_y = mh / float(input_h)
        px1 = max(0, min(mw - 1, int(np.floor(cx1 * proto_scale_x))))
        py1 = max(0, min(mh - 1, int(np.floor(cy1 * proto_scale_y))))
        px2 = max(px1 + 1, min(mw, int(np.ceil(cx2 * proto_scale_x))))
        py2 = max(py1 + 1, min(mh, int(np.ceil(cy2 * proto_scale_y))))

        crop = probability_map[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        probabilities = cv2.resize(
            crop,
            (box_w, box_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return (probabilities > DETECTOR_MASK_THRESHOLD).astype(np.uint8) * 255

'''


SANITY_EXTRA = r'''

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


model_contract_checks()
geometry_contract_checks()
publication_safety_checks()
windows_locked_reader_check()
print("backend foundation sanity: phases 1-2 PASS")
'''


def apply_model_contracts() -> None:
    (ROOT / "app/model_contracts.py").write_text(MODEL_CONTRACTS, encoding="utf-8")


def apply_detector() -> None:
    path = ROOT / "app/detector/bubble_detector.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from app.ort_utils import make_session\n",
        "from app.ort_utils import make_session\n"
        "from app.model_contracts import validate_detector_session\n",
        "detector contract import",
    )
    text = replace_once(
        text,
        "@dataclass\nclass BubbleBox:\n",
        LETTERBOX_CLASSES + "@dataclass\nclass BubbleBox:\n",
        "letterbox classes",
    )
    text = replace_once(
        text,
        "    needs_review: bool = False\n",
        "    needs_review: bool = False\n"
        "    source_role: str = \"unknown\"\n",
        "bubble source role",
    )

    old_init = '''    def __init__(self, model_path, conf_threshold: float, use_tta: bool | None = None):\n        self.model_path = str(model_path)\n        self.source_model = Path(model_path).name\n        self.session = make_session(model_path)\n        self.input_name = self.session.get_inputs()[0].name\n        self.conf_threshold = conf_threshold\n        self.use_tta = ENABLE_TTA if use_tta is None else use_tta\n\n    def _class_name(self, class_id: int, num_classes: int) -> str:\n        name = getattr(self, "source_model", "unknown").lower()\n        if "bubble" in name and num_classes >= 2:\n            return "text_bubble" if class_id == 0 else "text_free" if class_id == 1 else f"class_{class_id}"\n        if "text_segmenter" in name or num_classes == 1:\n            return "text_comic"\n        return f"class_{class_id}"\n'''
    new_init = '''    def __init__(\n        self,\n        model_path,\n        conf_threshold: float,\n        use_tta: bool | None = None,\n        *,\n        model_role: str,\n    ):\n        self.model_path = str(model_path)\n        self.source_model = Path(model_path).name\n        self.model_role = str(model_role)\n        self.session = make_session(model_path)\n        self.contract = validate_detector_session(\n            self.session,\n            role=self.model_role,\n            configured_input_size=INPUT_SIZE,\n        )\n        self.input_name = self.contract.input_name\n        self.conf_threshold = conf_threshold\n        self.use_tta = ENABLE_TTA if use_tta is None else use_tta\n\n    def _class_name(self, class_id: int, num_classes: int) -> str:\n        if num_classes != len(self.contract.class_names):\n            return f"class_{class_id}"\n        if 0 <= class_id < len(self.contract.class_names):\n            return self.contract.class_names[class_id]\n        return f"class_{class_id}"\n'''
    text = replace_once(text, old_init, new_init, "detector explicit role")
    text = replace_once(
        text,
        '        segmenter_evidence = "text_segmenter" in box.source_model.lower()\n',
        '        segmenter_evidence = box.source_role == "text_segmenter"\n',
        "detector semantic authority",
    )
    text = replace_once(
        text,
        "            needs_review=not safe,\n",
        "            needs_review=not safe,\n"
        "            source_role=self.model_role,\n",
        "detector role stamp",
    )
    text = replace_once(
        text,
        '            if "text_segmenter" in self.source_model.lower():\n',
        '            if self.model_role == "text_segmenter":\n',
        "tall text role",
    )

    old_plain = '''    def _detect_single_plain(self, image: np.ndarray, offset_x: int, offset_y: int) -> list[BubbleBox]:\n        h, w = image.shape[:2]\n        blob, scale, pad = self._preprocess(image)\n        if blob is None:\n            return []\n        outputs = self.session.run(None, {self.input_name: blob})\n        boxes = self._postprocess(outputs, scale, pad, w, h)\n        if offset_x or offset_y:\n            boxes = [\n                replace(\n                    b,\n                    x1=b.x1 + offset_x,\n                    y1=b.y1 + offset_y,\n                    x2=b.x2 + offset_x,\n                    y2=b.y2 + offset_y,\n                )\n                for b in boxes\n            ]\n        return boxes\n'''
    new_plain = '''    def _detect_single_plain(\n        self,\n        image: np.ndarray,\n        offset_x: int,\n        offset_y: int,\n    ) -> list[BubbleBox]:\n        blob, transform = self._preprocess(\n            image, offset_x=offset_x, offset_y=offset_y\n        )\n        if blob is None or transform is None:\n            return []\n        outputs = self.session.run(None, {self.input_name: blob})\n        return self._postprocess(outputs, transform)\n'''
    text = replace_once(text, old_plain, new_plain, "detector transform ownership")

    start = text.index("    def _preprocess(self, image: np.ndarray):\n")
    end = text.index("    @staticmethod\n    def _candidate_fields", start)
    text = text[:start] + DETECTOR_GEOMETRY_BLOCK + text[end:]

    text = replace_once(
        text,
        '        source_name = str(members[kept[0]].source_model).lower()\n'
        '        if "text_segmenter" not in source_name:\n',
        '        if members[kept[0]].source_role != "text_segmenter":\n',
        "nms role authority",
    )
    text = replace_once(
        text,
        "                score, cid, num_classes, canvas_box, mask_coeffs = self._candidate_fields(c)\n"
        "                mask = self._decode_mask(mask_coeffs, prototypes, canvas_box, x2 - x1, y2 - y1)\n",
        "                score, cid, num_classes, decode_geometry, mask_coeffs = self._candidate_fields(c)\n"
        "                mask = self._decode_mask(\n"
        "                    mask_coeffs, prototypes, decode_geometry, x2 - x1, y2 - y1\n"
        "                )\n",
        "nms decode geometry",
    )
    text = replace_once(
        text,
        "                    semantic_type=self._semantic_type(class_name),\n",
        "                    semantic_type=self._semantic_type(class_name),\n"
        "                    source_role=self.model_role,\n",
        "decoded box role",
    )
    text = replace_once(
        text,
        "        by_class: dict[tuple[str, int], list[BubbleBox]] = {}\n"
        "        for b in boxes:\n"
        "            by_class.setdefault((b.source_model, b.class_id), []).append(b)\n",
        "        by_class: dict[tuple[str, str, int], list[BubbleBox]] = {}\n"
        "        for b in boxes:\n"
        "            by_class.setdefault(\n"
        "                (b.source_role, b.source_model, b.class_id), []\n"
        "            ).append(b)\n",
        "nms role grouping",
    )
    path.write_text(text, encoding="utf-8")


def apply_combined_detector() -> None:
    path = ROOT / "app/detector/combined_detector.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''        self.bubble_detector = YoloDetector(\n            BUBBLE_DETECTOR_MODEL, BUBBLE_PROPOSAL_CONF_THRESHOLD\n        )\n        self.text_detector = YoloDetector(TEXT_SEGMENTER_MODEL, TEXT_CONF_THRESHOLD)\n''',
        '''        self.bubble_detector = YoloDetector(\n            BUBBLE_DETECTOR_MODEL,\n            BUBBLE_PROPOSAL_CONF_THRESHOLD,\n            model_role="bubble_detector",\n        )\n        self.text_detector = YoloDetector(\n            TEXT_SEGMENTER_MODEL,\n            TEXT_CONF_THRESHOLD,\n            model_role="text_segmenter",\n        )\n''',
        "combined detector roles",
    )
    text = replace_once(
        text,
        '        if box.verified_mask and "text_segmenter" in box.source_model.lower():\n',
        '        if box.verified_mask and box.source_role == "text_segmenter":\n',
        "combined destructive role",
    )
    text = replace_once(
        text,
        '            "text_segmenter" in box.source_model.lower()\n'
        '            or box.source_model == "opencv_mser"\n',
        '            box.source_role == "text_segmenter"\n'
        '            or box.source_model == "opencv_mser"\n',
        "combined text evidence role",
    )
    path.write_text(text, encoding="utf-8")


def apply_lama() -> None:
    path = ROOT / "app/inpaint/lama_inpainter.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from app.logging_config import logger\n",
        "from app.logging_config import logger\n"
        "from app.model_contracts import decode_lama_output, validate_lama_session\n",
        "lama contract import",
    )
    text = replace_once(
        text,
        "        self.dynamic_lama = False\n",
        "        self.dynamic_lama = False\n"
        "        self.lama_contract = None\n"
        "        self.output_name = None\n",
        "lama contract state",
    )

    old_configure = '''    def _configure_loaded_session(self, session, model_path) -> None:\n        inputs = session.get_inputs()\n        image_shape = inputs[0].shape\n        dynamic_lama = any(\n            isinstance(dim, str) or dim is None for dim in image_shape[2:4]\n        )\n        self.session = session\n        self.image_input = inputs[0].name\n        self.mask_input = inputs[1].name\n        self.dynamic_lama = dynamic_lama\n        self._serialize_fixed_inference = bool(\n            not dynamic_lama and type(session).__name__ == "_SerializedSession"\n        )\n        self.lama_model_path = model_path\n        self._session_run_count = 0\n        self._recycle_fixed_session = (\n            self._serialize_fixed_inference and _should_recycle_fixed_session()\n        )\n'''
    new_configure = '''    def _configure_loaded_session(\n        self,\n        session,\n        model_path,\n        *,\n        expected_dynamic: bool,\n    ) -> None:\n        contract = validate_lama_session(\n            session,\n            dynamic=bool(expected_dynamic),\n            fixed_size=INPAINT_SIZE,\n        )\n        self.session = session\n        self.lama_contract = contract\n        self.image_input = contract.image_input_name\n        self.mask_input = contract.mask_input_name\n        self.output_name = contract.output_name\n        self.dynamic_lama = contract.dynamic\n        self._serialize_fixed_inference = bool(\n            not contract.dynamic and type(session).__name__ == "_SerializedSession"\n        )\n        self.lama_model_path = model_path\n        self._session_run_count = 0\n        self._recycle_fixed_session = (\n            self._serialize_fixed_inference and _should_recycle_fixed_session()\n        )\n'''
    text = replace_once(text, old_configure, new_configure, "lama configure contract")
    text = replace_once(
        text,
        "            self._configure_loaded_session(session, model_path)\n",
        "            self._configure_loaded_session(\n"
        "                session,\n"
        "                model_path,\n"
        "                expected_dynamic=(model_path == LAMA_DYNAMIC_MODEL),\n"
        "            )\n",
        "lama initial contract",
    )
    text = replace_once(
        text,
        "        self._configure_loaded_session(session, self.lama_model_path)\n",
        "        self._configure_loaded_session(\n"
        "            session, self.lama_model_path, expected_dynamic=False\n"
        "        )\n",
        "lama recycle contract",
    )

    run_count = text.count("output = self.session.run(None, feed)[0]")
    if run_count != 2:
        raise SystemExit(f"lama explicit output: expected two run sites, got {run_count}")
    text = text.replace(
        "output = self.session.run(None, feed)[0]",
        "output = self.session.run([self.output_name], feed)[0]",
    )
    text = replace_once(
        text,
        '''        painted_rgb = output[0].transpose(1, 2, 0)\n        if painted_rgb.max() <= 1.0:\n            painted_rgb = painted_rgb * 255.0\n        painted_rgb = np.clip(painted_rgb, 0, 255).astype(np.uint8)\n        return cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)\n''',
        '''        painted_rgb = decode_lama_output(output, self.lama_contract)\n        return cv2.cvtColor(painted_rgb, cv2.COLOR_RGB2BGR)\n''',
        "lama explicit output range",
    )
    text = replace_once(
        text,
        "                    needs_review=bool(b.needs_review),\n",
        "                    needs_review=bool(b.needs_review),\n"
        "                    source_role=b.source_role,\n",
        "lama local role provenance",
    )
    path.write_text(text, encoding="utf-8")


def apply_sanity() -> None:
    path = ROOT / "scripts/backend_foundation_sanity.py"
    text = path.read_text(encoding="utf-8")
    final = (
        "publication_safety_checks(); windows_locked_reader_check(); "
        "print('backend foundation sanity: publication safety PASS')\n"
    )
    text = replace_once(text, final, SANITY_EXTRA, "phase 2 sanity tail")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    apply_model_contracts()
    apply_detector()
    apply_combined_detector()
    apply_lama()
    apply_sanity()
    print("phase 2 patch applied")


if __name__ == "__main__":
    main()
