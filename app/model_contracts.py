from __future__ import annotations

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
class DetectorTensorContract:
    input_name: str
    input_height: int
    input_width: int
    detection_output_name: str
    detection_shape: tuple
    output_names: tuple[str, ...]
    prototype_output_name: str | None = None
    prototype_shape: tuple | None = None


@dataclass(frozen=True)
class DetectorModelContract:
    """Legacy production schema layered on top of a tensor-only contract."""

    role: str
    input_name: str
    input_height: int
    input_width: int
    class_names: tuple[str, ...]
    provides_prototypes: bool


def validate_detector_tensor_session(
    session,
    *,
    configured_input_size: int,
    expected_feature_count: int,
    prototype_channels: int | None = None,
) -> DetectorTensorContract:
    """Validate ONNX tensor geometry without assigning cleanup authority."""

    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 1:
        raise ValueError(f"detector requires exactly one input; got {len(inputs)}")

    inp = inputs[0]
    if inp.name != "images":
        raise ValueError(f"detector input must be named 'images'; got {inp.name!r}")
    _check_float(inp, "detector input")
    shape = _shape(inp)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise ValueError(f"detector input must be NCHW [1,3,H,W]; got {shape}")
    if not all(isinstance(v, int) and v > 0 for v in shape[2:4]):
        raise ValueError(f"detector spatial shape must be static; got {shape}")

    height, width = int(shape[2]), int(shape[3])
    if height != width or height != int(configured_input_size):
        raise ValueError(
            f"detector model is {width}x{height}, incompatible with "
            f"configured input size {configured_input_size}"
        )

    output_by_name = {item.name: item for item in outputs}
    output0 = output_by_name.get("output0")
    if output0 is None:
        raise ValueError("detector is missing output0")
    _check_float(output0, "detector output0")
    detection_shape = _shape(output0)
    if len(detection_shape) != 3 or detection_shape[0] != 1:
        raise ValueError(
            f"detector output0 must be rank-3 with batch 1; got {detection_shape}"
        )
    if detection_shape[1] != int(expected_feature_count):
        raise ValueError(
            f"detector output0 feature count must be {expected_feature_count}; "
            f"got {detection_shape}"
        )

    prototype_name = None
    prototype_shape = None
    if prototype_channels is not None:
        proto = output_by_name.get("output1")
        if proto is None:
            raise ValueError("segmentation detector is missing output1 prototypes")
        _check_float(proto, "detector output1")
        prototype_shape = _shape(proto)
        if (
            len(prototype_shape) != 4
            or prototype_shape[0] != 1
            or prototype_shape[1] != int(prototype_channels)
            or not all(isinstance(v, int) and v > 0 for v in prototype_shape[2:4])
        ):
            raise ValueError(
                "detector prototype shape must be static "
                f"[1,{prototype_channels},H,W]; got {prototype_shape}"
            )
        prototype_name = "output1"

    return DetectorTensorContract(
        input_name=inp.name,
        input_height=height,
        input_width=width,
        detection_output_name="output0",
        detection_shape=detection_shape,
        output_names=tuple(item.name for item in outputs),
        prototype_output_name=prototype_name,
        prototype_shape=prototype_shape,
    )


def validate_detector_session(
    session,
    *,
    role: str,
    configured_input_size: int,
) -> DetectorModelContract:
    """Compatibility validator for the two current production YOLOv8 models."""

    specs = {
        "bubble_detector": (("text_bubble", "text_free"), False),
        "text_segmenter": (("text_comic",), True),
    }
    if role not in specs:
        raise ValueError(f"Unknown detector model role: {role!r}")

    class_names, needs_proto = specs[role]
    expected_features = 4 + len(class_names) + (32 if needs_proto else 0)
    tensor = validate_detector_tensor_session(
        session,
        configured_input_size=configured_input_size,
        expected_feature_count=expected_features,
        prototype_channels=32 if needs_proto else None,
    )

    expected_outputs = {"output0", "output1"} if needs_proto else {"output0"}
    if set(tensor.output_names) != expected_outputs:
        raise ValueError(
            f"{role} outputs must be exactly {sorted(expected_outputs)}; "
            f"got {sorted(tensor.output_names)}"
        )

    return DetectorModelContract(
        role=role,
        input_name=tensor.input_name,
        input_height=tensor.input_height,
        input_width=tensor.input_width,
        class_names=class_names,
        provides_prototypes=needs_proto,
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
