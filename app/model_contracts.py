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

    return np.clip(np.rint(arr[0].transpose(1, 2, 0)), 0, 255).astype(np.uint8)
