from __future__ import annotations

import math

import torch

MODEL_ID = "Qwen/Qwen3.8-27B"
PINNED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PREFIX = "model.language_model.layers"
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
MLP_PARTS = ("gate_proj", "up_proj", "down_proj")


def validate_official_metadata(config: dict, index: dict) -> dict:
    errors: list[str] = []
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        errors.append(f"architectures={config.get('architectures')!r}")
    if config.get("model_type") != "qwen3_5":
        errors.append(f"model_type={config.get('model_type')!r}")
    tc = config.get("text_config") or {}
    expected = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "full_attention_interval": 4,
    }
    for key, wanted in expected.items():
        if tc.get(key) != wanted:
            errors.append(f"text_config.{key}={tc.get(key)!r} expected={wanted!r}")
    layer_types = tc.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != LAYERS:
        errors.append("text_config.layer_types must contain 64 entries")
    else:
        for layer, kind in enumerate(layer_types):
            wanted = "full_attention" if layer % 4 == 3 else "linear_attention"
            if kind != wanted:
                errors.append(f"layer_types[{layer}]={kind!r} expected={wanted!r}")
                break

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        errors.append("model.safetensors.index.json has no weight_map")
    else:
        for layer in range(LAYERS):
            for part in MLP_PARTS:
                name = f"{PREFIX}.{layer}.mlp.{part}.weight"
                if name not in weight_map:
                    errors.append(f"missing tensor {name}")
                    break
            if errors:
                break

    if errors:
        raise ValueError("official Qwen3.8-27B metadata validation failed: " + "; ".join(errors))
    return {
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "weight_map_entries": len(weight_map),
        "source_shards": len(set(weight_map.values())),
    }


def tensor_name(layer: int, part: str) -> str:
    if layer < 0 or layer >= LAYERS:
        raise ValueError(f"layer must be in [0,{LAYERS - 1}]")
    if part not in MLP_PARTS:
        raise ValueError(f"unsupported MLP part: {part}")
    return f"{PREFIX}.{layer}.mlp.{part}.weight"


def qdq_rows(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Symmetric RTN fake-quantization, grouping along each row's input dimension."""
    if w.ndim != 2:
        raise ValueError("w must be rank-2")
    if not 2 <= bits <= 8:
        raise ValueError("bits must be in [2,8]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    _, cols = w.shape
    pad = (-cols) % group_size
    w2 = torch.nn.functional.pad(w, (0, pad)) if pad else w
    blocks = w2.reshape(w.shape[0], -1, group_size)
    qmax = (1 << (bits - 1)) - 1
    amax = blocks.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    q = torch.round(blocks / scale).clamp(-qmax, qmax)
    return (q * scale).reshape(w.shape[0], -1)[:, :cols]


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a = reference.reshape(-1).double()
    b = candidate.reshape(-1).double()
    err = b - a
    anorm = torch.linalg.vector_norm(a)
    enorm = torch.linalg.vector_norm(err)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b) / denom) if float(denom) else (1.0 if torch.equal(a, b) else 0.0)
    mse = torch.mean(err * err)
    return {
        "mse": float(mse),
        "rmse": float(torch.sqrt(mse)),
        "max_abs": float(err.abs().max()),
        "relative_l2": float(enorm / anorm) if float(anorm) else float(enorm),
        "cosine": cosine,
    }


def norm_channel_scores(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    """Legacy baseline: combined L2 weight energy per SwiGLU intermediate channel."""
    _validate_mlp_shapes(gate, up, down)
    score = torch.zeros(gate.shape[0], dtype=torch.float64)
    for w in (gate, up):
        for start in range(0, w.shape[0], chunk_size):
            c = w[start:start + chunk_size].float().double()
            score[start:start + c.shape[0]] += (c * c).sum(dim=1)
    for start in range(0, down.shape[0], chunk_size):
        c = down[start:start + chunk_size].float().double()
        score += (c * c).sum(dim=0)
    return score


def activation_energy_channel_scores(
    calibration_x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, dict]:
    """Rank channels by estimated expected squared output contribution.

    score_j = E[(SiLU(gate_j(x)) * up_j(x))^2] * ||down[:,j]||_2^2

    This captures activation weakness/sparsity while remaining cheap enough for
    a layer-local pruning pilot. Cross-channel cancellation is not modeled.
    """
    _validate_mlp_shapes(gate, up, down)
    if calibration_x.ndim != 2 or calibration_x.shape[1] != gate.shape[1]:
        raise ValueError("calibration_x shape must be [samples, hidden]")
    if calibration_x.shape[0] == 0:
        raise ValueError("calibration_x must contain at least one sample")

    inter = gate.shape[0]
    score = torch.empty(inter, dtype=torch.float64)
    rms = torch.empty(inter, dtype=torch.float64)
    mean_abs = torch.empty(inter, dtype=torch.float64)
    x = calibration_x.float()

    down_energy = torch.zeros(inter, dtype=torch.float64)
    for start in range(0, down.shape[0], chunk_size):
        c = down[start:start + chunk_size].float().double()
        down_energy += (c * c).sum(dim=0)

    for start in range(0, inter, chunk_size):
        end = min(start + chunk_size, inter)
        g = x @ gate[start:end].float().T
        u = x @ up[start:end].float().T
        h = torch.nn.functional.silu(g) * u
        h64 = h.double()
        activation_energy = (h64 * h64).mean(dim=0)
        score[start:end] = activation_energy * down_energy[start:end]
        rms[start:end] = torch.sqrt(activation_energy)
        mean_abs[start:end] = h64.abs().mean(dim=0)

    return score, {
        "post_gate_rms_mean": float(rms.mean()),
        "post_gate_rms_median": float(rms.median()),
        "post_gate_mean_abs_mean": float(mean_abs.mean()),
        "score_min": float(score.min()),
        "score_median": float(score.median()),
        "score_max": float(score.max()),
    }


def select_keep_indices(scores: torch.Tensor, prune_channels: int, alignment: int = 128) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    inter = scores.numel()
    if prune_channels <= 0 or prune_channels >= inter:
        raise ValueError("prune_channels must be in (0, intermediate)")
    keep_n = inter - prune_channels
    if alignment > 1 and keep_n % alignment:
        raise ValueError(f"kept intermediate {keep_n} is not aligned to {alignment}")
    return torch.topk(scores, k=keep_n, largest=True, sorted=False).indices.sort().values


def linear_rows(
    x: torch.Tensor,
    w: torch.Tensor,
    row_idx: torch.Tensor | None = None,
    bits: int | None = None,
    group_size: int = 128,
    chunk_size: int = 256,
) -> torch.Tensor:
    nout = w.shape[0] if row_idx is None else row_idx.numel()
    out = torch.empty((x.shape[0], nout), dtype=torch.float32)
    for start in range(0, nout, chunk_size):
        wc = w[start:start + chunk_size].float() if row_idx is None else w.index_select(0, row_idx[start:start + chunk_size]).float()
        if bits is not None:
            wc = qdq_rows(wc, bits, group_size)
        out[:, start:start + wc.shape[0]] = x.float() @ wc.T
    return out


def linear_down(
    x: torch.Tensor,
    down: torch.Tensor,
    keep: torch.Tensor | None = None,
    bits: int | None = None,
    group_size: int = 128,
    chunk_size: int = 256,
) -> torch.Tensor:
    out = torch.empty((x.shape[0], down.shape[0]), dtype=torch.float32)
    for start in range(0, down.shape[0], chunk_size):
        wc = down[start:start + chunk_size].float()
        if keep is not None:
            wc = wc.index_select(1, keep)
        if bits is not None:
            wc = qdq_rows(wc, bits, group_size)
        out[:, start:start + wc.shape[0]] = x.float() @ wc.T
    return out


def mlp_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    keep: torch.Tensor | None = None,
    bits: int | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    g = linear_rows(x, gate, keep, bits, group_size)
    u = linear_rows(x, up, keep, bits, group_size)
    h = torch.nn.functional.silu(g) * u
    return linear_down(h, down, keep, bits, group_size)


def projected_qbytes(hidden: int, intermediate: int, bits: int, group_size: int) -> dict:
    params = 3 * hidden * intermediate
    scales = 2 * intermediate * math.ceil(hidden / group_size) + hidden * math.ceil(intermediate / group_size)
    payload_bytes = (params * bits + 7) // 8
    fp16_scale_bytes = scales * 2
    return {
        "params": params,
        "payload_bytes": payload_bytes,
        "fp16_scale_bytes": fp16_scale_bytes,
        "projected_total_bytes": payload_bytes + fp16_scale_bytes,
    }


def aligned_prune_count(intermediate: int, fraction: float, alignment: int = 128) -> int:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    target_keep = intermediate * (1.0 - fraction)
    aligned_keep = int(round(target_keep / alignment) * alignment)
    aligned_keep = min(intermediate - alignment, max(alignment, aligned_keep))
    return intermediate - aligned_keep


def parse_csv_ints(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def _validate_mlp_shapes(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("MLP weights must be rank-2")
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")
    if down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError(f"down shape {tuple(down.shape)} incompatible with gate {tuple(gate.shape)}")
