#!/usr/bin/env python3
"""Real-weight Q6_K/Q8_0 row gates against the native scalar quant bridges."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from huggingface_hub import hf_hub_download

from gguf_stream import parse_gguf
from gguf_quant_ref import dequantize_q6_k, dequantize_q8_0, row_nbytes

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_SIZE = 32
Q6K_BLOCK_BYTES = 210
Q8K_BLOCK_BYTES = 292
QK_K = 256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_by_name(directory, name: str):
    for tensor in directory.tensors:
        if tensor.name == name:
            return tensor
    raise KeyError(name)


def read_row(fd: int, tensor, row: int) -> tuple[bytes, int, int]:
    if len(tensor.shape) != 2:
        raise ValueError(f"{tensor.name}: expected 2-D tensor, got {tensor.shape}")
    ne0, ne1 = (int(tensor.shape[0]), int(tensor.shape[1]))
    stride = row_nbytes(tensor.type_name, ne0)
    if tensor.nbytes != stride * ne1:
        raise ValueError(f"{tensor.name}: packed bytes {tensor.nbytes} != row stride {stride} * {ne1}")
    if row < 0 or row >= ne1:
        raise ValueError(f"{tensor.name}: row {row} outside [0,{ne1})")
    data = os.pread(fd, stride, tensor.data_offset + row * stride)
    if len(data) != stride:
        raise IOError(f"{tensor.name}: short row read")
    return data, ne0, ne1


def activation(ne0: int, seed: int) -> list[float]:
    vals = [((i * (37 + seed) + 17 * seed) % 1009 - 504) / 257.0 for i in range(ne0)]
    for block in range(0, ne0, QK_K):
        vals[block] = 2.0 + 0.001 * (block // QK_K + seed)
    return vals


def q8k_dequant(data: bytes, n: int) -> list[float]:
    if n % QK_K or len(data) != (n // QK_K) * Q8K_BLOCK_BYTES:
        raise ValueError("invalid Q8_K packed vector")
    out: list[float] = []
    for base in range(0, len(data), Q8K_BLOCK_BYTES):
        d = float(struct.unpack_from("<f", data, base)[0])
        qs = struct.unpack_from("<256b", data, base + 4)
        out.extend(d * q for q in qs)
    return out


def q6_scale_summary(data: bytes, n: int) -> dict:
    if n % QK_K or len(data) != (n // QK_K) * Q6K_BLOCK_BYTES:
        raise ValueError("invalid Q6_K packed vector")
    bits: list[int] = []
    values: list[float] = []
    for base in range(0, len(data), Q6K_BLOCK_BYTES):
        raw = struct.unpack_from("<H", data, base + 208)[0]
        value = float(struct.unpack("<e", struct.pack("<H", raw))[0])
        bits.append(raw)
        values.append(value)
    subnormal_count = sum(1 for raw in bits if (raw & 0x7C00) == 0 and (raw & 0x03FF) != 0)
    return {
        "block_count": len(bits),
        "subnormal_count": subnormal_count,
        "first8_bits_hex": [f"0x{raw:04x}" for raw in bits[:8]],
        "first8_values": values[:8],
        "min_abs_nonzero": min((abs(v) for v in values if v != 0.0), default=0.0),
        "max_abs": max((abs(v) for v in values), default=0.0),
    }


def q8_0_scale_summary(data: bytes, n: int) -> dict:
    expected = (n // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES
    if n % Q8_0_BLOCK_SIZE or len(data) != expected:
        raise ValueError("invalid Q8_0 packed vector")
    bits: list[int] = []
    values: list[float] = []
    for base in range(0, len(data), Q8_0_BLOCK_BYTES):
        raw = struct.unpack_from("<H", data, base)[0]
        bits.append(raw)
        values.append(float(struct.unpack("<e", struct.pack("<H", raw))[0]))
    return {
        "block_count": len(bits),
        "subnormal_count": sum(1 for raw in bits if (raw & 0x7C00) == 0 and (raw & 0x03FF) != 0),
        "first8_bits_hex": [f"0x{raw:04x}" for raw in bits[:8]],
        "first8_values": values[:8],
        "min_abs_nonzero": min((abs(v) for v in values if v != 0.0), default=0.0),
        "max_abs": max((abs(v) for v in values), default=0.0),
    }


def load_native(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.qwen_quantize_q8_0_scalar.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ]
    lib.qwen_quantize_q8_0_scalar.restype = ctypes.c_int
    lib.qwen_vec_dot_q8_0_q8_0_scalar.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.qwen_vec_dot_q8_0_q8_0_scalar.restype = ctypes.c_float
    lib.qwen_quantize_q8_k_scalar.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ]
    lib.qwen_quantize_q8_k_scalar.restype = ctypes.c_int
    lib.qwen_vec_dot_q6_k_q8_k_scalar.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.qwen_vec_dot_q6_k_q8_k_scalar.restype = ctypes.c_float
    return lib


def native_q6_case(fd: int, directory, lib, name: str, row: int, seed: int) -> dict:
    tensor = tensor_by_name(directory, name)
    if tensor.type_name != "Q6_K":
        raise ValueError(f"{name}: expected Q6_K, got {tensor.type_name}")
    packed, ne0, ne1 = read_row(fd, tensor, row)
    x = activation(ne0, seed)
    x_arr = (ctypes.c_float * ne0)(*x)
    q8_bytes = (ne0 // QK_K) * Q8K_BLOCK_BYTES
    q8_arr = (ctypes.c_uint8 * q8_bytes)()
    rc = lib.qwen_quantize_q8_k_scalar(x_arr, ne0, q8_arr, q8_bytes)
    if rc != 0:
        raise RuntimeError(f"native Q8_K quantization failed rc={rc}")
    q6_arr = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
    native = float(lib.qwen_vec_dot_q6_k_q8_k_scalar(q6_arr, len(packed), q8_arr, q8_bytes, ne0))
    q8_raw = bytes(q8_arr)
    w = dequantize_q6_k(packed, ne0)
    qx = q8k_dequant(q8_raw, ne0)
    expected = math.fsum(a * b for a, b in zip(w, qx))
    err = abs(native - expected)
    limit = 2e-5 * max(1.0, abs(expected))
    return {
        "name": name,
        "type": tensor.type_name,
        "shape": list(tensor.shape),
        "row": row,
        "row_bytes": len(packed),
        "row_sha256": hashlib.sha256(packed).hexdigest(),
        "ne0": ne0,
        "ne1": ne1,
        "q8_k_bytes": q8_bytes,
        "native_dot": native,
        "python_reference_dot": expected,
        "abs_error": err,
        "error_limit": limit,
        "pass": err <= limit,
        "q6_super_scales": q6_scale_summary(packed, ne0),
    }


def native_q8_0_case(fd: int, directory, lib, name: str, row: int, seed: int) -> dict:
    tensor = tensor_by_name(directory, name)
    if tensor.type_name != "Q8_0":
        raise ValueError(f"{name}: expected Q8_0, got {tensor.type_name}")
    packed, ne0, ne1 = read_row(fd, tensor, row)
    x = activation(ne0, seed)
    x_arr = (ctypes.c_float * ne0)(*x)
    q8_bytes = (ne0 // Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES
    q8_arr = (ctypes.c_uint8 * q8_bytes)()
    rc = lib.qwen_quantize_q8_0_scalar(x_arr, ne0, q8_arr, q8_bytes)
    if rc != 0:
        raise RuntimeError(f"native Q8_0 quantization failed rc={rc}")
    weight_arr = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
    native = float(lib.qwen_vec_dot_q8_0_q8_0_scalar(weight_arr, len(packed), q8_arr, q8_bytes, ne0))
    q8_raw = bytes(q8_arr)
    w = dequantize_q8_0(packed, ne0)
    qx = dequantize_q8_0(q8_raw, ne0)
    finite = bool(w) and all(math.isfinite(v) for v in w)
    expected = math.fsum(a * b for a, b in zip(w, qx))
    err = abs(native - expected)
    limit = 2e-5 * max(1.0, abs(expected))
    return {
        "name": name,
        "type": tensor.type_name,
        "shape": list(tensor.shape),
        "row": row,
        "row_bytes": len(packed),
        "row_sha256": hashlib.sha256(packed).hexdigest(),
        "ne0": ne0,
        "ne1": ne1,
        "finite": finite,
        "q8_0_activation_bytes": q8_bytes,
        "native_dot": native,
        "python_reference_dot": expected,
        "abs_error": err,
        "error_limit": limit,
        "pass": finite and err <= limit,
        "weight_scales": q8_0_scale_summary(packed, ne0),
        "activation_scales": q8_0_scale_summary(q8_raw, ne0),
        "min": min(w) if w else None,
        "max": max(w) if w else None,
        "l2_sq": math.fsum(v * v for v in w),
        "first8": w[:8],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = Path(hf_hub_download(REPO, filename=FILE, local_dir=str(args.work_dir)))
    digest = sha256_file(model)
    if digest != SHA256:
        raise RuntimeError(f"SHA256 mismatch: {digest}")
    directory = parse_gguf(model)
    lib = load_native(args.native_lib)

    fd = os.open(model, os.O_RDONLY)
    try:
        q6_cases = [
            native_q6_case(fd, directory, lib, "blk.0.attn_gate.weight", 0, 1),
            native_q6_case(fd, directory, lib, "blk.3.ffn_down.weight", 17, 2),
        ]
        q8_cases = [
            native_q8_0_case(fd, directory, lib, "blk.0.attn_qkv.weight", 0, 3),
            native_q8_0_case(fd, directory, lib, "blk.3.attn_q.weight", 17, 4),
        ]
    finally:
        os.close(fd)

    q6_ok = all(case["pass"] for case in q6_cases)
    q8_ok = all(case["pass"] for case in q8_cases)
    result = {
        "schema": "qwen38-real-quant-row-slice-v3",
        "status": "PASS" if q6_ok and q8_ok else "FAIL",
        "repo": REPO,
        "file": FILE,
        "file_bytes": model.stat().st_size,
        "sha256": digest,
        "architecture": directory.metadata.get("general.architecture"),
        "llama_cpp_reference_revision": "557614e0296ff4a5b6f649737a65ae2076eea2fd",
        "q6_k_native_cases": q6_cases,
        "q8_0_native_cases": q8_cases,
        "max_q6_native_abs_error": max(case["abs_error"] for case in q6_cases),
        "max_q8_0_native_abs_error": max(case["abs_error"] for case in q8_cases),
        "all_q6_native_cases_pass": q6_ok,
        "all_q8_0_native_cases_pass": q8_ok,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
