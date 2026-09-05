#!/usr/bin/env python3
"""Cross-platform fingerprint for exact libm-dependent Qwen3.8 scalar helpers.

This is diagnostic-only.  It does not change decoder arithmetic.  The same
fixed F32 input corpus is evaluated on Linux/glibc and Windows/UCRT so hosted
CI can prove whether platform libm semantics are bitwise identical before a
full GGUF run is spent on another cross-platform anchor check.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys

if sys.platform == "win32":
    from qwen38_win32_bootstrap import install_resource_compat

    install_resource_compat()

import qwen35_k3_full64_ggml_exact as exact

RECORD = struct.Struct("<IIIII")
RANDOM_CASES = 262_144
GRID_DENOM = 256
GRID_LIMIT = 8_192


def _bind_platform_expf() -> str:
    if sys.platform == "win32":
        ucrt = ctypes.CDLL("ucrtbase.dll")
        ucrt.expf.argtypes = [ctypes.c_float]
        ucrt.expf.restype = ctypes.c_float
        exact._LIBM = ucrt
        return "ucrtbase.dll!expf"
    return str(getattr(exact, "_name", None) or "python-math-exp-fallback")


def _f32(x: float) -> float:
    return exact.f32(float(x))


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(x)))[0]


def _from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def _softplusf(x: float) -> float:
    xf = _f32(x)
    if xf > 20.0:
        return xf
    if xf < -20.0:
        return _f32(exact.expf(xf))
    return _f32(math.log1p(float(exact.expf(xf))))


def _siluf(x: float) -> float:
    return exact.mul_f32(x, exact.sigmoid_f32(x))


def _corpus_bits() -> list[int]:
    fixed = (
        -80.0, -40.0, -32.0, -20.0, -10.0, -5.0, -2.0, -1.0,
        -0.5, -0.125, -0.0, 0.0, 0.125, 0.5, 1.0, 2.0, 5.0,
        10.0, 20.0, 32.0, 40.0, 80.0,
    )
    values: list[int] = [_bits(x) for x in fixed]
    values.extend(_bits(i / GRID_DENOM) for i in range(-GRID_LIMIT, GRID_LIMIT + 1))

    # Deterministic LCG, mapped to F32 values in [-32, 32).  This samples many
    # mantissas rather than checking only a friendly regular grid.
    state = 0x6D2B79F5
    scale = 64.0 / float(1 << 24)
    for _ in range(RANDOM_CASES):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append(_bits(((state >> 8) * scale) - 32.0))

    seen: set[int] = set()
    unique: list[int] = []
    for bits in values:
        if bits not in seen:
            seen.add(bits)
            unique.append(bits)
    return unique


def fingerprint(output: Path, summary: Path) -> dict:
    backend = _bind_platform_expf()
    corpus = _corpus_bits()
    digest = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        for input_bits in corpus:
            x = _from_bits(input_bits)
            record = RECORD.pack(
                input_bits,
                _bits(exact.expf(x)),
                _bits(exact.sigmoid_f32(x)),
                _bits(_softplusf(x)),
                _bits(_siluf(x)),
            )
            f.write(record)
            digest.update(record)
    payload = {
        "schema": "qwen38-cross-platform-libm-fingerprint-v1",
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "python": sys.version,
        "expf_backend": backend,
        "cases": len(corpus),
        "record_bytes": RECORD.size,
        "fingerprint_sha256": digest.hexdigest(),
        "output_bytes": output.stat().st_size,
        "arithmetic_change": False,
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("QWEN38_CROSS_PLATFORM_LIBM_FINGERPRINT_PASS")
    return payload


def compare(linux: Path, windows: Path, output: Path) -> dict:
    left = linux.read_bytes()
    right = windows.read_bytes()
    if len(left) != len(right) or len(left) % RECORD.size:
        raise RuntimeError(
            f"fingerprint shape mismatch linux={len(left)} windows={len(right)} record={RECORD.size}")

    fields = ("input", "expf", "sigmoid", "softplus", "silu")
    mismatch_counts = {name: 0 for name in fields[1:]}
    first_mismatches: list[dict] = []
    cases = len(left) // RECORD.size
    for idx in range(cases):
        start = idx * RECORD.size
        lrec = RECORD.unpack_from(left, start)
        wrec = RECORD.unpack_from(right, start)
        if lrec[0] != wrec[0]:
            raise RuntimeError(f"input corpus diverged at case {idx}: {lrec[0]:08x} != {wrec[0]:08x}")
        changed: dict[str, dict[str, str]] = {}
        for pos, name in enumerate(fields[1:], start=1):
            if lrec[pos] != wrec[pos]:
                mismatch_counts[name] += 1
                changed[name] = {
                    "linux_bits": f"0x{lrec[pos]:08x}",
                    "windows_bits": f"0x{wrec[pos]:08x}",
                }
        if changed and len(first_mismatches) < 24:
            first_mismatches.append({
                "case": idx,
                "input_bits": f"0x{lrec[0]:08x}",
                "input_f32": _from_bits(lrec[0]),
                "changed": changed,
            })

    total = sum(mismatch_counts.values())
    payload = {
        "schema": "qwen38-cross-platform-libm-compare-v1",
        "cases": cases,
        "bitwise_match": total == 0,
        "mismatch_counts": mismatch_counts,
        "first_mismatches": first_mismatches,
        "claim": (
            "diagnostic comparison only; a divergence identifies a platform-libm candidate "
            "but does not by itself prove the full-model hidden-anchor root cause"
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if total:
        print("QWEN38_CROSS_PLATFORM_LIBM_DIVERGENCE_FOUND")
    else:
        print("QWEN38_CROSS_PLATFORM_LIBM_BITWISE_MATCH")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fp = sub.add_parser("fingerprint")
    p_fp.add_argument("--output", type=Path, required=True)
    p_fp.add_argument("--summary", type=Path, required=True)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--linux", type=Path, required=True)
    p_cmp.add_argument("--windows", type=Path, required=True)
    p_cmp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "fingerprint":
        fingerprint(args.output, args.summary)
    else:
        compare(args.linux, args.windows, args.output)


if __name__ == "__main__":
    main()
