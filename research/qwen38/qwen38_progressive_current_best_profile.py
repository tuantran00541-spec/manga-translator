#!/usr/bin/env python3
"""Exact current-best Q6 profile with the progressive execution-ordered K3 backend.

All decoder arithmetic and promoted quant/native helpers are inherited from
qwen38_current_best_q6_profile.  This wrapper changes only the K3 packer and
reader selected by StatefulK3Generator, then reuses the existing hidden/state,
coverage, direct-I/O, ring-budget, and K3-byte gates unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import qwen35_k3_generate as gen
import qwen38_current_best_q6_profile as base
from qwen38_k3_progressive import (
    K3_STREAM_BYTES,
    LAYOUT_POLICY,
    ProgressiveK3Trunk,
    pack_gguf_layers_progressive,
    synthetic_sanity,
)


def sanity() -> None:
    base.sanity()
    synthetic_sanity()
    print("QWEN38_PROGRESSIVE_CURRENT_BEST_PROFILE_SANITY PASS")


def run(args):
    old_pack = gen.pack_gguf_layers
    old_reader = gen.K3Trunk
    gen.pack_gguf_layers = pack_gguf_layers_progressive
    gen.K3Trunk = ProgressiveK3Trunk
    try:
        payload = base.run(args)
    finally:
        gen.pack_gguf_layers = old_pack
        gen.K3Trunk = old_reader

    reader = payload["reader"]
    if reader.get("layout_policy") != LAYOUT_POLICY:
        raise RuntimeError(f"progressive layout policy missing from reader: {reader}")
    if not bool(reader.get("progressive_readiness")):
        raise RuntimeError("progressive readiness was not active")
    if int(reader.get("storage_io_concurrency", -1)) != 1:
        raise RuntimeError("progressive profile changed storage I/O concurrency")
    if int(reader.get("ring_slots", -1)) != 2:
        raise RuntimeError("progressive profile changed ring slots")
    if int(reader.get("planned_bytes", -1)) != 672_899_072:
        raise RuntimeError("progressive profile changed ring residency budget")
    if int(reader.get("bytes_read", -1)) != K3_STREAM_BYTES:
        raise RuntimeError("progressive profile changed K3 stream bytes")
    if int(reader.get("max_queued_requests_observed", 99)) > 1:
        raise RuntimeError("progressive profile exceeded one deferred layer request")

    wait_seconds = float(reader.get("tensor_wait_seconds", 0.0))
    raw_residual = float(payload["residual_non_bind_orchestration_seconds"])
    non_io_residual = raw_residual - wait_seconds
    if non_io_residual < -0.15:
        raise RuntimeError(
            f"progressive profile accounting became negative after tensor waits: {non_io_residual}")

    payload["schema"] = "qwen38-progressive-current-best-q6-profile-v1"
    payload["claim"] = (
        "exact promoted Q6-pool2 11-token decoder-prefill with execution-ordered "
        "two-slot progressive K3 readiness and one storage I/O at a time")
    payload["progressive_tensor_wait_seconds"] = wait_seconds
    payload["residual_non_bind_orchestration_seconds_before_progressive_wait_split"] = raw_residual
    payload["residual_non_io_orchestration_seconds"] = non_io_residual
    payload["optimization"].update({
        "reader_policy_change": True,
        "execution_ordered_k3_layout": True,
        "progressive_tensor_readiness": True,
        "storage_io_concurrency": 1,
        "max_deferred_layer_requests": 1,
        "ring_change": False,
        "ring_slots": 2,
        "planned_bytes": 672_899_072,
        "tensor_byte_change": False,
        "arithmetic_change": False,
    })
    payload["proof_sub_layer_frontier_run"] = 33863640313
    payload["proof_sub_layer_frontier_commit"] = "3ee9c823fc091cb579d14cd3fc8698331b20cdcd"

    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "prefill_seconds": payload["prefill_seconds"],
        "k3_bytes": payload["k3_bytes"],
        "hidden_sha256": payload["hidden_sha256"],
        "state_sha256": payload["state_sha256"],
        "reader_bind_seconds": payload["reader_bind_seconds"],
        "progressive_tensor_wait_seconds": wait_seconds,
        "q6_boundary_seconds": payload["q6_boundary_seconds"],
        "residual_non_io_orchestration_seconds": non_io_residual,
        "reader": reader,
        "max_rss_gib": payload["max_rss_gib"],
    }, indent=2, ensure_ascii=False))
    print("QWEN38_PROGRESSIVE_CURRENT_BEST_REAL_BITWISE_PASS")
    return payload


def main() -> int:
    args = base.parser().parse_args()
    if args.mode == "sanity":
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
