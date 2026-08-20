# LaMa Inpaint Benchmark Harness Reference (Phase 0.2.1 Final Gate)

## Overview

The **LaMa Inpaint Benchmark Harness** is a rigorous, reproducible benchmarking suite designed to measure the execution latency, throughput, memory footprint, and model invocation metrics of the LaMa background inpainting pipeline.

> [!IMPORTANT]
> **Phase 0 Scope Disclaimer**: This phase establishes an immutable, reproducible measurement baseline. **No optimization or algorithmic changes to LaMa or production inpainting have been implemented in this phase.**

---

## 1. Benchmark Architecture (3 Measurement Levels)

```text
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1 — MODEL ONLY                                        │
│   Isolates raw ONNX Runtime inference latency               │
│   ORT session.run({image: 512x512, mask: 512x512})          │
│   Metrics: session_create_ms, first_inference_ms, p50/p95   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2 — PRODUCTION LaMa PIPELINE BREAKDOWN                │
│   Invokes production Inpainter._lama_fill_single() directly │
│   Intercepts execution stages via TelemetrySessionProxy:    │
│   • Preprocess: scale, border replicate, normalize, tensor  │
│   • Inference: session.run()                                │
│   • Postprocess: unpad, clip, RGB->BGR, resize              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 3 — END-TO-END INPAINTING                             │
│   Invokes production Inpainter with full pipeline:          │
│   Clustering ──▶ Crops ──▶ Shortcuts ──▶ Tiling ──▶ Blend   │
│   Telemetry: model_calls per invocation (independent reset),│
│   cluster_count, active_tile_count, crop dimensions,        │
│   exact shortcut detection ('white', 'black', 'low_std')   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Correctness Guarantees (Phase 0.2.1 Final Gate)

1. **Real Inpainter Execution**:
   All benchmark levels exercise the real `Inpainter` class (`Inpainter.inpaint`, `Inpainter.inpaint_mask`, `Inpainter._lama_fill_single`, `Inpainter._lama_fill_tiled`, `Inpainter._cluster_boxes`).
2. **Exact Shortcut Type Validation**:
   Metadata carries explicit `expected_execution` (`model_required`, `shortcut`) and `expected_shortcut_type` (`white`, `black`, `low_std`). Runner validates exact observed shortcut types and raises error on mismatches.
3. **Telemetry Invariant Semantics**:
   `model_calls_per_invocation` represents the exact per-invocation count when invariant across repetitions; otherwise `None` (mean is preserved in `telemetry_summary.model_calls.mean`).
4. **Production Code Immutability**:
   Production files (`app/inpaint/lama_inpainter.py` and `app/ort_utils.py`) remain completely untouched.

---

## 3. CLI Reference

### 1. Generate Deterministic Synthetic Corpus
```bash
python -m tools.benchmark_inpaint --generate-corpus data/benchmark_corpus
```

### 2. Run Level 1 (Model-Only) Benchmark
```bash
python -m tools.benchmark_inpaint --run --mode model --threads 4 --repetitions 30
```

### 3. Run Level 2 (Pipeline Breakdown) Benchmark
```bash
python -m tools.benchmark_inpaint --run --mode pipeline --threads 4 --repetitions 30 --output results_l2.json
```

### 4. Run Level 3 (End-to-End) Benchmark
```bash
python -m tools.benchmark_inpaint --run --mode end-to-end --threads 4 --repetitions 5 --output results_l3.json
```

### 5. Multi-Archetype Quick Smoke Test
```bash
python -m tools.benchmark_inpaint --run --mode all --limit 4 --repetitions 1 --warmup 1
```
