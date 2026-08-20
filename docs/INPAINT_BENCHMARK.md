# LaMa Inpaint Benchmark Harness Reference (Phase 0.2 Hardened)

## Overview

The **LaMa Inpaint Benchmark Harness** is a dedicated benchmarking suite designed to evaluate the execution latency, throughput, memory consumption, and model invocation metrics of the LaMa background inpainting pipeline.

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
│   shortcut detection (white / black / low-std)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Hardening & Correctness Guarantees (Phase 0.2)

1. **Real Inpainter Execution**:
   All benchmark levels and tests exercise the 100% real `Inpainter` class (`Inpainter.inpaint`, `Inpainter.inpaint_mask`, `Inpainter._lama_fill_single`, `Inpainter._lama_fill_tiled`, `Inpainter._cluster_boxes`).
2. **Accurate Shortcut Detection**:
   `InpaintTelemetryContext` inspects the non-mask region when `_smart_paint_region` exits without entering `_lama_fill`, correctly distinguishing `white`, `black`, and `low_std` shortcuts with `model_calls == 0`.
3. **Deterministic Execution Modes**:
   Corpus cases carry explicit `expected_execution` metadata (`model_required`, `shortcut_white`, `shortcut_black`, `shortcut_low_std`, `mixed`) ensuring that `model_required` cases have rich manga texture and do not accidentally trigger shortcuts.
4. **Aggregate Telemetry Statistics**:
   Telemetry is summarized across all repetitions into `TelemetryAggregate` with `min`, `max`, `mean`, and `invariant` flags instead of assuming `invocations[0]`.

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

### 5. Quick Smoke Test with Subset Limit
```bash
python -m tools.benchmark_inpaint --run --mode all --limit 1 --repetitions 1 --warmup 1
```
