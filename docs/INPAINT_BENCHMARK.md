# LaMa Inpaint Benchmark Harness Reference (Phase 0.2.3 Final Gate)

## Overview

The **LaMa Inpaint Benchmark Harness** is a dedicated, reproducible benchmarking suite designed to measure the latency, memory footprint, and telemetry of the LaMa background inpainting pipeline.

> [!IMPORTANT]
> **Phase 0 Scope Disclaimer**: This phase establishes an immutable, reproducible measurement baseline. **No optimization or algorithmic changes to LaMa or production inpainting have been implemented in this phase.**

---

## 1. Benchmark Architecture (3 Distinct Measurement Levels)

```text
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1 — MODEL ONLY                                        │
│   Isolates raw ONNX Runtime inference latency               │
│   ORT session.run({image: 512x512, mask: 512x512})          │
│   No corpus execution archetypes / shortcuts                │
│   Metrics: session_create_ms, first_inference_ms, p50/p95   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2 — PRODUCTION LaMa PIPELINE BREAKDOWN                │
│   Invokes production Inpainter._lama_fill_single() directly │
│   Only accepts 'model_required' cases                       │
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
│   Evaluates all execution archetypes:                       │
│   • model_required: model_calls >= 1, shortcuts == 0        │
│   • shortcut (white/black/low_std): calls == 0, sc == 1     │
│   • mixed: calls >= 1, sc >= 1                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Integrity & Comparison Hardening (Phase 0.2.3)

1. **Immutable Production Integrity Gate**:
   `tools/inpaint_bench/integrity.py` verifies byte-for-byte SHA-256 hashes against literal baseline constants for `app/inpaint/lama_inpainter.py` and `app/ort_utils.py`. Fails on any single-byte mutation or missing file.
2. **Strict Golden Image Regression**:
   `compare_benchmarks()` enforces explicit quality thresholds (`min_psnr`, `min_ssim`, `max_mae`). Any image quality drop or missing golden output triggers `quality_regression=True` and `regression=True`.
3. **Exact Archetype Comparison**:
   Detects and rejects mismatched execution archetypes and shortcut types between baseline and candidate (e.g. `white -> black`, `shortcut -> model_required`).
4. **Missing Case Detection**:
   Ensures 100% case alignment. Missing baseline cases or unexpected candidate cases flag `incompatible=True` and `regression=True`.
5. **Schema Compatibility**:
   Rejects mismatched or missing `schema_version`.
6. **Fail-Closed Comparison**:
   Runs containing `status == "error"`, missing telemetry, or integrity failures cannot produce valid performance deltas.

---

## 3. CLI Usage

### Verify Production Integrity
```bash
python -m tools.benchmark_inpaint --verify-integrity
```

### Generate Synthetic Benchmark Corpus
```bash
python -m tools.benchmark_inpaint --generate-corpus data/benchmark_corpus
```

### Run Benchmark Suite & Compare
```bash
python -m tools.benchmark_inpaint --run --mode all --output results.json --report report.md
python -m tools.benchmark_inpaint --run --compare baseline.json
```
