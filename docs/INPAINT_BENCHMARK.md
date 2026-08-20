# LaMa Inpaint Benchmark Harness Reference (Phase 0.2.2 Final Gate)

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

## 2. Telemetry Semantics & Guarantees (Phase 0.2.2)

1. **Per-Invocation Exact Validation**:
   `validate_case_execution()` verifies every single measured invocation independently, rejecting any run where an invocation deviates from the expected archetype.
2. **Non-Invasive Shortcut Classification**:
   When `_smart_paint_region` exits without entering `_lama_fill`, the non-mask pixel statistics are evaluated to classify `white`, `black`, or `low_std` shortcuts. Any unclassified shortcut (`unknown`) is treated as a validation failure.
3. **Strict Telemetry Semantics**:
   - `model_calls_per_invocation`: Holds the exact scalar integer count **only** when invariant across all repetitions; otherwise `None`.
   - `telemetry_summary.model_calls.mean`: Holds the exact arithmetic mean across repetitions.
4. **Exception-Safe Restoration**:
   `InpaintTelemetryContext` guarantees that all original production methods and sessions on `Inpainter` are restored even if an exception occurs.
5. **Production Code Immutability**:
   `app/inpaint/lama_inpainter.py` and `app/ort_utils.py` remain strictly untouched.

---

## 3. CLI Usage

### Generate Synthetic Benchmark Corpus
```bash
python -m tools.benchmark_inpaint --generate-corpus data/benchmark_corpus
```

### Run Benchmark Modes
```bash
python -m tools.benchmark_inpaint --run --mode model --threads 1 --repetitions 30
python -m tools.benchmark_inpaint --run --mode pipeline --threads 1 --repetitions 30
python -m tools.benchmark_inpaint --run --mode end-to-end --threads 1 --repetitions 5
```

### Strict Archetype Smoke Test
```bash
python -m tools.benchmark_inpaint --run --mode all --limit 4 --repetitions 1 --warmup 1
```
