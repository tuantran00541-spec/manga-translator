# LaMa Inpaint Benchmark Harness Reference (Phase 0.2.5 Official Baseline Trust Boundary)

## Overview

The **LaMa Inpaint Benchmark Harness** is a dedicated, reproducible benchmarking suite designed to measure the latency, memory footprint, telemetry, and output quality of the LaMa background inpainting pipeline.

> [!IMPORTANT]
> **Phase 0 Scope Disclaimer**: This phase establishes an immutable, fail-closed trust boundary. **No optimization or algorithmic changes to LaMa or production inpainting have been implemented in this phase.**

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
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Trust-Boundary & Fail-Closed Guarantees (Phase 0.2.5)

1. **Immutable Production & Model Integrity Gate**:
   `tools/inpaint_bench/integrity.py` and `tools/inpaint_bench/baseline_manifest.json` verify byte-for-byte SHA-256 hashes against literal baseline constants for:
   - `app/inpaint/lama_inpainter.py`: `1d6046e7fbb64f2db163a8301fa3839aa6400dbdc270fe17fa008fe37ba42a42`
   - `app/ort_utils.py`: `9d5b066d7cefa089d81d2ef39d22be3f5ea27b949bc54b66dfa891e4f4841f39`
   - `models/lama.onnx`: `e4b3e648c668b556942ad7096e23616a2ef74092b1be753d0c9c7f66a2e48fae`
2. **Zero NaN / Inf Tolerance**:
   All numeric fields in raw benchmark JSON payloads, timing, metrics, deltas, and image metrics are strictly verified to be finite real numbers. NaN, $\pm\infty$, booleans-as-ints, and negative values in counts/timings fail immediately.
3. **Elimination of Silent Numeric Fallbacks**:
   No `.get("x", 0)` or default zero fallback for required comparison fields. If telemetry or timing is missing, the comparison reports `incompatible=True`, `regression=True`, and leaves the delta as `None`.
4. **Telemetry Aggregate Consistency Check**:
   Aggregates (`min`, `max`, `mean`, `invariant`) are recomputed from per-invocation telemetry and verified to match the summary dictionary. Contradictions trigger fail-closed rejection.
5. **Exact Case Set & Archetype Matching**:
   Comparisons require an exact 100% 1-to-1 match of case IDs and `(expected_execution, expected_shortcut_type)` tuples.
6. **Golden Image Quality & Degradation Gate**:
   - Floor: `min_psnr = 30.0`, `min_ssim = 0.85`, `max_mae = 5.0`
   - Degradation: `max_psnr_drop = 2.0 dB`, `max_ssim_drop = 0.05`, `max_mae_increase = 2.0`
   - Image comparison is mandatory by default; skipped only when `--compare-telemetry-only` is explicitly declared.
7. **CPU-Only Enforcement**:
   Enforces `CPUExecutionProvider` only. No silent GPU fallback or unverified execution provider.

---

## 3. CLI Usage

### Verify Production and Model Integrity
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
