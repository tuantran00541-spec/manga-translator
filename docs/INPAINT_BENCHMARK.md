# LaMa Inpaint Benchmark Harness Reference (Phase 0.2.4 Official Baseline)

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

## 2. Fail-Closed Guarantees (Phase 0.2.4)

1. **Immutable Production Integrity Gate**:
   `tools/inpaint_bench/integrity.py` verifies byte-for-byte SHA-256 hashes against literal baseline constants for `app/inpaint/lama_inpainter.py` and `app/ort_utils.py`. Fails on any single-byte mutation, deletion, line-ending difference (CRLF/LF), or missing file.
2. **Raw JSON Payload Validation Before Deserialization**:
   Raw benchmark dictionaries are strictly validated before conversion to dataclasses. Missing required fields (such as `timing.p50_ms`, `telemetry_summary.model_calls`, `invocations`) never silently fall back to zero.
3. **Fail-Closed Status Contract**:
   Any case with `status != "ok"` (including `"skipped"`, `"error"`, or `null`) automatically triggers `incompatible=True` and `regression=True`.
4. **Exact Case Set & Archetype Matching**:
   Comparisons require an exact 1-to-1 match between baseline and candidate case IDs and `(expected_execution, expected_shortcut_type)`. Mismatched, missing, or unexpected cases cause immediate comparison failure.
5. **Golden Image Absolute Quality & Degradation Thresholds**:
   - Floor: `min_psnr = 30.0`, `min_ssim = 0.85`, `max_mae = 5.0`
   - Degradation: `max_psnr_drop = 2.0 dB`, `max_ssim_drop = 0.05`, `max_mae_increase = 2.0`
   - Missing, unreadable, or dimension-mismatched golden images fail closed.
6. **CLI Comparison File & Directory Support**:
   `--compare` accepts a single benchmark JSON file or a directory containing benchmark JSON results and golden directories.
7. **Strict CLI Exit Code**:
   Returns non-zero exit code on any integrity mismatch, schema discrepancy, benchmark error case, or comparison regression.

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
