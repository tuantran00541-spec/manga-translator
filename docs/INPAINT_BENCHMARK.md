# LaMa Inpaint Benchmark Harness Reference (Phase 0 / Phase 0.1 Correctness Fixes)

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
│   cluster_count, active_tile_count, crop dimensions         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Correctness Guarantees (Phase 0.1)

1. **Zero Duplication of Production Code (Level 2)**:
   Level 2 measures the live production method `Inpainter._lama_fill_single(crop, local_mask)`. Stage timing boundaries are measured transparently via `PipelineStageTrackerProxy` without duplicating any image resizing, normalization, or conversion logic.

2. **Per-Invocation Telemetry Reset (Level 3)**:
   Telemetry (`model_calls`, `cluster_count`, `tile_count`, `active_tile_count`, `shortcut_count`, `crop_dimensions`) is reset immediately before every single measured invocation and warmup run. Values never accumulate across repetitions.

3. **Safe Session Proxying**:
   The harness uses `TelemetrySessionProxy` wrapping `InferenceSession` instances instead of monkey-patching C-extension methods directly on the ORT class.

---

## 3. Dataset & Mask Matrix

The benchmark corpus consists of deterministic synthetic manga artwork across 12 resolutions and 8 mask archetypes:

### Supported Image Sizes
- `128x128`, `256x256`, `384x384`, `512x512`, `640x640`, `768x768`, `1024x1024`, `1536x1024`
- Wide / Tall Webtoon strips: `1000x300`, `300x1000`, `1600x300`, `300x1600`

### Supported Mask Archetypes (M1 to M8)
| Mask ID | Type | Description |
| :--- | :--- | :--- |
| `M1_bubble_10pct` | 10% Bubble | Standard centered oval dialogue bubble |
| `M2_bubble_25pct` | 25% Bubble | Large dialogue bubble |
| `M3_bubble_50pct` | 50% Bubble | Half-panel dialogue / splash box |
| `M4_thin_horizontal` | Narration Strip | Thin wide horizontal banner (aspect ~5:1) |
| `M5_thin_vertical` | Vertical Text | Thin tall vertical Japanese dialogue column (aspect ~1:5) |
| `M6_irregular_blob` | SFX Polygon | Jagged starburst sound effect shape |
| `M7_disconnected_multi` | Disconnected Multi-box | 3 separate speech bubbles scattered across quadrants |
| `M8_clustered_multi` | Clustered Multi-box | 3 adjacent speech boxes within 20px clustering threshold |

---

## 4. CLI Reference

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

### 5. Multi-Thread Sweeping (Isolated Subprocesses)
```bash
python -m tools.benchmark_inpaint --run --mode model --threads 1,2,4,8 --output results_sweep.json
```

### 6. Quick Smoke Test with Subset Limit
```bash
python -m tools.benchmark_inpaint --run --mode all --limit 2 --repetitions 1 --warmup 1
```

### 7. Generate Golden Baseline
```bash
python -m tools.benchmark_inpaint --run --mode all --golden data/golden_baseline --output data/golden_baseline/baseline.json
```

### 8. Regression Comparison
```bash
python -m tools.benchmark_inpaint --run --mode all --compare data/golden_baseline/baseline.json --report comparison_report.md
```
