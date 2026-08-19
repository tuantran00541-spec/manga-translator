# LaMa Inpaint Benchmark Harness Reference (Phase 0)

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
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2 — LaMa PIPELINE BREAKDOWN                           │
│   Measures individual stages on single crops:               │
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
│   Telemetry: model_calls, cluster_count, active_tile_count  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset & Mask Matrix

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

### 5. Multi-Thread Sweeping (Isolated Subprocesses)
```bash
python -m tools.benchmark_inpaint --run --mode model --threads 1,2,4,8 --output results_sweep.json
```

### 6. Generate Golden Baseline
```bash
python -m tools.benchmark_inpaint --run --mode all --golden data/golden_baseline --output data/golden_baseline/baseline.json
```

### 7. Regression Comparison
```bash
python -m tools.benchmark_inpaint --run --mode all --compare data/golden_baseline/baseline.json --report comparison_report.md
```

---

## 4. Primary Metrics Tracked

- **p50 (Median Latency)**: Primary performance metric representing typical execution time.
- **p95 Latency**: Tail latency representing 95th percentile under CPU contention.
- **Model Calls (`model_calls`)**: P0 metric counting exact number of `session.run()` executions.
- **Preprocess / Postprocess Latency**: Time spent on image resizing, memory formatting, and color space conversions.
- **Active Tile Count**: Number of 512x512 sub-tiles containing active mask pixels during tiled inference.
- **Memory RSS Profile**: Resident Set Size before session creation, during peak inference, and after execution.
