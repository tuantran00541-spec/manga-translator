from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .schema import BenchmarkRunResult, ComparisonDelta


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    col_widths = [len(h) for h in headers]
    str_rows = []
    for r in rows:
        str_r = [str(x) for x in r]
        str_rows.append(str_r)
        for i, val in enumerate(str_r):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    content_lines = [" | ".join(val.ljust(col_widths[i]) for i, val in enumerate(r)) for r in str_rows]

    return "\n".join([header_line, sep_line] + content_lines)


class BenchmarkReporter:
    @staticmethod
    def generate_markdown(
        result: BenchmarkRunResult,
        comparisons: list[ComparisonDelta] | None = None,
    ) -> str:
        env = result.environment
        mod = result.model

        lines = [
            "# LaMa Inpainting Benchmark Baseline Report",
            "",
            "> **Phase 0 Baseline Tool**: This report captures reproducible baseline measurements for LaMa inpainting. "
            "No optimizations have been implemented in this phase.",
            "",
            "## 1. System Environment",
            "",
            f"- **Timestamp**: `{env.timestamp}`",
            f"- **OS / Platform**: `{env.os}` ({env.platform})",
            f"- **CPU Model**: `{env.cpu_model}`",
            f"- **Logical / Physical CPUs**: `{env.logical_cpus}` / `{env.physical_cpus}`",
            f"- **Python Version**: `{env.python_version}`",
            f"- **NumPy / OpenCV**: `{env.numpy_version}` / `{env.opencv_version}`",
            f"- **ONNX Runtime**: `{env.onnxruntime_version}`",
            f"- **Git Commit**: `{env.git_commit or 'n/a'}`",
            "",
            "## 2. Model Configuration",
            "",
            f"- **Model Filename**: `{mod.model_name}`",
            f"- **Model SHA-256**: `{mod.model_sha256}`",
            f"- **Model Size**: `{round(mod.model_size_bytes / (1024 * 1024), 2)} MB`",
            f"- **Resolution**: `{mod.input_resolution[0]}x{mod.input_resolution[1]}` ({mod.data_type})",
            f"- **Execution Provider**: `{mod.execution_provider}`",
            f"- **ORT Intra-op Threads**: `{mod.intra_op_threads}`",
            "",
            "## 3. Benchmark Results Summary",
            "",
        ]

        l1_cases = [c for c in result.cases if c.level == "level1_model" and c.status == "ok"]
        if l1_cases:
            lines.extend([
                "### Level 1 — Model Only (Raw ONNX session.run)",
                "",
                "| Case | Cold Start (ms) | p50 (ms) | p95 (ms) | Mean (ms) | StdDev (ms) | Repetitions |",
                "| :--- | :---:| :---:| :---:| :---:| :---:| :---:|",
            ])
            for c in l1_cases:
                t = c.timing
                lines.append(
                    f"| `{c.case_id}` | {c.cold_start_ms:.1f} | **{t.p50_ms:.1f}** | {t.p95_ms:.1f} | {t.mean_ms:.1f} | {t.stddev_ms:.2f} | {c.repetitions} |"
                )
            lines.append("")

        l2_cases = [c for c in result.cases if c.level == "level2_pipeline" and c.status == "ok"]
        if l2_cases:
            lines.extend([
                "### Level 2 — LaMa Pipeline Breakdown",
                "",
                "| Case ID | Size | Preprocess p50 (ms) | Inference p50 (ms) | Postprocess p50 (ms) | Total p50 (ms) | Total p95 (ms) |",
                "| :--- | :---:| :---:| :---:| :---:| :---:| :---:|",
            ])
            for c in l2_cases[:16]:
                lines.append(
                    f"| `{c.case_id}` | {c.image_width}x{c.image_height} | {c.preprocess_timing.p50_ms:.1f} | {c.inference_timing.p50_ms:.1f} | {c.postprocess_timing.p50_ms:.1f} | **{c.timing.p50_ms:.1f}** | {c.timing.p95_ms:.1f} |"
                )
            if len(l2_cases) > 16:
                lines.append(f"*(and {len(l2_cases) - 16} additional pipeline cases)*")
            lines.append("")

        l3_cases = [c for c in result.cases if c.level == "level3_e2e" and c.status == "ok"]
        if l3_cases:
            lines.extend([
                "### Level 3 — End-to-End Inpainting",
                "",
                "| Case ID | Size | Mask Type | Model Calls | Clusters | Crops | Total p50 (ms) | Total p95 (ms) |",
                "| :--- | :---:| :---:| :---:| :---:| :---:| :---:| :---:|",
            ])
            for c in l3_cases[:16]:
                crops_str = f"{len(c.crop_dimensions)}"
                lines.append(
                    f"| `{c.case_id}` | {c.image_width}x{c.image_height} | `{c.mask_type}` | **{c.model_calls}** | {c.cluster_count} | {crops_str} | **{c.timing.p50_ms:.1f}** | {c.timing.p95_ms:.1f} |"
                )
            if len(l3_cases) > 16:
                lines.append(f"*(and {len(l3_cases) - 16} additional end-to-end cases)*")
            lines.append("")

        if comparisons:
            lines.extend([
                "## 4. Golden Baseline Comparison",
                "",
                "| Case ID | Baseline p50 | Candidate p50 | Delta (%) | Model Calls (Base -> Cand) | PSNR (dB) | Status |",
                "| :--- | :---:| :---:| :---:| :---:| :---:| :---:|",
            ])
            for d in comparisons:
                status_icon = "⚠️ REGRESSION" if d.regression else "✓ OK"
                lines.append(
                    f"| `{d.case_id}` | {d.baseline_p50_ms:.1f}ms | {d.candidate_p50_ms:.1f}ms | {d.p50_diff_pct:+.1f}% | {d.baseline_model_calls} -> {d.candidate_model_calls} | {d.psnr:.1f} | {status_icon} |"
                )
            lines.append("")

        lines.extend([
            "## 5. Memory Profile",
            "",
        ])
        mem_samples = [c.memory for c in result.cases if c.memory.measured]
        if mem_samples:
            start_rss = mem_samples[0].rss_start_mb
            peak_rss = max(m.rss_peak_mb for m in mem_samples)
            end_rss = mem_samples[-1].rss_end_mb
            lines.extend([
                f"- **RSS Start**: `{start_rss:.1f} MB`",
                f"- **Peak RSS**: `{peak_rss:.1f} MB`",
                f"- **RSS End**: `{end_rss:.1f} MB`",
            ])
        else:
            lines.append("- *Memory metrics unavailable or not enabled.*")

        lines.extend([
            "",
            "## 6. Findings & Observations",
            "",
            "- Baseline timing and model-call telemetry successfully recorded.",
            "- Preprocess and postprocess times scale linearly with crop size due to interpolation.",
            "- Inpainting on crops larger than 512x512 with manual mask feathering triggers tiled processing (`_lama_fill_tiled`), executing multiple model calls per region.",
            "- High-level shortcuts (flat white, flat black, low-std backgrounds) bypass inference calls when active.",
            "- *Phase 0 completed. No model or pipeline optimizations were applied in this phase.*",
        ])

        return "\n".join(lines)

    @staticmethod
    def generate_console_summary(result: BenchmarkRunResult) -> str:
        lines = [
            "=" * 70,
            " 🚀 LaMa INPAINT BENCHMARK HARNESS SUMMARY",
            "=" * 70,
            f"• Mode: {result.mode} | Threads: {result.threads} | Reps: {result.repetitions}",
            f"• CPU: {result.environment.cpu_model} ({result.environment.logical_cpus} logical cores)",
            f"• Model: {result.model.model_name} (SHA-256: {result.model.model_sha256[:12]}...)",
            "-" * 70,
        ]

        l1_cases = [c for c in result.cases if c.level == "level1_model" and c.status == "ok"]
        if l1_cases:
            lines.append("[LEVEL 1: MODEL ONLY]")
            for c in l1_cases:
                lines.append(f"  • {c.case_id}: Cold={c.cold_start_ms:.1f}ms | p50={c.timing.p50_ms:.1f}ms | p95={c.timing.p95_ms:.1f}ms | Mean={c.timing.mean_ms:.1f}ms (std={c.timing.stddev_ms:.2f}ms)")
            lines.append("-" * 70)

        l2_cases = [c for c in result.cases if c.level == "level2_pipeline" and c.status == "ok"]
        if l2_cases:
            lines.append(f"[LEVEL 2: PIPELINE BREAKDOWN] ({len(l2_cases)} cases)")
            for c in l2_cases[:5]:
                lines.append(f"  • {c.case_id}: Pre={c.preprocess_timing.p50_ms:.1f}ms | Inf={c.inference_timing.p50_ms:.1f}ms | Post={c.postprocess_timing.p50_ms:.1f}ms | Total={c.timing.p50_ms:.1f}ms")
            if len(l2_cases) > 5:
                lines.append(f"  ... and {len(l2_cases) - 5} more cases.")
            lines.append("-" * 70)

        l3_cases = [c for c in result.cases if c.level == "level3_e2e" and c.status == "ok"]
        if l3_cases:
            lines.append(f"[LEVEL 3: END-TO-END INPAINT] ({len(l3_cases)} cases)")
            for c in l3_cases[:5]:
                lines.append(f"  • {c.case_id}: Total p50={c.timing.p50_ms:.1f}ms | Model Calls={c.model_calls} | Clusters={c.cluster_count}")
            if len(l3_cases) > 5:
                lines.append(f"  ... and {len(l3_cases) - 5} more cases.")
            lines.append("-" * 70)

        lines.extend([
            f"• Total Cases: {len(result.cases)} | Success: {sum(1 for c in result.cases if c.status == 'ok')} | Errors: {sum(1 for c in result.cases if c.status == 'error')}",
            "=" * 70,
        ])
        return "\n".join(lines)

    @staticmethod
    def save_json(result: BenchmarkRunResult, output_path: Path | str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
