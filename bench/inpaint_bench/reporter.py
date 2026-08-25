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
            "# LaMa Inpainting Benchmark Baseline Report (Phase 0.2 Hardened)",
            "",
            "> **Phase 0 Baseline Tool**: This report captures reproducible baseline measurements for LaMa inpainting. "
            "No optimizations have been implemented in this phase.",
            "",
            "## 1. System Environment",
            f"- **Timestamp**: `{env.timestamp}`",
            f"- **Platform**: `{env.platform}` (`{env.os}`)",
            f"- **CPU Model**: `{env.cpu_model}`",
            f"- **CPUs**: `{env.logical_cpus}` logical / `{env.physical_cpus}` physical cores",
            f"- **Python**: `{env.python_version}`",
            f"- **ONNX Runtime**: `{env.onnxruntime_version}`",
            f"- **OpenCV**: `{env.opencv_version}` | **NumPy**: `{env.numpy_version}`",
            f"- **Git Commit**: `{env.git_commit or 'N/A'}`",
            "",
            "## 2. Model & Baseline Identity",
            f"- **Model Name**: `{mod.model_name}`",
            f"- **Model SHA-256**: `{mod.model_sha256}`",
            f"- **Model Size**: `{mod.model_size_bytes:,}` bytes ({mod.model_size_bytes / (1024*1024):.2f} MB)",
            f"- **Input Shape**: `{mod.input_resolution}` ({mod.data_type})",
            f"- **Execution Provider**: `{mod.execution_provider}`",
            f"- **Baseline Commit SHA**: `{result.baseline_commit_sha}`",
            "",
            "## 3. Benchmark Summary Statistics",
            f"- **Total Cases**: `{len(result.cases)}`",
            f"- **Execution Mode**: `{result.mode}`",
            f"- **Repetitions**: `{result.repetitions}` (warmup: `{result.warmup_count}`)",
            f"- **Thread Configuration**: `{result.thread_configurations}`",
            "",
        ]

        l1_cases = [c for c in result.cases if c.level == "level1_model" and c.status == "ok"]
        if l1_cases:
            lines.extend([
                "### Level 1: Model Benchmark (Raw ORT Inference)",
                "",
            ])
            headers = ["Case ID", "Threads", "Cold (ms)", "p50 (ms)", "p95 (ms)", "Mean (ms)", "StdDev", "Calls/Inv", "RSS Peak (MB)"]
            rows = []
            for c in l1_cases:
                rows.append([
                    c.case_id,
                    c.thread_count or 1,
                    f"{c.cold_total_ms:.2f}",
                    f"{c.timing.p50_ms:.2f}",
                    f"{c.timing.p95_ms:.2f}",
                    f"{c.timing.mean_ms:.2f}",
                    f"{c.timing.stddev_ms:.2f}",
                    c.model_calls_per_invocation if c.model_calls_per_invocation is not None else "var",
                    f"{c.memory.rss_peak_mb:.1f}" if c.memory.measured else "N/A",
                ])
            lines.append("```text")
            lines.append(format_table(headers, rows))
            lines.append("```")
            lines.append("")

        l2_cases = [c for c in result.cases if c.level == "level2_pipeline" and c.status == "ok"]
        if l2_cases:
            lines.extend([
                "### Level 2: Pipeline Benchmark (Preprocess + Inference + Postprocess)",
                "",
            ])
            headers = ["Case ID", "Dim", "Pre (ms)", "Inf (ms)", "Post (ms)", "Total (ms)", "Calls/Inv"]
            rows = []
            for c in l2_cases:
                rows.append([
                    c.case_id,
                    f"{c.image_width}x{c.image_height}",
                    f"{c.preprocess_timing.p50_ms:.2f}",
                    f"{c.inference_timing.p50_ms:.2f}",
                    f"{c.postprocess_timing.p50_ms:.2f}",
                    f"{c.timing.p50_ms:.2f}",
                    c.model_calls_per_invocation if c.model_calls_per_invocation is not None else "var",
                ])
            lines.append("```text")
            lines.append(format_table(headers, rows))
            lines.append("```")
            lines.append("")

        l3_cases = [c for c in result.cases if c.level == "level3_e2e" and c.status == "ok"]
        if l3_cases:
            lines.extend([
                "### Level 3: End-to-End Inpainting Benchmark",
                "",
            ])
            headers = ["Case ID", "Dim", "Mask Ratio", "Total p50 (ms)", "p95 (ms)", "Model Calls", "Shortcuts", "Clusters", "Tiles"]
            rows = []
            for c in l3_cases:
                rows.append([
                    c.case_id,
                    f"{c.image_width}x{c.image_height}",
                    f"{c.mask_ratio:.2%}",
                    f"{c.timing.p50_ms:.2f}",
                    f"{c.timing.p95_ms:.2f}",
                    c.model_calls_per_invocation if c.model_calls_per_invocation is not None else f"var (tot {c.model_calls_total})",
                    c.shortcut_count if c.shortcut_count is not None else "var",
                    c.cluster_count if c.cluster_count is not None else "var",
                    f"{c.active_tile_count}/{c.tile_count}" if c.tile_count else "-",
                ])
            lines.append("```text")
            lines.append(format_table(headers, rows))
            lines.append("```")
            lines.append("")

        if comparisons:
            lines.extend([
                "## 4. Comparison vs Baseline",
                "",
            ])
            headers = ["Case ID", "Base p50", "Cand p50", "Delta (ms)", "Diff %", "Calls Delta", "Status", "Note"]
            rows = []
            for cmp in comparisons:
                status = "REGRESSION" if cmp.regression else "OK"
                if cmp.incompatible:
                    status = "INCOMPATIBLE"
                diff_str = f"{cmp.p50_diff_pct:+.1f}%" if cmp.p50_diff_pct is not None else "N/A"
                delta_str = f"{cmp.delta_p50_ms:+.2f}" if cmp.delta_p50_ms is not None else "N/A"
                base_str = f"{cmp.baseline_p50_ms:.2f}" if cmp.baseline_p50_ms is not None else "N/A"
                cand_str = f"{cmp.candidate_p50_ms:.2f}" if cmp.candidate_p50_ms is not None else "N/A"
                calls_str = f"{cmp.model_calls_delta:+d}" if cmp.model_calls_delta is not None else (
                    f"{cmp.model_calls_mean_delta:+.2f}" if cmp.model_calls_mean_delta is not None else "N/A"
                )
                rows.append([
                    cmp.case_id,
                    base_str,
                    cand_str,
                    delta_str,
                    diff_str,
                    calls_str,
                    status,
                    cmp.note,
                ])
            lines.append("```text")
            lines.append(format_table(headers, rows))
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_console_summary(result: BenchmarkRunResult) -> str:
        lines = [
            "=" * 70,
            f"LaMa INPAINTING BENCHMARK RESULTS (Mode: {result.mode})",
            "=" * 70,
        ]
        l1_cases = [c for c in result.cases if c.level == "level1_model" and c.status == "ok"]
        if l1_cases:
            lines.append(f"[LEVEL 1: MODEL ONLY] ({len(l1_cases)} cases)")
            for c in l1_cases:
                lines.append(f"  • {c.case_id} (th={c.thread_count}): p50={c.timing.p50_ms:.2f}ms | p95={c.timing.p95_ms:.2f}ms | mean={c.timing.mean_ms:.2f}ms (cold={c.cold_total_ms:.1f}ms)")
            lines.append("-" * 70)

        l2_cases = [c for c in result.cases if c.level == "level2_pipeline" and c.status == "ok"]
        if l2_cases:
            lines.append(f"[LEVEL 2: PIPELINE BREAKDOWN] ({len(l2_cases)} cases)")
            for c in l2_cases[:5]:
                lines.append(f"  • {c.case_id}: Pre={c.preprocess_timing.p50_ms:.1f}ms | Inf={c.inference_timing.p50_ms:.1f}ms | Post={c.postprocess_timing.p50_ms:.1f}ms | Total={c.timing.p50_ms:.1f}ms | Calls/Inv={c.model_calls_per_invocation}")
            if len(l2_cases) > 5:
                lines.append(f"  ... and {len(l2_cases) - 5} more cases.")
            lines.append("-" * 70)

        l3_cases = [c for c in result.cases if c.level == "level3_e2e" and c.status == "ok"]
        if l3_cases:
            lines.append(f"[LEVEL 3: END-TO-END INPAINT] ({len(l3_cases)} cases)")
            for c in l3_cases[:5]:
                lines.append(f"  • {c.case_id}: Total p50={c.timing.p50_ms:.1f}ms | Calls/Inv={c.model_calls_per_invocation} | Shortcuts={c.shortcut_count} | Total Calls={c.model_calls_total}")
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
