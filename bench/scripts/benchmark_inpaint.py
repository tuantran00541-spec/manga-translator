from __future__ import annotations
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.config import LAMA_MODEL
from bench.inpaint_bench.corpus_generator import generate_corpus
from bench.inpaint_bench.runner import BenchmarkRunner, compare_benchmarks
from bench.inpaint_bench.reporter import BenchmarkReporter
from bench.inpaint_bench.integrity import verify_production_integrity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0: Reproducible LaMa Inpaint Benchmark Harness"
    )
    parser.add_argument(
        "--generate-corpus",
        nargs="?",
        const="data/benchmark_corpus",
        help="Generate synthetic benchmark corpus to the specified directory (default: data/benchmark_corpus)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/benchmark_corpus",
        help="Path to synthetic or real manga benchmark corpus directory",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the benchmark suite",
    )
    parser.add_argument(
        "--mode",
        choices=["model", "pipeline", "end-to-end", "all"],
        default="all",
        help="Benchmark measurement level: 'model' (Level 1), 'pipeline' (Level 2), 'end-to-end' (Level 3), or 'all'",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1",
        help="Intra-op thread configuration: single int (e.g. '4') or comma-separated list for sweep (e.g. '1,2,4,8')",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=30,
        help="Number of measured repetition runs per case (default: 30 for L1/L2, capped to 10 for L3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup inference runs prior to measurement (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(LAMA_MODEL),
        help=f"Path to LaMa ONNX model (default: {LAMA_MODEL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of corpus cases to process (useful for smoke tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save benchmark result JSON to the specified filepath",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Save human-readable Markdown summary report to the specified filepath",
    )
    parser.add_argument(
        "--golden",
        type=str,
        help="Generate and save golden baseline output images to the specified directory",
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to baseline benchmark JSON file or directory to compare against candidate run",
    )
    parser.add_argument(
        "--compare-telemetry-only",
        action="store_true",
        help="Skip golden image comparison and compare telemetry/timing metrics only",
    )
    parser.add_argument(
        "--verify-model-hash",
        type=str,
        help="Assert that model SHA-256 matches this expected hash",
    )
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="Verify byte-for-byte SHA-256 integrity of production files and actual model before execution",
    )
    parser.add_argument(
        "--subproc",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    return parser.parse_args()


def load_baseline_data(compare_path_str: str) -> tuple[dict | None, Path | None, str | None]:
    path = Path(compare_path_str)
    if not path.exists():
        return None, None, f"Comparison path does not exist: {path}"

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            golden_dir = path.parent / "golden" if (path.parent / "golden").is_dir() else None
            return data, golden_dir, None
        except Exception as ex:
            return None, None, f"Failed to load comparison JSON from {path}: {ex}"

    if path.is_dir():
        canonical_candidates = [
            path / "benchmark_result.json",
            path / "results.json",
            path / "baseline.json",
            path / "result.json",
        ]
        json_file = None
        for c in canonical_candidates:
            if c.is_file():
                json_file = c
                break

        if json_file is None:
            all_json = list(path.glob("*.json"))
            if len(all_json) == 1:
                json_file = all_json[0]
            elif len(all_json) > 1:
                return None, None, f"Ambiguous directory: multiple JSON files found in {path} without canonical benchmark_result.json"
            else:
                return None, None, f"No benchmark JSON found in directory: {path}"

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            golden_dir = path / "golden" if (path / "golden").is_dir() else (path if (path / "output.png").is_file() else None)
            return data, golden_dir, None
        except Exception as ex:
            return None, None, f"Failed to load comparison JSON from {json_file}: {ex}"

    return None, None, f"Invalid path type for comparison: {path}"


def main() -> int:
    args = parse_args()

    if args.verify_integrity:
        valid, report = verify_production_integrity(actual_model_path=args.model)
        if not valid:
            if not args.quiet:
                print("❌ Production and model integrity verification failed!", file=sys.stderr)
                for f, info in report.items():
                    if not info["valid"]:
                        print(f"   {f}: {info['error']}", file=sys.stderr)
            return 1
        if not args.quiet:
            print("✓ Production file and actual model SHA-256 integrity verified.")

    if args.generate_corpus:
        out_dir = Path(args.generate_corpus)
        if not args.quiet:
            print(f"Generating deterministic synthetic benchmark corpus in: {out_dir}...")
        cases = generate_corpus(out_dir)
        if not args.quiet:
            print(f"✓ Generated {len(cases)} synthetic benchmark cases in {out_dir}.")
        if not args.run:
            return 0

    if not args.run and not args.compare:
        if args.verify_integrity:
            return 0
        print("No action specified. Use --help to view available options, --generate-corpus, or --run to execute.")
        return 0

    thread_list = []
    for t_str in args.threads.split(","):
        t_str = t_str.strip()
        if t_str.isdigit():
            thread_list.append(int(t_str))
    if not thread_list:
        thread_list = [1]

    runner = BenchmarkRunner(
        model_path=args.model,
        corpus_dir=args.corpus,
        mode=args.mode,
        threads=thread_list if len(thread_list) > 1 else thread_list[0],
        repetitions=args.repetitions,
        warmup=args.warmup,
        golden_dir=args.golden,
        expected_model_hash=args.verify_model_hash,
        case_limit=args.limit,
    )

    try:
        result = runner.run(isolated_subproc=args.subproc)
    except Exception as ex:
        if args.subproc:
            print(json.dumps({"error": str(ex)}))
        else:
            print(f"❌ Benchmark execution failed: {ex}", file=sys.stderr)
        return 1

    if args.subproc:
        print(json.dumps(result.to_dict()))
        return 0 if result.summary.get("error_cases", 0) == 0 else 1

    comparisons = None
    has_comparison_regression = False
    if args.compare:
        baseline_data, base_golden_dir, err_compare = load_baseline_data(args.compare)
        if err_compare:
            if not args.quiet:
                print(f"❌ {err_compare}", file=sys.stderr)
            return 1

        cand_golden_dir = Path(args.golden) if args.golden else None
        comparisons = compare_benchmarks(
            baseline_data,
            result,
            image_baseline_dir=base_golden_dir,
            image_candidate_dir=cand_golden_dir,
            telemetry_only=args.compare_telemetry_only,
        )
        has_comparison_regression = any(d.regression or d.incompatible for d in comparisons)

    if not args.quiet:
        print(BenchmarkReporter.generate_console_summary(result))

    if args.output:
        BenchmarkReporter.save_json(result, args.output)
        if not args.quiet:
            print(f"✓ Saved machine-readable JSON results to: {args.output}")

    if args.report:
        report_md = BenchmarkReporter.generate_markdown(result, comparisons=comparisons)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report_md)
        if not args.quiet:
            print(f"✓ Saved Markdown report to: {args.report}")

    if result.summary.get("error_cases", 0) > 0 or has_comparison_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
