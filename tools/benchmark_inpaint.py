from __future__ import annotations
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import LAMA_MODEL
from tools.inpaint_bench.corpus_generator import generate_corpus
from tools.inpaint_bench.runner import BenchmarkRunner, compare_benchmarks
from tools.inpaint_bench.reporter import BenchmarkReporter


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
        "--verify-model-hash",
        type=str,
        help="Assert that model SHA-256 matches this expected hash",
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


def main():
    args = parse_args()

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
        return 0

    comparisons = None
    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.is_file():
            with open(compare_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            comparisons = compare_benchmarks(baseline_data, result)

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

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
