# Benchmark workspace

`bench/` is the single source of truth for benchmark implementations and benchmark ground truth.

- `bench/inpaint_bench/` contains the reusable LaMa benchmark library with the same trusted behavior/contracts that existed on `main` before the move.
- `bench/scripts/` contains executable benchmark, profiler, reviewer, and audit entry points.
- `bench/ground_truth/` contains versioned benchmark ground-truth JSON.
- `debug/` contains diagnostic scripts that are useful during development but are not benchmark entry points.
- `tools/benchmark_inpaint.py` is a compatibility shim only; it delegates to the canonical CLI and contains no benchmark implementation.
- Other files in `tools/` are UI/developer utilities.

Run the inpaint harness with:

```bash
python -m bench.scripts.benchmark_inpaint --verify-integrity
python -m bench.scripts.benchmark_inpaint --generate-corpus data/benchmark_corpus
python -m bench.scripts.benchmark_inpaint --run --mode all --output results.json --report report.md
```

Run the current detect → box → mask correctness benchmark through the stable entry point:

```bash
python -m bench.scripts.detect_box_mask_bench --images path/to/page.webp
```

`detect_box_mask_bench.py` delegates to the current v3 correctness engine. The older v1/v2 implementations are intentionally not kept as parallel copies; benchmark behavior has one implementation source of truth.

Benchmark implementations must not be duplicated at the repository root or under `tools/`. Generated reports and corpora should be written to explicit output paths rather than committed beside the benchmark source.
