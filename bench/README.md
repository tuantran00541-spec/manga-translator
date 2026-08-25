# Benchmark workspace

`bench/` is the single source of truth for benchmark code and benchmark ground truth.

- `bench/inpaint_bench/` contains the reusable LaMa benchmark library.
- `bench/scripts/` contains executable benchmark, profiler, reviewer, and audit entry points.
- `bench/ground_truth/` contains versioned benchmark ground-truth JSON.
- `debug/` contains diagnostic scripts that are useful during development but are not benchmark entry points.
- `tools/` is reserved for UI/developer utilities that have not been migrated into the benchmark workspace.

Run the inpaint harness with:

```bash
python -m bench.scripts.benchmark_inpaint --verify-integrity
python -m bench.scripts.benchmark_inpaint --generate-corpus data/benchmark_corpus
python -m bench.scripts.benchmark_inpaint --run --mode all --output results.json --report report.md
```

Benchmark source files must not be duplicated at the repository root or under `tools/`. Generated reports and corpora should be written to explicit output paths rather than committed beside the benchmark source.
