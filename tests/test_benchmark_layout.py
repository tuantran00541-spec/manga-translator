from pathlib import Path

from bench.inpaint_bench.integrity import (
    LAMA_INPAINTER_BASELINE_SHA256,
    compute_file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_GROUND_TRUTH = (
    "bench/ground_truth/bench_gt_frozen_candidate.json",
    "bench/ground_truth/bench_gt_probe_002.json",
    "bench/ground_truth/m3_e2e.json",
)

LEGACY_BENCHMARK_PATHS = (
    "bench_gt_frozen_candidate.json",
    "bench_gt_probe_002.json",
    "m3_e2e.json",
    "debug_detect.py",
    "debug_text_threshold_sweep.py",
    "tools/bench_gt_frozen_candidate.json",
    "tools/benchmark_tta.py",
    "tools/detect_box_mask_baseline_cpu.py",
    "tools/detect_box_mask_bench.py",
    "tools/detect_box_mask_bench_v2_fixed.py",
    "tools/detect_box_mask_bench_v3.py",
    "tools/detect_box_mask_reviewer.py",
    "tools/gt_audit.py",
    "tools/inpaint_bench",
    "tools/iou_matcher.py",
    "tools/profile_yolo_detector.py",
    "tools/test_repaint_pipeline.py",
    "tools/test_stale_concurrency.py",
    "bench/scripts/detect_box_mask_bench_v2_fixed.py",
)


def test_benchmark_layout_has_one_source_of_truth():
    assert (ROOT / "bench/inpaint_bench").is_dir()
    assert (ROOT / "bench/scripts").is_dir()
    assert (ROOT / "debug").is_dir()

    missing = [path for path in CANONICAL_GROUND_TRUTH if not (ROOT / path).is_file()]
    assert not missing, f"Missing canonical benchmark ground truth: {missing}"

    leftovers = [path for path in LEGACY_BENCHMARK_PATHS if (ROOT / path).exists()]
    assert not leftovers, f"Legacy benchmark duplicates remain: {leftovers}"


def test_benchmark_source_directory_is_not_gitignored():
    ignored_lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "bench/" not in ignored_lines


def test_lama_integrity_baseline_matches_production_source():
    production_file = ROOT / "app/inpaint/lama_inpainter.py"
    assert compute_file_sha256(production_file) == LAMA_INPAINTER_BASELINE_SHA256


def test_detect_box_mask_benchmark_has_one_implementation():
    stable_entry = ROOT / "bench/scripts/detect_box_mask_bench.py"
    current_engine = ROOT / "bench/scripts/detect_box_mask_bench_v3.py"
    assert stable_entry.is_file()
    assert current_engine.is_file()
    entry_text = stable_entry.read_text(encoding="utf-8")
    assert "from bench.scripts.detect_box_mask_bench_v3 import main" in entry_text
    assert len(entry_text.splitlines()) < 40


def test_legacy_inpaint_cli_is_only_a_thin_compatibility_shim():
    shim = ROOT / "tools/benchmark_inpaint.py"
    assert shim.is_file()
    text = shim.read_text(encoding="utf-8")
    assert "from bench.scripts.benchmark_inpaint import main" in text
    assert len(text.splitlines()) < 30
