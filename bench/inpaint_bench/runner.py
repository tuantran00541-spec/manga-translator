from __future__ import annotations
import os
import sys
import json
import time
import math
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.config import LAMA_MODEL
from app.inpaint.lama_inpainter import Inpainter
from app.ort_utils import make_session
from .schema import (
    SCHEMA_VERSION,
    BenchmarkRunResult,
    CaseResult,
    ComparisonDelta,
    QualityThresholds,
    is_finite_number,
    is_finite_int,
    validate_case_execution,
    validate_case_payload_for_comparison,
    validate_benchmark_payload_for_comparison,
)
from .metrics import (
    get_environment_metadata,
    get_model_metadata,
    get_model_sha256,
)
from .integrity import (
    LAMA_MODEL_BASELINE_SHA256,
    compute_file_sha256,
    load_trusted_baseline_manifest,
)
from .corpus_generator import generate_corpus, load_corpus, compute_workload_sha256
from .model_bench import run_model_benchmark
from .pipeline_bench import run_pipeline_benchmark_case
from .e2e_bench import run_e2e_benchmark_case


def compute_image_metrics(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float, float]:
    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
        raise ValueError("Image inputs must be numpy ndarrays")
    if img1.size == 0 or img2.size == 0:
        raise ValueError("Image cannot be empty")
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    if len(img1.shape) != 3 or img1.shape[2] != 3:
        raise ValueError(f"Expected 3-channel BGR image, got shape {img1.shape}")
    if not np.all(np.isfinite(img1)) or not np.all(np.isfinite(img2)):
        raise ValueError("Image contains NaN or Inf pixel values")

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    mae = float(np.mean(diff))

    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        psnr = 100.0
    else:
        psnr = float(20.0 * math.log10(255.0 / math.sqrt(mse)))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu1 = float(np.mean(gray1))
    mu2 = float(np.mean(gray2))
    var1 = float(np.var(gray1))
    var2 = float(np.var(gray2))
    cov = float(np.mean((gray1 - mu1) * (gray2 - mu2)))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denom = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    if denom == 0:
        ssim = 1.0
    else:
        ssim = float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / denom)

    if not is_finite_number(psnr) or not is_finite_number(ssim) or not is_finite_number(mae):
        raise ValueError("Computed non-finite image metric")

    return round(psnr, 2), round(ssim, 4), round(mae, 4)


class BenchmarkRunner:
    def __init__(
        self,
        model_path: Path | str = LAMA_MODEL,
        corpus_dir: Path | str | None = None,
        mode: str = "all",
        threads: int | list[int] = 1,
        repetitions: int = 30,
        warmup: int = 3,
        save_golden_dir: Path | str | None = None,
    ):
        self.model_path = Path(model_path)
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.mode = mode
        self.threads = [threads] if isinstance(threads, int) else threads
        self.repetitions = repetitions
        self.warmup = warmup
        self.save_golden_dir = Path(save_golden_dir) if save_golden_dir else None

    def run(self, isolated_subproc: bool = True) -> BenchmarkRunResult:
        if isolated_subproc and os.getenv("INPAINT_BENCH_ISOLATED") != "1":
            return self._run_isolated_subprocess()
        return self._run_direct()

    def _run_isolated_subprocess(self) -> BenchmarkRunResult:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            env = dict(os.environ)
            env["INPAINT_BENCH_ISOLATED"] = "1"
            cmd = [
                sys.executable,
                "-m",
                "bench.inpaint_bench.runner",
                "--model", str(self.model_path),
                "--mode", self.mode,
                "--repetitions", str(self.repetitions),
                "--warmup", str(self.warmup),
                "--threads", ",".join(str(t) for t in self.threads),
                "--internal-out", str(tmp_path),
            ]
            if self.corpus_dir:
                cmd.extend(["--corpus", str(self.corpus_dir)])
            if self.save_golden_dir:
                cmd.extend(["--save-golden", str(self.save_golden_dir)])

            res = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Isolated benchmark subprocess failed (code {res.returncode}):\n{res.stderr}")

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BenchmarkRunResult.from_dict(data)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _run_direct(self) -> BenchmarkRunResult:
        env_meta = get_environment_metadata()
        model_meta = get_model_metadata(self.model_path)
        cases: list[CaseResult] = []

        baseline_manifest = load_trusted_baseline_manifest()
        baseline_commit = baseline_manifest.get("baseline_commit_sha", "")

        inpainter = None
        if self.mode in ("all", "pipeline", "end-to-end"):
            try:
                inpainter = Inpainter()
            except Exception as e:
                print(f"Warning: Inpainter initialization failed: {e}")

        if self.mode in ("all", "model"):
            for th in self.threads:
                res = run_model_benchmark(
                    model_path=self.model_path,
                    threads=th,
                    warmup=self.warmup,
                    repetitions=self.repetitions,
                )
                res.thread_count = th
                cases.append(res)

        corpus_cases = []
        if self.corpus_dir and self.corpus_dir.is_dir():
            corpus_cases = load_corpus(self.corpus_dir)
        elif self.mode in ("all", "pipeline", "end-to-end"):
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_d:
                corpus_cases = generate_corpus(tmp_d, seed=1234)
                for c in corpus_cases:
                    c["_temp_loaded_img"] = cv2.imread(c["original_path"])
                    c["_temp_loaded_mask"] = cv2.imread(c["mask_path"], cv2.IMREAD_GRAYSCALE)

        if self.mode in ("all", "pipeline") and inpainter:
            for cm in corpus_cases:
                if cm.get("expected_execution") != "model_required":
                    continue
                orig_img = cm.get("_temp_loaded_img") if "_temp_loaded_img" in cm else cv2.imread(cm["original_path"])
                mask_img = cm.get("_temp_loaded_mask") if "_temp_loaded_mask" in cm else cv2.imread(cm["mask_path"], cv2.IMREAD_GRAYSCALE)
                if orig_img is None or mask_img is None:
                    continue

                res = run_pipeline_benchmark_case(
                    inpainter=inpainter,
                    crop_img=orig_img,
                    local_mask=mask_img,
                    case_id=f"pipeline_{cm['case_id']}",
                    expected_execution=cm.get("expected_execution", "model_required"),
                    expected_shortcut_type=cm.get("expected_shortcut_type"),
                    warmup=self.warmup,
                    repetitions=self.repetitions,
                )
                res.workload_sha256 = cm.get("workload_sha256", "")
                cases.append(res)

        if self.mode in ("all", "end-to-end") and inpainter:
            for cm in corpus_cases:
                orig_img = cm.get("_temp_loaded_img") if "_temp_loaded_img" in cm else cv2.imread(cm["original_path"])
                mask_img = cm.get("_temp_loaded_mask") if "_temp_loaded_mask" in cm else cv2.imread(cm["mask_path"], cv2.IMREAD_GRAYSCALE)
                if orig_img is None or mask_img is None:
                    continue

                from app.detector.bubble_detector import BubbleBox
                boxes = [
                    BubbleBox(
                        x1=int(b["x1"]),
                        y1=int(b["y1"]),
                        x2=int(b["x2"]),
                        y2=int(b["y2"]),
                        confidence=float(b.get("confidence", 1.0)),
                    )
                    for b in cm.get("boxes", [])
                ]

                golden_path = None
                if self.save_golden_dir:
                    golden_path = Path(self.save_golden_dir) / f"{cm['case_id']}_golden.png"

                res, _ = run_e2e_benchmark_case(
                    inpainter=inpainter,
                    image=orig_img,
                    boxes=boxes,
                    mask=mask_img,
                    case_id=f"e2e_{cm['case_id']}",
                    expected_execution=cm.get("expected_execution", "model_required"),
                    expected_shortcut_type=cm.get("expected_shortcut_type"),
                    warmup=self.warmup,
                    repetitions=self.repetitions,
                    save_golden_to=golden_path,
                )
                res.workload_sha256 = cm.get("workload_sha256", "")
                cases.append(res)

        summary = {
            "total_cases": len(cases),
            "ok_cases": sum(1 for c in cases if c.status == "ok"),
            "error_cases": sum(1 for c in cases if c.status == "error"),
        }

        return BenchmarkRunResult(
            schema_version=SCHEMA_VERSION,
            mode=self.mode,
            thread_configurations=self.threads,
            repetitions=self.repetitions,
            warmup_count=self.warmup,
            environment=env_meta,
            model=model_meta,
            baseline_commit_sha=baseline_commit,
            summary=summary,
            cases=cases,
        )


def compare_benchmarks(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    telemetry_only: bool = False,
    golden_dir: Path | str | None = None,
) -> list[ComparisonDelta]:
    validate_benchmark_payload_for_comparison(baseline_payload)
    validate_benchmark_payload_for_comparison(candidate_payload)

    trusted_manifest = load_trusted_baseline_manifest()
    thresh_data = trusted_manifest.get("quality_thresholds", {})
    thresholds = QualityThresholds(
        min_psnr=thresh_data.get("min_psnr", 30.0),
        min_ssim=thresh_data.get("min_ssim", 0.85),
        max_mae=thresh_data.get("max_mae", 5.0),
        max_psnr_drop=thresh_data.get("max_psnr_drop", 2.0),
        max_ssim_drop=thresh_data.get("max_ssim_drop", 0.05),
        max_mae_increase=thresh_data.get("max_mae_increase", 2.0),
    )

    deltas: list[ComparisonDelta] = []

    cand_model = candidate_payload.get("model", {})
    cand_sha = cand_model.get("model_sha256", "")
    cand_ep = cand_model.get("execution_provider", "")

    if cand_sha != LAMA_MODEL_BASELINE_SHA256:
        deltas.append(
            ComparisonDelta(
                case_id="model_identity_validation",
                incompatible=True,
                regression=True,
                note=f"Candidate model SHA-256 mismatch: {cand_sha} != {LAMA_MODEL_BASELINE_SHA256}",
            )
        )

    if cand_ep != "CPUExecutionProvider":
        deltas.append(
            ComparisonDelta(
                case_id="model_provider_validation",
                incompatible=True,
                regression=True,
                note=f"Candidate execution provider is not CPU: {cand_ep}",
            )
        )

    base_cases = {c["case_id"]: c for c in baseline_payload.get("cases", []) if "case_id" in c}
    cand_cases = {c["case_id"]: c for c in candidate_payload.get("cases", []) if "case_id" in c}

    base_keys = set(base_cases.keys())
    cand_keys = set(cand_cases.keys())

    for missing in (base_keys - cand_keys):
        deltas.append(
            ComparisonDelta(
                case_id=missing,
                incompatible=True,
                regression=True,
                note="Case present in baseline but missing in candidate",
            )
        )

    for extra in (cand_keys - base_keys):
        deltas.append(
            ComparisonDelta(
                case_id=extra,
                incompatible=True,
                regression=True,
                note="Case present in candidate but missing in baseline",
            )
        )

    common_ids = sorted(list(base_keys & cand_keys))

    for cid in common_ids:
        b_case = base_cases[cid]
        c_case = cand_cases[cid]

        b_exec = b_case.get("expected_execution", "model_required")
        c_exec = c_case.get("expected_execution", "model_required")
        b_stype = b_case.get("expected_shortcut_type")
        c_stype = c_case.get("expected_shortcut_type")

        incompatible = False
        note_parts = []

        if b_exec != c_exec or b_stype != c_stype:
            incompatible = True
            note_parts.append(f"Archetype mismatch: ({b_exec},{b_stype}) vs ({c_exec},{c_stype})")

        b_w_hash = b_case.get("workload_sha256", "")
        c_w_hash = c_case.get("workload_sha256", "")
        if b_w_hash and c_w_hash and b_w_hash != c_w_hash:
            incompatible = True
            note_parts.append(f"Workload content hash mismatch: {b_w_hash[:8]} vs {c_w_hash[:8]}")

        b_valid, b_err = validate_case_execution(b_case)
        c_valid, c_err = validate_case_execution(c_case)
        if not b_valid:
            incompatible = True
            note_parts.append(f"Baseline validation error: {b_err}")
        if not c_valid:
            incompatible = True
            note_parts.append(f"Candidate validation error: {c_err}")

        b_timing = b_case.get("timing", {})
        c_timing = c_case.get("timing", {})

        b_p50 = b_timing.get("p50_ms")
        c_p50 = c_timing.get("p50_ms")
        b_p95 = b_timing.get("p95_ms")
        c_p95 = c_timing.get("p95_ms")

        d_p50 = None
        p50_pct = None
        d_p95 = None
        p95_pct = None

        if is_finite_number(b_p50) and is_finite_number(c_p50):
            d_p50 = round(float(c_p50) - float(b_p50), 4)
            p50_pct = round(((float(c_p50) - float(b_p50)) / float(b_p50)) * 100.0, 2) if float(b_p50) > 0 else 0.0

        if is_finite_number(b_p95) and is_finite_number(c_p95):
            d_p95 = round(float(c_p95) - float(b_p95), 4)
            p95_pct = round(((float(c_p95) - float(b_p95)) / float(b_p95)) * 100.0, 2) if float(b_p95) > 0 else 0.0

        b_calls = b_case.get("model_calls_per_invocation")
        c_calls = c_case.get("model_calls_per_invocation")
        calls_delta = None
        if is_finite_int(b_calls) and is_finite_int(c_calls):
            calls_delta = int(c_calls) - int(b_calls)

        b_telemetry = b_case.get("telemetry_summary", {}).get("model_calls", {})
        c_telemetry = c_case.get("telemetry_summary", {}).get("model_calls", {})
        b_calls_mean = b_telemetry.get("mean")
        c_calls_mean = c_telemetry.get("mean")
        model_calls_mean_delta = None
        if is_finite_number(b_calls_mean) and is_finite_number(c_calls_mean):
            model_calls_mean_delta = round(float(c_calls_mean) - float(b_calls_mean), 4)

        quality_regression = False
        img_psnr = None
        img_ssim = None
        img_mae = None
        psnr_drop = None
        ssim_drop = None
        mae_inc = None

        if not telemetry_only and c_case.get("level") == "level3_e2e":
            cand_golden = c_case.get("golden_output_path")
            base_golden = b_case.get("golden_output_path")

            if golden_dir:
                g_p = Path(golden_dir) / f"{cid}_golden.png"
                if g_p.is_file():
                    base_golden = str(g_p)

            if not cand_golden or not Path(cand_golden).is_file():
                note_parts.append("Candidate golden image missing for Level 3 comparison")
            elif not base_golden or not Path(base_golden).is_file():
                note_parts.append("Baseline golden image missing for Level 3 comparison")
            else:
                img_c = cv2.imread(cand_golden)
                img_b = cv2.imread(base_golden)
                if img_c is None or img_b is None:
                    incompatible = True
                    note_parts.append("Could not load golden comparison images")
                else:
                    try:
                        img_psnr, img_ssim, img_mae = compute_image_metrics(img_b, img_c)
                        if img_psnr < thresholds.min_psnr:
                            quality_regression = True
                            note_parts.append(f"PSNR below floor: {img_psnr} < {thresholds.min_psnr}")
                        if img_ssim < thresholds.min_ssim:
                            quality_regression = True
                            note_parts.append(f"SSIM below floor: {img_ssim} < {thresholds.min_ssim}")
                        if img_mae > thresholds.max_mae:
                            quality_regression = True
                            note_parts.append(f"MAE above ceiling: {img_mae} > {thresholds.max_mae}")
                    except Exception as ex:
                        incompatible = True
                        note_parts.append(f"Image metric error: {ex}")

        is_regression = (
            incompatible
            or quality_regression
            or (p50_pct is not None and p50_pct > 5.0)
            or (calls_delta is not None and calls_delta > 0)
            or (calls_delta is None and model_calls_mean_delta is not None and model_calls_mean_delta > 0.0)
        )

        deltas.append(
            ComparisonDelta(
                case_id=cid,
                workload_sha256=b_w_hash or c_w_hash,
                baseline_p50_ms=b_p50,
                candidate_p50_ms=c_p50,
                delta_p50_ms=d_p50,
                p50_diff_pct=p50_pct,
                baseline_p95_ms=b_p95,
                candidate_p95_ms=c_p95,
                delta_p95_ms=d_p95,
                p95_diff_pct=p95_pct,
                baseline_model_calls=b_calls,
                candidate_model_calls=c_calls,
                model_calls_delta=calls_delta,
                model_calls_mean_delta=model_calls_mean_delta,
                psnr=img_psnr,
                ssim=img_ssim,
                mae=img_mae,
                psnr_drop=psnr_drop,
                ssim_drop=ssim_drop,
                mae_increase=mae_inc,
                psnr_delta=None,
                ssim_delta=None,
                mae_delta=None,
                quality_regression=quality_regression,
                regression=is_regression,
                incompatible=incompatible,
                note="; ".join(note_parts),
            )
        )

    return deltas
