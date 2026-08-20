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
        golden_dir: Path | str | None = None,
        expected_model_hash: str | None = None,
        case_limit: int | None = None,
    ):
        self.model_path = Path(model_path).resolve()
        self.corpus_dir = Path(corpus_dir).resolve() if corpus_dir else None
        self.mode = mode
        self.threads = [threads] if isinstance(threads, int) else list(threads)
        self.repetitions = repetitions
        self.warmup = warmup
        self.golden_dir = Path(golden_dir).resolve() if golden_dir else None
        self.expected_model_hash = expected_model_hash or LAMA_MODEL_BASELINE_SHA256
        self.case_limit = case_limit

    def validate_runtime(self):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Model and pipeline benchmarks cannot execute.")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LaMa ONNX model not found at: {self.model_path}")

        actual_hash = compute_file_sha256(self.model_path)
        if actual_hash.lower() != self.expected_model_hash.lower():
            raise ValueError(
                f"Model SHA-256 mismatch for {self.model_path}! Expected {self.expected_model_hash}, but found {actual_hash}"
            )

        test_sess = make_session(self.model_path, intra_op_threads=1)
        providers = test_sess.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise ValueError(f"Execution provider mismatch! Expected ['CPUExecutionProvider'], but got {providers}")

    def run(self, isolated_subproc: bool = False) -> BenchmarkRunResult:
        self.validate_runtime()

        if len(self.threads) > 1 and not isolated_subproc:
            return self._run_thread_sweep()

        active_thread_count = self.threads[0] if self.threads else 1
        env_meta = get_environment_metadata()
        model_meta = get_model_metadata(self.model_path, intra_op_threads=active_thread_count)

        cases: list[CaseResult] = []

        if self.mode in ("model", "all"):
            case_l1 = run_model_benchmark(
                self.model_path,
                threads=active_thread_count,
                warmup=self.warmup,
                repetitions=self.repetitions,
            )
            cases.append(case_l1)

        corpus_cases = []
        if self.mode in ("pipeline", "end-to-end", "all"):
            if self.corpus_dir and self.corpus_dir.is_dir():
                corpus_cases = load_corpus(self.corpus_dir)
            if not corpus_cases:
                synthetic_dir = Path("data/benchmark_corpus")
                if not synthetic_dir.is_dir():
                    generate_corpus(synthetic_dir)
                corpus_cases = load_corpus(synthetic_dir)

            if self.case_limit and self.case_limit > 0:
                corpus_cases = corpus_cases[: self.case_limit]

        if self.mode in ("pipeline", "all") and corpus_cases:
            inpainter = Inpainter()
            inpainter.session = make_session(self.model_path, intra_op_threads=active_thread_count)

            l2_cases = [c for c in corpus_cases if c.get("expected_execution") == "model_required"]
            for c_info in l2_cases:
                orig_img = cv2.imread(c_info["original_path"])
                mask_img = cv2.imread(c_info["mask_path"], cv2.IMREAD_GRAYSCALE)
                if orig_img is None or mask_img is None:
                    continue
                case_id = f"pipeline_{c_info.get('case_id', 'unknown')}"
                case_l2 = run_pipeline_benchmark_case(
                    inpainter,
                    orig_img,
                    mask_img,
                    case_id=case_id,
                    expected_execution="model_required",
                    expected_shortcut_type=None,
                    warmup=self.warmup,
                    repetitions=self.repetitions,
                )
                case_l2.workload_sha256 = c_info.get("workload_sha256", "")
                case_l2.original_sha256 = c_info.get("original_sha256", "")
                case_l2.mask_sha256 = c_info.get("mask_sha256", "")

                valid, err_msg = validate_case_execution(case_l2)
                if not valid:
                    case_l2.status = "error"
                    case_l2.error_message = err_msg

                cases.append(case_l2)

        if self.mode in ("end-to-end", "all") and corpus_cases:
            inpainter = Inpainter()
            inpainter.session = make_session(self.model_path, intra_op_threads=active_thread_count)

            e2e_reps = max(1, min(self.repetitions, 10))
            for c_info in corpus_cases:
                orig_img = cv2.imread(c_info["original_path"])
                mask_img = cv2.imread(c_info["mask_path"], cv2.IMREAD_GRAYSCALE)
                if orig_img is None:
                    continue

                case_id = f"e2e_{c_info.get('case_id', 'unknown')}"
                exp_exec = c_info.get("expected_execution", "model_required")
                exp_sc_type = c_info.get("expected_shortcut_type", None)
                golden_path = None
                if self.golden_dir:
                    golden_path = self.golden_dir / case_id / "output.png"

                case_l3, _ = run_e2e_benchmark_case(
                    inpainter,
                    orig_img,
                    mask=mask_img,
                    case_id=case_id,
                    expected_execution=exp_exec,
                    expected_shortcut_type=exp_sc_type,
                    warmup=min(self.warmup, 2),
                    repetitions=e2e_reps,
                    save_golden_path=golden_path,
                )
                case_l3.workload_sha256 = c_info.get("workload_sha256", "")
                case_l3.original_sha256 = c_info.get("original_sha256", "")
                case_l3.mask_sha256 = c_info.get("mask_sha256", "")

                valid, err_msg = validate_case_execution(case_l3)
                if not valid:
                    case_l3.status = "error"
                    case_l3.error_message = err_msg

                cases.append(case_l3)

        summary: dict[str, Any] = {
            "total_cases": len(cases),
            "ok_cases": sum(1 for c in cases if c.status == "ok"),
            "error_cases": sum(1 for c in cases if c.status == "error"),
        }

        return BenchmarkRunResult(
            schema_version=SCHEMA_VERSION,
            mode=self.mode,
            threads=active_thread_count,
            warmup_count=self.warmup,
            repetitions=self.repetitions,
            environment=env_meta,
            model=model_meta,
            cases=cases,
            summary=summary,
        )

    def _run_thread_sweep(self) -> BenchmarkRunResult:
        env_meta = get_environment_metadata()
        model_meta = get_model_metadata(self.model_path)
        all_cases: list[CaseResult] = []

        for t in self.threads:
            if t > env_meta.logical_cpus:
                all_cases.append(
                    CaseResult(
                        case_id=f"thread_sweep_{t}T",
                        level=f"threads_{t}",
                        status="error",
                        error_message=f"Host CPU has {env_meta.logical_cpus} logical cores; {t} threads configuration is invalid.",
                    )
                )
                continue

            cmd = [
                sys.executable,
                "-m",
                "tools.benchmark_inpaint",
                "--run",
                "--mode",
                self.mode,
                "--threads",
                str(t),
                "--repetitions",
                str(self.repetitions),
                "--warmup",
                str(self.warmup),
                "--subproc",
            ]
            if self.corpus_dir:
                cmd.extend(["--corpus", str(self.corpus_dir)])
            if self.golden_dir:
                cmd.extend(["--golden", str(self.golden_dir)])
            if self.case_limit:
                cmd.extend(["--limit", str(self.case_limit)])

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                try:
                    data = json.loads(res.stdout)
                    valid_payload, err_payload = validate_benchmark_payload_for_comparison(data)
                    if not valid_payload:
                        all_cases.append(
                            CaseResult(
                                case_id=f"thread_{t}T",
                                level=f"threads_{t}",
                                status="error",
                                error_message=f"Subprocess output failed schema validation: {err_payload}",
                            )
                        )
                        continue

                    sub_cases = data.get("cases", [])
                    for sc in sub_cases:
                        valid_c, err_c = validate_case_payload_for_comparison(sc)
                        if not valid_c:
                            all_cases.append(
                                CaseResult(
                                    case_id=f"[{t}T] {sc.get('case_id', 'unknown')}",
                                    level=f"threads_{t}",
                                    status="error",
                                    error_message=f"Subprocess case failed validation: {err_c}",
                                )
                            )
                        else:
                            sc["case_id"] = f"[{t}T] {sc.get('case_id', '')}"
                            all_cases.append(CaseResult.from_dict(sc))
                except Exception as ex:
                    all_cases.append(
                        CaseResult(
                            case_id=f"thread_{t}T",
                            level=f"threads_{t}",
                            status="error",
                            error_message=f"Failed to parse subprocess output: {ex}",
                        )
                    )
            else:
                all_cases.append(
                    CaseResult(
                        case_id=f"thread_{t}T",
                        level=f"threads_{t}",
                        status="error",
                        error_message=f"Subprocess exit code {res.returncode}: {res.stderr}",
                    )
                )

        return BenchmarkRunResult(
            schema_version=SCHEMA_VERSION,
            mode=self.mode,
            threads=self.threads[0] if self.threads else 1,
            warmup_count=self.warmup,
            repetitions=self.repetitions,
            environment=env_meta,
            model=model_meta,
            cases=all_cases,
            summary={
                "total_cases": len(all_cases),
                "ok_cases": sum(1 for c in all_cases if c.status == "ok"),
                "error_cases": sum(1 for c in all_cases if c.status == "error"),
            },
        )


def compare_benchmarks(
    baseline_result: BenchmarkRunResult | dict,
    candidate_result: BenchmarkRunResult | dict,
    image_baseline_dir: Path | None = None,
    image_candidate_dir: Path | None = None,
    quality_thresholds: QualityThresholds | None = None,
    telemetry_only: bool = False,
) -> list[ComparisonDelta]:
    try:
        manifest = load_trusted_baseline_manifest()
        q_dict = manifest.get("quality_thresholds", {})
        thresholds = QualityThresholds(
            min_psnr=float(q_dict.get("min_psnr", 30.0)),
            min_ssim=float(q_dict.get("min_ssim", 0.85)),
            max_mae=float(q_dict.get("max_mae", 5.0)),
            max_psnr_drop=float(q_dict.get("max_psnr_drop", 2.0)),
            max_ssim_drop=float(q_dict.get("max_ssim_drop", 0.05)),
            max_mae_increase=float(q_dict.get("max_mae_increase", 2.0)),
        )
    except Exception:
        thresholds = quality_thresholds or QualityThresholds()

    b_dict = baseline_result if isinstance(baseline_result, dict) else baseline_result.to_dict()
    c_dict = candidate_result if isinstance(candidate_result, dict) else candidate_result.to_dict()

    valid_b, err_b = validate_benchmark_payload_for_comparison(b_dict)
    valid_c, err_c = validate_benchmark_payload_for_comparison(c_dict)

    if not valid_b or not valid_c:
        err_msg = err_b if not valid_b else err_c
        return [
            ComparisonDelta(
                case_id="<global_schema_validation>",
                incompatible=True,
                regression=True,
                note=f"Benchmark payload validation failed: {err_msg}",
            )
        ]

    b_model = b_dict.get("model", {})
    c_model = c_dict.get("model", {})
    if c_model.get("model_sha256") and c_model["model_sha256"].lower() != LAMA_MODEL_BASELINE_SHA256.lower():
        return [
            ComparisonDelta(
                case_id="<model_identity_validation>",
                incompatible=True,
                regression=True,
                note=f"Candidate model SHA-256 mismatch: {c_model['model_sha256']}",
            )
        ]
    if c_model.get("execution_provider") and c_model["execution_provider"] != "CPUExecutionProvider":
        return [
            ComparisonDelta(
                case_id="<provider_identity_validation>",
                incompatible=True,
                regression=True,
                note=f"Candidate execution provider mismatch: {c_model['execution_provider']}",
            )
        ]

    b_cases = {c["case_id"]: c for c in b_dict.get("cases", [])}
    c_cases = {c["case_id"]: c for c in c_dict.get("cases", [])}

    all_case_ids = list(b_cases.keys())
    for cid in c_cases.keys():
        if cid not in all_case_ids:
            all_case_ids.append(cid)

    deltas: list[ComparisonDelta] = []

    for cid in all_case_ids:
        incompatible = False
        quality_regression = False
        note_parts: list[str] = []

        if cid not in c_cases:
            deltas.append(
                ComparisonDelta(
                    case_id=cid,
                    incompatible=True,
                    regression=True,
                    note=f"Baseline case '{cid}' missing in candidate run",
                )
            )
            continue

        if cid not in b_cases:
            deltas.append(
                ComparisonDelta(
                    case_id=cid,
                    incompatible=True,
                    regression=True,
                    note=f"Candidate case '{cid}' unexpected (absent from baseline)",
                )
            )
            continue

        b_case = b_cases[cid]
        c_case = c_cases[cid]

        c_case_valid_b, err_cb = validate_case_payload_for_comparison(b_case)
        c_case_valid_c, err_cc = validate_case_payload_for_comparison(c_case)

        if not c_case_valid_b:
            incompatible = True
            note_parts.append(f"Baseline case payload invalid: {err_cb}")
        if not c_case_valid_c:
            incompatible = True
            note_parts.append(f"Candidate case payload invalid: {err_cc}")

        if incompatible:
            deltas.append(
                ComparisonDelta(
                    case_id=cid,
                    incompatible=True,
                    regression=True,
                    note="; ".join(note_parts),
                )
            )
            continue

        b_w_hash = b_case.get("workload_sha256", "")
        c_w_hash = c_case.get("workload_sha256", "")
        if b_w_hash and c_w_hash and b_w_hash != c_w_hash:
            incompatible = True
            note_parts.append(f"Workload content SHA-256 mismatch for case '{cid}'")

        b_exec = b_case.get("expected_execution", "")
        c_exec = c_case.get("expected_execution", "")
        b_sc = b_case.get("expected_shortcut_type", None)
        c_sc = c_case.get("expected_shortcut_type", None)

        if b_exec != c_exec or b_sc != c_sc:
            incompatible = True
            note_parts.append(f"Archetype mismatch: ({b_exec}, {b_sc}) vs ({c_exec}, {c_sc})")

        b_t = b_case["timing"]
        c_t = c_case["timing"]

        b_p50 = float(b_t["p50_ms"])
        c_p50 = float(c_t["p50_ms"])
        d_p50 = round(c_p50 - b_p50, 2)
        p50_pct = round((d_p50 / max(1e-4, b_p50)) * 100.0, 2)

        b_p95 = float(b_t["p95_ms"])
        c_p95 = float(c_t["p95_ms"])
        d_p95 = round(c_p95 - b_p95, 2)
        p95_pct = round((d_p95 / max(1e-4, b_p95)) * 100.0, 2)

        b_calls = b_case.get("model_calls_per_invocation")
        c_calls = c_case.get("model_calls_per_invocation")

        b_mean = float(b_case["telemetry_summary"]["model_calls"]["mean"])
        c_mean = float(c_case["telemetry_summary"]["model_calls"]["mean"])
        model_calls_mean_delta = round(c_mean - b_mean, 2)

        if b_calls is not None and c_calls is not None:
            calls_delta = c_calls - b_calls
        else:
            calls_delta = None

        psnr, ssim, mae = None, None, None
        psnr_drop, ssim_drop, mae_increase = None, None, None
        psnr_delta, ssim_delta, mae_delta = None, None, None

        if not telemetry_only:
            if not image_baseline_dir or not image_candidate_dir:
                incompatible = True
                note_parts.append("Golden comparison required but directory not specified")
            else:
                b_img_path = image_baseline_dir / cid / "output.png"
                c_img_path = image_candidate_dir / cid / "output.png"
                if not b_img_path.is_file():
                    incompatible = True
                    note_parts.append(f"Missing baseline golden image for {cid}")
                elif not c_img_path.is_file():
                    incompatible = True
                    note_parts.append(f"Missing candidate golden image for {cid}")
                else:
                    b_img = cv2.imread(str(b_img_path))
                    c_img = cv2.imread(str(c_img_path))
                    if b_img is None or c_img is None:
                        incompatible = True
                        note_parts.append("Failed to decode golden image")
                    else:
                        try:
                            cand_psnr, cand_ssim, cand_mae = compute_image_metrics(b_img, c_img)
                            psnr, ssim, mae = cand_psnr, cand_ssim, cand_mae

                            if cand_psnr < thresholds.min_psnr:
                                quality_regression = True
                                note_parts.append(f"PSNR below floor: {cand_psnr} < {thresholds.min_psnr}")
                            if cand_ssim < thresholds.min_ssim:
                                quality_regression = True
                                note_parts.append(f"SSIM below floor: {cand_ssim} < {thresholds.min_ssim}")
                            if cand_mae > thresholds.max_mae:
                                quality_regression = True
                                note_parts.append(f"MAE above ceiling: {cand_mae} > {thresholds.max_mae}")

                            base_psnr_recorded = b_case.get("psnr", 100.0) or 100.0
                            base_ssim_recorded = b_case.get("ssim", 1.0) or 1.0
                            base_mae_recorded = b_case.get("mae", 0.0) or 0.0

                            p_drop = round(base_psnr_recorded - cand_psnr, 2)
                            s_drop = round(base_ssim_recorded - cand_ssim, 4)
                            m_inc = round(cand_mae - base_mae_recorded, 4)

                            psnr_drop = p_drop
                            ssim_drop = s_drop
                            mae_increase = m_inc
                            psnr_delta = round(-p_drop, 2)
                            ssim_delta = round(-s_drop, 4)
                            mae_delta = m_inc

                            if p_drop > thresholds.max_psnr_drop:
                                quality_regression = True
                                note_parts.append(f"PSNR drop: {p_drop:.2f} dB (drop > {thresholds.max_psnr_drop})")
                            if s_drop > thresholds.max_ssim_drop:
                                quality_regression = True
                                note_parts.append(f"SSIM drop: {s_drop:.4f} (drop > {thresholds.max_ssim_drop})")
                            if m_inc > thresholds.max_mae_increase:
                                quality_regression = True
                                note_parts.append(f"MAE increase: {m_inc:.4f} (increase > {thresholds.max_mae_increase})")
                        except Exception as ex:
                            incompatible = True
                            note_parts.append(f"Image metric error: {ex}")

        is_regression = (
            incompatible
            or quality_regression
            or p50_pct > 5.0
            or (calls_delta is not None and calls_delta > 0)
            or (calls_delta is None and model_calls_mean_delta > 0.0)
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
                psnr=psnr,
                ssim=ssim,
                mae=mae,
                psnr_drop=psnr_drop,
                ssim_drop=ssim_drop,
                mae_increase=mae_increase,
                psnr_delta=psnr_delta,
                ssim_delta=ssim_delta,
                mae_delta=mae_delta,
                quality_regression=quality_regression,
                regression=is_regression,
                incompatible=incompatible,
                note="; ".join(note_parts),
            )
        )

    return deltas
