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
    BenchmarkRunResult,
    CaseResult,
    ComparisonDelta,
)
from .metrics import (
    get_environment_metadata,
    get_model_metadata,
    get_model_sha256,
)
from .corpus_generator import generate_corpus, load_corpus
from .model_bench import run_model_benchmark
from .pipeline_bench import run_pipeline_benchmark_case
from .e2e_bench import run_e2e_benchmark_case


def compute_image_metrics(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float, float]:
    if img1.shape != img2.shape:
        return 0.0, 0.0, 255.0

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    mae = float(np.mean(diff))

    mse = float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20.0 * math.log10(255.0 / math.sqrt(mse))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    var1 = np.var(gray1)
    var2 = np.var(gray2)
    cov = np.mean((gray1 - mu1) * (gray2 - mu2))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)))

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
        self.model_path = Path(model_path)
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.mode = mode
        self.threads = [threads] if isinstance(threads, int) else list(threads)
        self.repetitions = repetitions
        self.warmup = warmup
        self.golden_dir = Path(golden_dir) if golden_dir else None
        self.expected_model_hash = expected_model_hash
        self.case_limit = case_limit

    def validate_runtime(self):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Model and pipeline benchmarks cannot execute.")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LaMa ONNX model not found at: {self.model_path}")
        if self.expected_model_hash:
            actual_hash = get_model_sha256(self.model_path)
            if actual_hash.lower() != self.expected_model_hash.lower():
                raise ValueError(
                    f"Model SHA-256 mismatch! Expected {self.expected_model_hash}, but found {actual_hash}"
                )

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
            for c_info in corpus_cases:
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
                    warmup=self.warmup,
                    repetitions=self.repetitions,
                )
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
                golden_path = None
                if self.golden_dir:
                    golden_path = self.golden_dir / case_id / "output.png"

                case_l3, _ = run_e2e_benchmark_case(
                    inpainter,
                    orig_img,
                    mask=mask_img,
                    case_id=case_id,
                    warmup=min(self.warmup, 2),
                    repetitions=e2e_reps,
                    save_golden_path=golden_path,
                )
                cases.append(case_l3)

        summary: dict[str, Any] = {
            "total_cases": len(cases),
            "ok_cases": sum(1 for c in cases if c.status == "ok"),
            "error_cases": sum(1 for c in cases if c.status == "error"),
        }

        return BenchmarkRunResult(
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
                        status="skipped",
                        error_message=f"Host CPU has {env_meta.logical_cpus} logical cores; {t} threads skipped.",
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
                    sub_cases = data.get("cases", [])
                    for sc in sub_cases:
                        sc["case_id"] = f"[{t}T] {sc.get('case_id', '')}"
                        all_cases.append(CaseResult(**{k: v for k, v in sc.items() if k in CaseResult.__annotations__}))
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
            mode=self.mode,
            threads=self.threads[0] if self.threads else 1,
            warmup_count=self.warmup,
            repetitions=self.repetitions,
            environment=env_meta,
            model=model_meta,
            cases=all_cases,
            summary={"sweep_threads": self.threads, "total_cases": len(all_cases)},
        )


def compare_benchmarks(
    baseline_result: BenchmarkRunResult | dict,
    candidate_result: BenchmarkRunResult | dict,
    image_baseline_dir: Path | None = None,
    image_candidate_dir: Path | None = None,
) -> list[ComparisonDelta]:
    b_dict = baseline_result if isinstance(baseline_result, dict) else baseline_result.to_dict()
    c_dict = candidate_result if isinstance(candidate_result, dict) else candidate_result.to_dict()

    b_cases = {c["case_id"]: c for c in b_dict.get("cases", [])}
    c_cases = {c["case_id"]: c for c in c_dict.get("cases", [])}

    deltas = []
    for cid, b_case in b_cases.items():
        if cid not in c_cases:
            continue
        c_case = c_cases[cid]

        b_t = b_case.get("timing", {})
        c_t = c_case.get("timing", {})

        b_p50 = float(b_t.get("p50_ms", 0.0))
        c_p50 = float(c_t.get("p50_ms", 0.0))
        d_p50 = c_p50 - b_p50
        p50_pct = (d_p50 / max(1e-4, b_p50)) * 100.0

        b_p95 = float(b_t.get("p95_ms", 0.0))
        c_p95 = float(c_t.get("p95_ms", 0.0))
        d_p95 = c_p95 - b_p95
        p95_pct = (d_p95 / max(1e-4, b_p95)) * 100.0

        b_calls = int(b_case.get("model_calls_per_invocation", b_case.get("model_calls", 0)))
        c_calls = int(c_case.get("model_calls_per_invocation", c_case.get("model_calls", 0)))
        calls_delta = c_calls - b_calls

        psnr, ssim, mae = 0.0, 0.0, 0.0
        if image_baseline_dir and image_candidate_dir:
            b_img_path = image_baseline_dir / cid / "output.png"
            c_img_path = image_candidate_dir / cid / "output.png"
            if b_img_path.is_file() and c_img_path.is_file():
                b_img = cv2.imread(str(b_img_path))
                c_img = cv2.imread(str(c_img_path))
                if b_img is not None and c_img is not None:
                    psnr, ssim, mae = compute_image_metrics(b_img, c_img)

        is_regression = p50_pct > 5.0 or calls_delta > 0

        deltas.append(
            ComparisonDelta(
                case_id=cid,
                baseline_p50_ms=round(b_p50, 2),
                candidate_p50_ms=round(c_p50, 2),
                delta_p50_ms=round(d_p50, 2),
                p50_diff_pct=round(p50_pct, 2),
                baseline_p95_ms=round(b_p95, 2),
                candidate_p95_ms=round(c_p95, 2),
                delta_p95_ms=round(d_p95, 2),
                p95_diff_pct=round(p95_pct, 2),
                baseline_model_calls=b_calls,
                candidate_model_calls=c_calls,
                model_calls_delta=calls_delta,
                psnr=psnr,
                ssim=ssim,
                mae=mae,
                regression=is_regression,
            )
        )

    return deltas
