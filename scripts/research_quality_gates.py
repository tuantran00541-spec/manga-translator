#!/usr/bin/env python3
"""Research-only A/B gates for OCR, text masks, and inpainting.

Nothing in this file is imported by the production app.  Its job is to reject a
paper-derived technique before integration when real manga/manhua/webtoon data
shows a quality or safety regression.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import unicodedata
from typing import Any, Iterable

import cv2
import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.detector.stroke_refinement import refine_stroke_mask
from migan_onnx_adapter import MIGANPipelineInpainter

RESULT_PREFIX = "@@RESULT@@"
READY_PREFIX = "@@READY@@"
PROTECTED_OCR_TAGS = ("vertical", "furigana")
SUPPORTED_PPOCRV6_LANGS = {"ja", "japan", "ch", "zh", "en", "english"}


@dataclass(frozen=True)
class Sample:
    raw: dict[str, Any]
    base_dir: Path

    @property
    def id(self) -> str:
        return str(self.raw.get("id") or "")

    @property
    def task(self) -> str:
        return str(self.raw.get("task") or "").lower()

    def path(self, key: str, required: bool = True) -> Path | None:
        value = self.raw.get(key)
        if value in (None, ""):
            if required:
                raise ValueError(f"Sample {self.id!r} is missing {key!r}")
            return None
        result = Path(str(value))
        return result if result.is_absolute() else (self.base_dir / result).resolve()


def load_manifest(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"Manifest row {line_no} must be an object")
            sample = Sample(raw, path.parent)
            if not sample.id or sample.id in seen:
                raise ValueError(f"Missing/duplicate sample id at row {line_no}: {sample.id!r}")
            if sample.task not in {"ocr", "mask", "inpaint"}:
                raise ValueError(f"Sample {sample.id}: unsupported task {sample.task!r}")
            seen.add(sample.id)
            rows.append(sample)
    return rows


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    buf.tofile(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(clean) if clean else None


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def rss_mb(pid: int | None = None) -> float | None:
    try:
        proc = psutil.Process(pid) if pid else psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except (OSError, psutil.Error):
        return None


# ---------------------------------------------------------------------------
# OCR gate


def normalize_text(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text or "") if not ch.isspace())


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def text_metrics(reference: str, prediction: str) -> dict[str, Any]:
    ref, pred = normalize_text(reference), normalize_text(prediction)
    distance = edit_distance(ref, pred)
    return {
        "cer": distance / max(1, len(ref)),
        "similarity": max(0.0, 1.0 - distance / max(1, len(ref), len(pred))),
        "exact": ref == pred,
        "empty": not bool(pred),
    }


def ocr_stage(sample: Sample) -> str:
    stage = str(sample.raw.get("ocr_stage") or "pipeline").lower()
    if stage not in {"line", "pipeline"}:
        raise ValueError(f"Sample {sample.id}: ocr_stage must be line or pipeline")
    return stage


class V6Worker:
    def __init__(self, python: str, tier: str, mode: str, args: argparse.Namespace):
        cmd = [
            python,
            str(REPO_ROOT / "scripts" / "ppocrv6_worker.py"),
            "--tier", tier,
            "--mode", mode,
            "--engine", args.ppocrv6_engine,
            "--cpu-threads", str(args.cpu_threads),
        ]
        if args.ppocrv6_textline_orientation and mode == "pipeline":
            cmd.append("--use-textline-orientation")
        if args.ppocrv6_hpi:
            cmd.append("--enable-hpi")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ready = self._read(READY_PREFIX)

    def _read(self, prefix: str) -> dict[str, Any]:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"PP-OCRv6 worker exited early (code={self.proc.poll()})")
            line = line.strip()
            if line.startswith(prefix):
                return json.loads(line[len(prefix):])

    def predict(self, sample: Sample) -> dict[str, Any]:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"id": sample.id, "image_path": str(sample.path("image")), "lang": sample.raw.get("lang", "")}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        result = self._read(RESULT_PREFIX)
        result["rss_mb"] = rss_mb(self.proc.pid)
        return result

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()


def baseline_ocr(samples: list[Sample]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from app.ocr.multi_lang_ocr import MultiLangOCR

    engine = MultiLangOCR()
    names = {"line": "current-mangaocr+paddle2-line", "pipeline": "current-mangaocr+paddle2-pipeline"}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        image = read_image(sample.path("image"))
        started = time.perf_counter()
        try:
            prediction = engine.read(image, str(sample.raw.get("lang") or ""))
            error = None
        except Exception as exc:
            prediction, error = "", f"{type(exc).__name__}: {exc}"
        rows.append(ocr_row(sample, names[ocr_stage(sample)], prediction, None, (time.perf_counter() - started) * 1000.0, rss_mb(), error))
    return rows, names


def ocr_row(sample: Sample, engine: str, prediction: str, confidence: Any, latency_ms: Any, memory: Any, error: Any) -> dict[str, Any]:
    reference = str(sample.raw.get("text") or "")
    return {
        "id": sample.id,
        "engine": engine,
        "stage": ocr_stage(sample),
        "lang": str(sample.raw.get("lang") or ""),
        "tags": list(sample.raw.get("tags") or []),
        "reference": reference,
        "prediction": prediction,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "rss_mb": memory,
        "error": error,
        **text_metrics(reference, prediction),
    }


def summarize_ocr(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["engine"]].append(row)
    result: dict[str, Any] = {}
    for name, items in grouped.items():
        lat = [float(x["latency_ms"]) for x in items if x.get("latency_ms") is not None]
        memory = [float(x["rss_mb"]) for x in items if x.get("rss_mb") is not None]
        stages = sorted({x["stage"] for x in items})
        entry: dict[str, Any] = {
            "stage": stages[0] if len(stages) == 1 else stages,
            "samples": len(items),
            "mean_cer": mean(x["cer"] for x in items),
            "mean_similarity": mean(x["similarity"] for x in items),
            "exact_rate": mean(1.0 if x["exact"] else 0.0 for x in items),
            "empty_rate": mean(1.0 if x["empty"] else 0.0 for x in items),
            "error_rate": mean(1.0 if x.get("error") else 0.0 for x in items),
            "latency_first_ms": lat[0] if lat else None,
            "latency_steady_p50_ms": percentile(lat[1:] or lat, 50),
            "latency_steady_p95_ms": percentile(lat[1:] or lat, 95),
            "rss_peak_mb": max(memory) if memory else None,
            "by_tag": {},
        }
        for tag in sorted({str(tag) for x in items for tag in x.get("tags", [])}):
            subset = [x for x in items if tag in x.get("tags", [])]
            entry["by_tag"][tag] = {"samples": len(subset), "mean_cer": mean(x["cer"] for x in subset)}
        result[name] = entry
    return result


def run_ocr(samples: list[Sample], args: argparse.Namespace, output: Path) -> dict[str, Any]:
    samples = [s for s in samples if s.task == "ocr"]
    if not samples:
        return {"status": "skipped", "reason": "no OCR rows"}
    rows, baseline_names = baseline_ocr(samples)
    init: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    if args.ppocrv6_python:
        for tier in args.ppocrv6_tiers:
            for mode in args.ppocrv6_modes:
                eligible: list[Sample] = []
                for sample in samples:
                    if ocr_stage(sample) != mode:
                        continue
                    lang = str(sample.raw.get("lang") or "").lower()
                    if lang and lang not in SUPPORTED_PPOCRV6_LANGS:
                        skipped.append({"id": sample.id, "candidate": f"ppocrv6-{tier}-{mode}", "reason": f"unsupported unified-v6 language: {lang}"})
                    else:
                        eligible.append(sample)
                if not eligible:
                    continue
                name = f"ppocrv6-{tier}-{mode}-{args.ppocrv6_engine}" + ("-ori" if args.ppocrv6_textline_orientation and mode == "pipeline" else "")
                worker = None
                try:
                    worker = V6Worker(args.ppocrv6_python, tier, mode, args)
                    init[name] = worker.ready
                    for sample in eligible:
                        pred = worker.predict(sample)
                        rows.append(ocr_row(sample, name, str(pred.get("text") or ""), pred.get("confidence"), pred.get("latency_ms"), pred.get("rss_mb"), pred.get("error")))
                except Exception as exc:
                    init[name] = {"error": f"{type(exc).__name__}: {exc}"}
                finally:
                    if worker:
                        worker.close()
    summary = summarize_ocr(rows)
    gates: dict[str, Any] = {}
    for name, candidate in summary.items():
        if name in baseline_names.values():
            continue
        base_name = baseline_names.get(str(candidate.get("stage")))
        base = summary.get(base_name or "")
        reasons: list[str] = []
        if not base:
            reasons.append("matching stage baseline missing")
        else:
            if candidate["mean_cer"] > base["mean_cer"] + args.max_cer_regression:
                reasons.append("overall CER regression exceeds tolerance")
            if float(candidate.get("error_rate") or 0) > 0.01:
                reasons.append("worker error rate exceeds 1%")
            for tag in PROTECTED_OCR_TAGS:
                b, c = base["by_tag"].get(tag), candidate["by_tag"].get(tag)
                if b and c and c["mean_cer"] > b["mean_cer"] + args.protected_tag_regression:
                    reasons.append(f"{tag} CER regression exceeds protected tolerance")
        gates[name] = {"baseline": base_name, "eligible_for_next_stage": not reasons, "reasons": reasons}
    write_jsonl(output / "ocr_rows.jsonl", rows)
    return {"init": init, "summary": summary, "gates": gates, "skipped_candidate_rows": skipped}


# ---------------------------------------------------------------------------
# Mask gate


def binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred, gt = predicted > 127, truth > 127
    tp = int(np.count_nonzero(pred & gt)); fp = int(np.count_nonzero(pred & ~gt)); fn = int(np.count_nonzero(~pred & gt))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "iou": tp / max(1, tp + fp + fn),
        "false_positive_share": fp / max(1, int(np.count_nonzero(pred))),
    }


def outside_share(mask: np.ndarray, envelope: np.ndarray | None) -> float | None:
    if envelope is None:
        return None
    pred = mask > 127
    return int(np.count_nonzero(pred & ~(envelope > 127))) / max(1, int(np.count_nonzero(pred)))


def run_mask(samples: list[Sample], args: argparse.Namespace, output: Path) -> dict[str, Any]:
    from app.detector.mask_builder import adaptive_dilate_mask

    samples = [s for s in samples if s.task == "mask"]
    if not samples:
        return {"status": "skipped", "reason": "no mask rows"}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        image_path = sample.path("image", False)
        image = read_image(image_path) if image_path else None
        seed = read_image(sample.path("seed_mask"), cv2.IMREAD_GRAYSCALE)
        truth_path = sample.path("truth_mask", False); envelope_path = sample.path("safe_envelope", False)
        truth = read_image(truth_path, cv2.IMREAD_GRAYSCALE) if truth_path else None
        envelope = read_image(envelope_path, cv2.IMREAD_GRAYSCALE) if envelope_path else None
        started = time.perf_counter(); baseline = adaptive_dilate_mask(seed.copy(), image); base_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter(); candidate, stats = refine_stroke_mask(seed, image, safe_envelope=envelope, min_radius=args.stroke_min_radius, max_radius=args.stroke_max_radius); cand_ms = (time.perf_counter() - started) * 1000
        for name, mask, latency in (("current-adaptive-dilate", baseline, base_ms), ("stroke-width-refinement", candidate, cand_ms)):
            row: dict[str, Any] = {
                "id": sample.id, "variant": name, "tags": list(sample.raw.get("tags") or []),
                "source_pixels": int(np.count_nonzero(seed > 127)), "mask_pixels": int(np.count_nonzero(mask > 127)),
                "growth_ratio": int(np.count_nonzero(mask > 127)) / max(1, int(np.count_nonzero(seed > 127))),
                "outside_safe_share": outside_share(mask, envelope), "latency_ms": latency,
            }
            if truth is not None:
                row.update(binary_metrics(mask, truth))
            if name == "stroke-width-refinement":
                row.update({"components": stats.components, "max_radius_used": stats.max_radius_used, "mean_radius_used": stats.mean_radius_used})
            rows.append(row)
            if args.save_artifacts:
                write_image(output / "mask_artifacts" / f"{sample.id}__{name}.png", mask)
    summary: dict[str, Any] = {}
    for variant in ("current-adaptive-dilate", "stroke-width-refinement"):
        subset = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "samples": len(subset), "mean_f1": mean(r.get("f1") for r in subset), "mean_iou": mean(r.get("iou") for r in subset),
            "mean_recall": mean(r.get("recall") for r in subset), "mean_false_positive_share": mean(r.get("false_positive_share") for r in subset),
            "mean_outside_safe_share": mean(r.get("outside_safe_share") for r in subset), "mean_growth_ratio": mean(r.get("growth_ratio") for r in subset),
            "latency_p95_ms": percentile([float(r["latency_ms"]) for r in subset], 95),
        }
    base, cand = summary["current-adaptive-dilate"], summary["stroke-width-refinement"]
    reasons: list[str] = []
    if base["mean_f1"] is not None and cand["mean_f1"] + args.mask_f1_tolerance < base["mean_f1"]: reasons.append("mean F1 regressed")
    if base["mean_recall"] is not None and cand["mean_recall"] + args.mask_recall_tolerance < base["mean_recall"]: reasons.append("recall regressed")
    if base["mean_false_positive_share"] is not None and cand["mean_false_positive_share"] > base["mean_false_positive_share"] + args.mask_fp_tolerance: reasons.append("artwork-overreach increased")
    if cand["mean_outside_safe_share"] is not None and cand["mean_outside_safe_share"] > 0: reasons.append("safe envelope crossed")
    write_jsonl(output / "mask_rows.jsonl", rows)
    return {"summary": summary, "gate": {"eligible_for_next_stage": not reasons, "reasons": reasons}}


# ---------------------------------------------------------------------------
# Inpaint gate


def allowed_change(mask: np.ndarray, radius: int = 10) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate((mask > 127).astype(np.uint8), kernel, iterations=1) > 0


def inpaint_metrics(original: np.ndarray, result: np.ndarray, mask: np.ndarray, reference: np.ndarray | None) -> dict[str, Any]:
    allowed = allowed_change(mask)
    delta = np.max(np.abs(result.astype(np.int16) - original.astype(np.int16)), axis=2)
    outside = ~allowed
    metrics: dict[str, Any] = {"outside_allowed_change_share": int(np.count_nonzero((delta > 2) & outside)) / max(1, int(np.count_nonzero(outside)))}
    if reference is None:
        return metrics
    if reference.shape != result.shape:
        raise ValueError("reference and output shape differ")
    diff = result.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(np.square(diff[allowed]))) if np.any(allowed) else 0.0
    metrics["allowed_region_mae"] = float(np.mean(np.abs(diff[allowed]))) if np.any(allowed) else 0.0
    metrics["allowed_region_psnr"] = 99.0 if mse <= 1e-12 else 20 * math.log10(255 / math.sqrt(mse))
    ref_edges = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 64, 128, L2gradient=True) > 0
    out_edges = cv2.Canny(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 64, 128, L2gradient=True) > 0
    tp = int(np.count_nonzero(ref_edges & out_edges & allowed)); fp = int(np.count_nonzero(~ref_edges & out_edges & allowed)); fn = int(np.count_nonzero(ref_edges & ~out_edges & allowed))
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    metrics["edge_f1"] = 2 * p * r / max(1e-12, p + r)
    return metrics


def run_inpaint(samples: list[Sample], args: argparse.Namespace, output: Path) -> dict[str, Any]:
    samples = [s for s in samples if s.task == "inpaint"]
    if not samples:
        return {"status": "skipped", "reason": "no inpaint rows"}
    from app.inpaint.lama_inpainter import Inpainter
    try:
        backends: list[tuple[str, Any]] = [("current-lama", Inpainter())]
    except Exception as exc:
        return {"status": "blocked", "reason": f"LaMa init failed: {type(exc).__name__}: {exc}"}
    if args.migan_model:
        try:
            backends.append(("migan-official-onnx-pipeline", MIGANPipelineInpainter(args.migan_model)))
        except Exception as exc:
            return {"status": "blocked", "reason": f"MI-GAN contract rejected: {type(exc).__name__}: {exc}"}
    rows: list[dict[str, Any]] = []
    for name, backend in backends:
        for sample in samples:
            image = read_image(sample.path("image")); mask = read_image(sample.path("mask"), cv2.IMREAD_GRAYSCALE)
            ref_path = sample.path("reference", False); reference = read_image(ref_path) if ref_path else None
            started = time.perf_counter(); result = backend.inpaint_mask(image, mask); latency = (time.perf_counter() - started) * 1000
            rows.append({"id": sample.id, "backend": name, "tags": list(sample.raw.get("tags") or []), "latency_ms": latency, "rss_mb": rss_mb(), **inpaint_metrics(image, result, mask, reference)})
            if args.save_artifacts:
                write_image(output / "inpaint_artifacts" / f"{sample.id}__{name}.png", result)
    summary: dict[str, Any] = {}
    for name in sorted({r["backend"] for r in rows}):
        subset = [r for r in rows if r["backend"] == name]; memory = [float(r["rss_mb"]) for r in subset if r.get("rss_mb") is not None]
        summary[name] = {
            "samples": len(subset), "latency_p50_ms": percentile([float(r["latency_ms"]) for r in subset], 50), "latency_p95_ms": percentile([float(r["latency_ms"]) for r in subset], 95),
            "rss_peak_mb": max(memory) if memory else None, "mean_outside_allowed_change_share": mean(r.get("outside_allowed_change_share") for r in subset),
            "mean_allowed_region_mae": mean(r.get("allowed_region_mae") for r in subset), "mean_allowed_region_psnr": mean(r.get("allowed_region_psnr") for r in subset), "mean_edge_f1": mean(r.get("edge_f1") for r in subset),
        }
    gate: dict[str, Any] = {"status": "not_evaluated"}
    if "migan-official-onnx-pipeline" in summary:
        base, cand = summary["current-lama"], summary["migan-official-onnx-pipeline"]; reasons: list[str] = []
        if base["mean_allowed_region_mae"] is None or cand["mean_allowed_region_mae"] is None:
            reasons.append("clean reference images required for quality decision")
        elif cand["mean_allowed_region_mae"] > base["mean_allowed_region_mae"] * args.inpaint_mae_ratio:
            reasons.append("reference-region MAE regressed")
        if base["mean_edge_f1"] is not None and cand["mean_edge_f1"] is not None and cand["mean_edge_f1"] + args.inpaint_edge_tolerance < base["mean_edge_f1"]:
            reasons.append("edge continuity regressed")
        if cand["mean_outside_allowed_change_share"] > base["mean_outside_allowed_change_share"] + 1e-6:
            reasons.append("changed more artwork outside allowed region")
        gate = {"eligible_for_next_stage": not reasons, "reasons": reasons}
    write_jsonl(output / "inpaint_rows.jsonl", rows)
    return {"summary": summary, "gate": gate}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p.add_argument("--gate", choices=("all", "ocr", "mask", "inpaint"), default="all"); p.add_argument("--save-artifacts", action="store_true")
    p.add_argument("--ppocrv6-python"); p.add_argument("--ppocrv6-tiers", nargs="+", choices=("small", "medium"), default=["small", "medium"])
    p.add_argument("--ppocrv6-modes", nargs="+", choices=("line", "pipeline"), default=["pipeline"]); p.add_argument("--ppocrv6-engine", choices=("paddle_static", "onnxruntime"), default="paddle_static")
    p.add_argument("--ppocrv6-textline-orientation", action="store_true"); p.add_argument("--ppocrv6-hpi", action="store_true"); p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--max-cer-regression", type=float, default=0.01); p.add_argument("--protected-tag-regression", type=float, default=0.02)
    p.add_argument("--stroke-min-radius", type=int, default=1); p.add_argument("--stroke-max-radius", type=int, default=6)
    p.add_argument("--mask-f1-tolerance", type=float, default=0.005); p.add_argument("--mask-fp-tolerance", type=float, default=0.005); p.add_argument("--mask-recall-tolerance", type=float, default=0.02)
    p.add_argument("--migan-model", type=Path); p.add_argument("--inpaint-mae-ratio", type=float, default=1.05); p.add_argument("--inpaint-edge-tolerance", type=float, default=0.02)
    return p


def main() -> int:
    args = build_parser().parse_args(); samples = load_manifest(args.manifest.resolve()); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"manifest": str(args.manifest.resolve()), "samples": len(samples), "generated_at_unix": time.time()}
    if args.gate in {"all", "ocr"}: report["ocr"] = run_ocr(samples, args, output)
    if args.gate in {"all", "mask"}: report["mask"] = run_mask(samples, args, output)
    if args.gate in {"all", "inpaint"}: report["inpaint"] = run_inpaint(samples, args, output)
    with (output / "report.json").open("w", encoding="utf-8") as handle: json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
