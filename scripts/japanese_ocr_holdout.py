#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from japanese_ocr_benchmark import (
    DATASET_LICENSE,
    DATASET_REPO,
    MangaOCRPipeline,
    _download,
    _levenshtein,
    _normalize,
    _payload,
    _percentile,
)
from japanese_ppocr_order_probe import _content_normalize, _reconstruct

# Held out from the 24-sample development set used while designing the
# geometry/ruby heuristic. These rows are fixed before this validation run.
HOLDOUT_SAMPLES = [
    ("jh001", "manga_images/chunk_00004/manga_c5b1527a5ef94ab28c23c9b2ecca31d5.png", "ただ、"),
    ("jh002", "manga_images/chunk_00005/manga_db3dd7567c1c49fc824dbece20a12589.png", "前々から走ってみ"),
    ("jh003", "manga_images/chunk_00001/manga_4ecf9967a2274cb49075ee309fa6a049.png", "彼らには"),
    ("jh004", "manga_images/chunk_00001/manga_45a932766bb64edb8899c5322688de37.png", "全然苦に\nならず、"),
    ("jh005", "manga_images/chunk_00006/manga_fd7d1b1d595d405b96c44dea8c7543d7.png", "これは加齢に伴い増えて\nいき、背骨の可動\n域が狭まって"),
    ("jh006", "manga_images/chunk_00004/manga_a89f2aeb049b4ff5986f4bc9ca06b997.png", "銀時に\nとって\nその十\n年は、\n長いようで"),
    ("jh007", "manga_images/chunk_00003/manga_85eb060b109b45a9bc4ce73e00d5559c.png", "道を分かつまでの\n二十年弱は"),
    ("jh008", "manga_images/chunk_00003/manga_88a8441ea89f4a82a5a1f4c93fbfec83.png", "こいつは\n昔から\nそうだ。\n出会った\n当初もいつの\n間にか\n道場に\n上がり"),
    ("jh009", "manga_images/chunk_00003/manga_9d666b3981ee4daa9e7d00b4cafa8729.png", "いや、"),
    ("jh010", "manga_images/chunk_00005/manga_d8b2c74a8bfb4fcdaaf82404fee9a713.png", "どうして、と"),
    ("jh011", "manga_images/chunk_00000/manga_0021ef88477a4cb894dc3434d055e9ae.png", "成人"),
    ("jh012", "manga_images/chunk_00000/manga_0eff2cb6a2104e78b8c314ccef88ee87.png", "皆まで"),
    ("jh013", "manga_images/chunk_00004/manga_bf852b942465401ead883b3e6ebe9d68.png", "そんな"),
    ("jh014", "manga_images/chunk_00000/manga_0843d21c7c8f4b8fadb312e83711b2f8.png", "見た"),
    ("jh015", "manga_images/chunk_00001/manga_2fdc1962e59c4ed299ee71ac772b2dc6.png", "冗談で"),
    ("jh016", "manga_images/chunk_00005/manga_eb12f48499c347158186a20c07ec7c8d.png", "こっちの"),
    ("jh017", "manga_images/chunk_00002/manga_6be94897dc864596a58afaf6a7d0d1d4.png", "本当に失ってしまったの\nかもしれないと"),
    ("jh018", "manga_images/chunk_00000/manga_066e37fd56fc49c29d640b98d1a344b3.png", "何を\n考えて"),
    ("jh019", "manga_images/chunk_00000/manga_244b10be8b824bd7bfc17f9695273390.png", "掴んだ腰は、"),
    ("jh020", "manga_images/chunk_00000/manga_117f82907a6948198fbcac3a6264aa40.png", "銀時を"),
    ("jh021", "manga_images/chunk_00006/manga_fa89c173c2104d0dbcbcb8e011ef5595.png", "きっと、"),
    ("jh022", "manga_images/chunk_00002/manga_4ff498b89af0438a91ab8371edcbc777.png", "触れたらきっと\n最後だと、ずっと\n耐えてきたの"),
    ("jh023", "manga_images/chunk_00003/manga_76e2fed7d37f4092b6c2c4e02a35c6cc.png", "俺の気持ちが\n分かって"),
    ("jh024", "manga_images/chunk_00004/manga_b0658bd1d5aa4628ac2c4b219b996bac.png", "いきなり"),
    ("jh025", "manga_images/chunk_00004/manga_a9cf5c2e28ff47acb396e285d728b8af.png", "だが同時に、\nその世界は、"),
    ("jh026", "manga_images/chunk_00003/manga_7cc3da928a23424c833c4dafe604ae7d.png", "私の幼少期は、\n自分自身が周囲から\n受けた直截の\n経験から"),
    ("jh027", "manga_images/chunk_00002/manga_7420ec5310ef4e308626a83cbbccba79.png", "十代の終わりに\nこの町を離れてから"),
    ("jh028", "manga_images/chunk_00003/manga_7e909fdac82046cba656de1be1889ae6.png", "父親に言わせれば\n「俺たちには\n死ね、と言って\nいるのと同じ」"),
    ("jh029", "manga_images/chunk_00003/manga_91e593785ffe430ab8127c93de7b42c2.png", "当の「実家」に"),
    ("jh030", "manga_images/chunk_00006/manga_eed7188336d544ce8d0397705b502047.png", "だが、最大の\n変化は、生涯の"),
    ("jh031", "manga_images/chunk_00005/manga_d2c2f6190cbc4efa8ebc0b932693916d.png", "当然、"),
    ("jh032", "manga_images/chunk_00004/manga_b75dd945ed01488b9b866f3fdc64475d.png", "しかしそれ以上に"),
]


class PPOCRGeometryPipeline:
    name = "ppocrv6-small-geometry-v2"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        started = time.perf_counter()
        self.model = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="cpu",
        )
        self.init_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, path: Path) -> tuple[str, dict[str, Any]]:
        outputs = self.model.predict(input=str(path))
        texts: list[str] = []
        scores: list[Any] = []
        polygons: list[Any] = []
        for result in outputs:
            data = _payload(result)
            item_texts = list(data.get("rec_texts") or [])
            item_scores = list(data.get("rec_scores") or [])
            item_polygons = data.get("rec_polys")
            if item_polygons is None or len(item_polygons) == 0:
                item_polygons = data.get("dt_polys") or []
            item_polygons = list(item_polygons)
            n = min(len(item_texts), len(item_polygons))
            texts.extend(str(item_texts[i] or "").strip() for i in range(n))
            scores.extend(item_scores[i] if i < len(item_scores) else None for i in range(n))
            polygons.extend(item_polygons[:n])
        geometry = _reconstruct(texts, scores, polygons)
        return str(geometry["prediction"]), geometry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("ppocrv6", "mangaocr"), required=True)
    parser.add_argument("--image-dir", default="benchmark-results/japanese-holdout/images")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    paths: dict[str, Path] = {}
    for sample_id, remote_path, _gt in HOLDOUT_SAMPLES:
        paths[sample_id] = _download(sample_id, remote_path, image_dir)

    runner: Any
    if args.engine == "ppocrv6":
        runner = PPOCRGeometryPipeline()
    else:
        runner = MangaOCRPipeline()

    rows: list[dict[str, Any]] = []
    strict_edits = 0
    strict_chars = 0
    strict_exact = 0
    content_edits = 0
    content_chars = 0
    content_exact = 0
    latencies: list[float] = []

    for sample_id, remote_path, gt in HOLDOUT_SAMPLES:
        path = paths[sample_id]
        started = time.perf_counter()
        if args.engine == "ppocrv6":
            prediction, geometry = runner.predict(path)
        else:
            prediction = runner.predict(path)
            geometry = None
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        gt_norm = _normalize(gt)
        pred_norm = _normalize(prediction)
        strict_distance = _levenshtein(gt_norm, pred_norm)
        strict_cer = strict_distance / max(1, len(gt_norm))
        strict_edits += strict_distance
        strict_chars += len(gt_norm)
        strict_exact += int(gt_norm == pred_norm)

        content_gt = _content_normalize(gt)
        content_pred = _content_normalize(prediction)
        content_distance = _levenshtein(content_gt, content_pred)
        content_cer = content_distance / max(1, len(content_gt))
        content_edits += content_distance
        content_chars += len(content_gt)
        content_exact += int(content_gt == content_pred)

        row = {
            "id": sample_id,
            "dataset_path": remote_path,
            "image": str(path),
            "ground_truth": gt,
            "prediction": prediction,
            "ground_truth_normalized": gt_norm,
            "prediction_normalized": pred_norm,
            "strict_cer": strict_cer,
            "content_ground_truth": content_gt,
            "content_prediction": content_pred,
            "content_cer": content_cer,
            "latency_ms": latency_ms,
        }
        if geometry is not None:
            row.update({
                "orientation": geometry["orientation"],
                "removed_indices": geometry["removed_indices"],
                "ordered_indices": geometry["ordered_indices"],
                "regions": geometry["regions"],
            })
        rows.append(row)
        print("@@JP_HOLDOUT_SAMPLE@@" + json.dumps({
            "engine": runner.name,
            "id": sample_id,
            "gt": gt_norm,
            "pred": pred_norm,
            "strict_cer": strict_cer,
            "content_cer": content_cer,
            "orientation": None if geometry is None else geometry["orientation"],
            "removed": None if geometry is None else geometry["removed_indices"],
            "latency_ms": latency_ms,
        }, ensure_ascii=False), flush=True)

    summary = {
        "engine": runner.name,
        "dataset": DATASET_REPO,
        "dataset_license": DATASET_LICENSE,
        "sample_set": "held-out-after-geometry-v2-frozen",
        "samples": len(rows),
        "init_ms": runner.init_ms,
        "strict_aggregate_cer": strict_edits / max(1, strict_chars),
        "strict_exact_match_rate": strict_exact / max(1, len(rows)),
        "content_aggregate_cer": content_edits / max(1, content_chars),
        "content_exact_match_rate": content_exact / max(1, len(rows)),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("@@JP_HOLDOUT_SUMMARY@@" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
