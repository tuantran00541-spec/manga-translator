#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import unicodedata
import urllib.request
from typing import Any

DATASET_REPO = "shigure451/Japanese_Manga"
DATASET_LICENSE = "Apache-2.0"
BASE_URL = "https://huggingface.co/datasets/shigure451/Japanese_Manga/resolve/main"

# Fixed, deterministic sample from the public dataset viewer. Keep this small
# enough for GitHub-hosted CPU runners while covering short/long and multiline
# Japanese text instances. Ground truth is copied from the dataset's text_info.
SAMPLES = [
    ("jp001", "manga_images/chunk_00001/manga_391654729baf4376a4cf266ae4759653.png", "午後から\n雨が心配"),
    ("jp002", "manga_images/chunk_00000/manga_09858a574ad7442abae0f69a45414e00.png", "うぅ〜〜、\n私が途中で\nトイレに行き\nたくなって"),
    ("jp003", "manga_images/chunk_00001/manga_2817d222f9d4418cb6c67a8827195bba.png", "その時点で"),
    ("jp004", "manga_images/chunk_00005/manga_c9a63b3efeb849279250b34994de0ca9.png", "昔はヒアルロン\n酸の目薬なども\n処方されてた"),
    ("jp005", "manga_images/chunk_00005/manga_d2c779a78e1e45b18de1cd04a7bec587.png", "丁寧とは程遠い\n雑さで腰を"),
    ("jp006", "manga_images/chunk_00004/manga_c07cc1275e7d4f1aafb97d05805028b4.png", "確かに少し動けば"),
    ("jp007", "manga_images/chunk_00005/manga_eaca8f81bce64cbc94192dcd3d3662a8.png", "「わ、いいですね。\nじゃぁ僕も今日は\nこっちでい"),
    ("jp008", "manga_images/chunk_00002/manga_6d89751ded56429aba64d2f968579ab4.png", "口から生まれたと"),
    ("jp009", "manga_images/chunk_00004/manga_ada9c8ccb991475e99faa4e3526ec66a.png", "何を言って\nいるんだ、と"),
    ("jp010", "manga_images/chunk_00002/manga_655d6df26925448797748a0bd67e0efd.png", "生娘でもある\nまいし、"),
    ("jp011", "manga_images/chunk_00000/manga_03f421dcbb584ab1aa0dfcca410f53b4.png", "と、そこまで\n考えてはたと\n気付く。こいつ\nだって"),
    ("jp012", "manga_images/chunk_00001/manga_366ab535937b4ef18ab8c3bbdf7264e1.png", "まだ何か言いたげにしていたが、"),
    ("jp013", "manga_images/chunk_00003/manga_959916d8cf4d4d55ad839d61eceea9ea.png", "なんて馬鹿なことを\n考えていると、"),
    ("jp014", "manga_images/chunk_00005/manga_d7205ba461724ef09558bc0e52d4b666.png", "ふと、\n縁淵に\n無造作に"),
    ("jp015", "manga_images/chunk_00004/manga_affb15f20a944f2fb7e710a7ed82ca52.png", "どうせあいつの\nことだから、"),
    ("jp016", "manga_images/chunk_00000/manga_1614d8a4027d4c7c834935860402b435.png", "そんなつもりで目を\n凝らして見るが、\nいくら"),
    ("jp017", "manga_images/chunk_00003/manga_854ea822371846c3880d90ac5f89eec6.png", "いや、\nもしかしたら\n思い違いかも"),
    ("jp018", "manga_images/chunk_00000/manga_22532a4d89b24ee3b94b5ec1c46ca0a6.png", "いやいやまさかね、と\n思い至った考えの\nせいで急激に"),
    ("jp019", "manga_images/chunk_00000/manga_14c92a6fd6034dc5b89ceba2e189b765.png", "当てられたのが\nタオルだと\n分かってい\nたけれど、\nそんな乱暴な\n渡し方をされて"),
    ("jp020", "manga_images/chunk_00006/manga_f0b0339ef6214eb9b0cae1f22080fa2b.png", "戦時中、敵が「桂を\n討ち取った」と"),
    ("jp021", "manga_images/chunk_00000/manga_06bf32919fc24b25982eb67118e43d5e.png", "これから\n何十年もかけて\n伝えていくから。"),
    ("jp022", "manga_images/chunk_00000/manga_157c54c8a0d34e31b4c0b97ffb957b39.png", "もちろん、潟上\n市に対応して"),
    ("jp023", "manga_images/chunk_00002/manga_68cbca5bd10946849210a12f6f27d795.png", "潟上市に\nある地元の\n法律事務所や\n法務事務所の\n門をたたき、"),
    ("jp024", "manga_images/chunk_00000/manga_25e7f96558364f938c95e3df7e66118e.png", "潟上"),
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in text if not ch.isspace())


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        value = dict(value)
    inner = value.get("res", value)
    return inner if isinstance(inner, dict) else value


def _download(sample_id: str, remote_path: str, image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(remote_path).suffix or ".png"
    dest = image_dir / f"{sample_id}{suffix}"
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    url = f"{BASE_URL}/{remote_path}?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "manga-translator-research/1.0"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            if len(data) <= 100:
                raise RuntimeError(f"Downloaded image is unexpectedly small: {len(data)} bytes")
            dest.write_bytes(data)
            return dest
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {sample_id}: {last_error}")


class PPOCRV6Pipeline:
    name = "ppocrv6-small-pipeline"

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

    def predict(self, path: Path) -> str:
        outputs = self.model.predict(input=str(path))
        texts: list[str] = []
        for item in outputs:
            data = _payload(item)
            texts.extend(str(text or "").strip() for text in (data.get("rec_texts") or []) if str(text or "").strip())
        return "".join(texts)


class MangaOCRPipeline:
    name = "manga-ocr-current"

    def __init__(self) -> None:
        from manga_ocr import MangaOcr

        started = time.perf_counter()
        self.model = MangaOcr()
        self.init_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, path: Path) -> str:
        from PIL import Image

        with Image.open(path) as image:
            return str(self.model(image.convert("RGB")) or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("ppocrv6", "mangaocr"), required=True)
    parser.add_argument("--image-dir", default="benchmark-results/japanese-ocr/images")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    paths: dict[str, Path] = {}
    for sample_id, remote_path, _gt in SAMPLES:
        paths[sample_id] = _download(sample_id, remote_path, image_dir)

    runner = PPOCRV6Pipeline() if args.engine == "ppocrv6" else MangaOCRPipeline()
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    edit_total = 0
    char_total = 0
    exact_count = 0

    for sample_id, remote_path, gt in SAMPLES:
        path = paths[sample_id]
        started = time.perf_counter()
        prediction = runner.predict(path)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        gt_norm = _normalize(gt)
        pred_norm = _normalize(prediction)
        edits = _levenshtein(gt_norm, pred_norm)
        denom = max(1, len(gt_norm))
        cer = edits / denom
        exact = pred_norm == gt_norm
        edit_total += edits
        char_total += len(gt_norm)
        exact_count += int(exact)

        row = {
            "id": sample_id,
            "dataset_path": remote_path,
            "image": str(path),
            "ground_truth": gt,
            "prediction": prediction,
            "ground_truth_normalized": gt_norm,
            "prediction_normalized": pred_norm,
            "edit_distance": edits,
            "cer": cer,
            "exact": exact,
            "latency_ms": latency_ms,
        }
        rows.append(row)
        print("@@JP_SAMPLE@@" + json.dumps({
            "id": sample_id,
            "gt": gt_norm,
            "pred": pred_norm,
            "cer": cer,
            "exact": exact,
            "latency_ms": latency_ms,
        }, ensure_ascii=False), flush=True)

    summary = {
        "engine": runner.name,
        "dataset": DATASET_REPO,
        "dataset_license": DATASET_LICENSE,
        "samples": len(rows),
        "init_ms": runner.init_ms,
        "aggregate_cer": edit_total / max(1, char_total),
        "exact_match_rate": exact_count / max(1, len(rows)),
        "exact_match_count": exact_count,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "rows": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("@@JP_SUMMARY@@" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
