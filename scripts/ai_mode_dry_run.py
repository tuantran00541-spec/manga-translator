"""Run A.I mode end to end on a real chapter with a stand-in AI (no API key).

Everything that is not an AI call is real: download, preserve regions,
skipping, cleanup, rendering, stitching and the ZIP. The stand-in AI:

- scan: marks the LAST slice as a credit slice and boxes the top of slice 0
  as a logo, so both the skip and preserve paths run;
- visual QC: has no key, so the stage records that it was skipped;
- translation: returns "Dịch thử <n>" for every text object.

Writes <out>/summary.json (job snapshot with per-stage timings) and a few
stitched pages from the ZIP as JPEGs for a visual check.

Usage:
    python scripts/ai_mode_dry_run.py <chapter_url> --out audit-results/ai-mode
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALLOWED_ROOTS = (ROOT, Path(tempfile.gettempdir()).resolve())

import app.ai_mode.job as ai_job  # noqa: E402
import app.routers.translation as translation_router  # noqa: E402
from app.ai_mode.page_scan import SliceScan  # noqa: E402
from app.ai_providers import PROVIDERS  # noqa: E402
from app.translation.vision import VisionPageTranslator, VisionTranslationResult  # noqa: E402


def fake_scan(provider, model, api_key, images):
    last = FAKE_STATE["last_index"]
    scans = []
    for index, image in images:
        height, width = image.shape[:2]
        logos = ((width // 4, 0, 3 * width // 4, max(40, height // 20)),) if index == 0 else ()
        scans.append(SliceScan(index, index == last, 0.95 if index == last else 0.0, logos, "dry run"))
    return scans, None


def fake_translate(self, original_path, cleaned_path, items, **kwargs):
    FAKE_STATE["calls"] += 1
    return VisionTranslationResult(
        {item["id"]: f"Dịch thử {n + 1}" for n, item in enumerate(items)}, self.model, {}, None,
    )


FAKE_STATE = {"last_index": -1, "calls": 0}


class DryRunRunner(ai_job.AIModeRunner):
    async def scan(self) -> None:
        FAKE_STATE["last_index"] = len(self._manifest().get("pages", [])) - 1
        await super().scan()


async def run(url: str, workers: int) -> dict:
    manager = ai_job.AIModeJobManager(runner_factory=DryRunRunner)
    settings = ai_job.AIModeSettings(url=url, provider="openai", model="dry-run", workers=workers)
    snapshot = manager.start(settings, provider=PROVIDERS["openai"], api_key="dry-run")
    job_id = snapshot["job_id"]
    last_line = ""
    while snapshot["status"] in {"pending", "running"}:
        await asyncio.sleep(2)
        snapshot = manager.snapshot(job_id)
        stage = next((s for s in snapshot["stages"] if s["key"] == snapshot["stage"]), {})
        line = f"{snapshot['stage']}: {stage.get('done')}/{stage.get('total')} {stage.get('detail') or ''}"
        if line != last_line:
            print(line, flush=True)
            last_line = line
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out = args.out.resolve()
    if not any(out.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise SystemExit(f"{args.out} must be inside {ROOT} or the temp directory")
    out.mkdir(parents=True, exist_ok=True)  # NOSONAR(S8707)

    ai_job.scan_slices = fake_scan
    VisionPageTranslator.translate_page = fake_translate
    translation_router.get_provider_api_key = lambda *a, **kw: "dry-run"

    started = time.perf_counter()
    snapshot = asyncio.run(run(args.url, args.workers))
    snapshot["wall_s"] = round(time.perf_counter() - started, 1)
    snapshot["translate_calls"] = FAKE_STATE["calls"]

    archive_path = None
    if snapshot["status"] == "completed":
        archive_path = ai_job.OUTPUT_DIR / snapshot["chapter_id"] / f"ai_mode_{snapshot['chapter_id']}.zip"
    if archive_path and archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            names = sorted(archive.namelist())
            snapshot["zip_pages"] = len(names)
            snapshot["zip_bytes"] = archive_path.stat().st_size
            for name in names[:2] + names[-1:]:
                data = np.frombuffer(archive.read(name), np.uint8)
                page = cv2.imdecode(data, cv2.IMREAD_COLOR)
                scale = 600 / page.shape[1]
                page = cv2.resize(page, (600, max(1, round(page.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                # Keep the evidence small: at most the first 6000 px of each page.
                ok, buf = cv2.imencode(".jpg", page[:6000], [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    (out / Path(name).with_suffix(".jpg").name).write_bytes(buf.tobytes())  # NOSONAR(S8707)

    (out / "summary.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")  # NOSONAR(S8707)
    print(json.dumps({k: snapshot.get(k) for k in ("status", "error", "wall_s", "zip_pages")}, ensure_ascii=False))
    for stage in snapshot["stages"]:
        print(f"  {stage['label']}: {stage['status']} {stage.get('elapsed_s')}s {stage.get('detail') or ''}")
    print("report:", json.dumps(snapshot["report"], ensure_ascii=False))
    return 0 if snapshot["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
