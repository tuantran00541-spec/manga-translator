"""One-off: the same scan prompt on the same slices through several vision models."""
import base64, json, os, shutil, sys, time
from pathlib import Path

import cv2, requests

sys.path.insert(0, os.getcwd())
from app.ai_mode.page_scan import SCAN_PROMPT, _thumbnail, parse_scan
from app.downloader.registry import download_chapter
from app.downloader.slicer import slice_image

url, out, models, count = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(), int(sys.argv[4])
raw, cut = out / "_raw", out / "_cut"
raw.mkdir(parents=True, exist_ok=True); cut.mkdir(parents=True, exist_ok=True)
slices = []
for n, page in enumerate(download_chapter(url, raw)):
    slices += sorted(slice_image(page, cut, f"p{n:03d}"))
step = max(1, len(slices) // count)
chosen = slices[step // 2::step][:count]
(out / "slices").mkdir(exist_ok=True)
rows = []
for k, path in enumerate(chosen, 1):
    image = cv2.imread(str(path))
    thumb = _thumbnail(image)
    cv2.imwrite(str(out / "slices" / f"{k:02d}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = base64.b64encode(cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()
    for model in models:
        body = {"model": model, "max_tokens": 6000, "reasoning_effort": "none", "messages": [{"role": "user", "content": [
            {"type": "text", "text": SCAN_PROMPT}, {"type": "text", "text": f"SLICE {k}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}]}]}
        started = time.time()
        try:
            r = requests.post("https://api.minirouter.sh/v1/chat/completions", json=body, timeout=(10, 180),
                              headers={"Authorization": "Bearer " + os.environ["MINIROUTER_API_KEY"]})
            js = r.json() if r.ok else {}
            text = ((js.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            usage, status, error = js.get("usage") or {}, r.status_code, "" if r.ok else r.text[:300]
        except requests.RequestException as exc:
            text, usage, status, error = "", {}, 0, str(exc)[:300]
        try:
            parsed = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
            scan = parse_scan(parsed, {k: (image.shape[1], image.shape[0])})
            lines = [list(line) for s in scan for line in s.lines]
            no_text = any(s.no_text for s in scan)
        except (ValueError, AttributeError, TypeError):
            parsed, lines, no_text = None, None, None
        rows.append({"slice": k, "model": model, "status": status, "seconds": round(time.time() - started, 1),
                     "usage": usage, "valid_json": parsed is not None, "no_text": no_text, "lines": lines,
                     "raw": text[:4000], "error": error})
        print(k, model, status, rows[-1]["seconds"], "lines", None if lines is None else len(lines), flush=True)
(out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
shutil.rmtree(raw); shutil.rmtree(cut)
