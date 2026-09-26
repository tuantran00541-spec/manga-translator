from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.translation.context import ChapterMemory, text_system_prompt  # noqa: E402
from app.translation.deepseek import _language_name  # noqa: E402
from app.translation.vision import parse_vision_translation  # noqa: E402

CHUNK = 60
CONTEXT_DONE = 40
MAX_TOKENS = 8192


def confined(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise SystemExit(f"{path} must be inside {ROOT}")
    return resolved


def ocr(url: str, out: Path, workers: int) -> None:
    from app.manifest_utils import load_manifest_raw
    from app.ocr.multi_lang_ocr import MultiLangOCR
    from app.ocr.service import OCRService
    from app.processing_pipeline_factory import build_processing_pipeline
    from chapter_e2e_ocr_render_gate import _seed_text_objects

    pipeline = build_processing_pipeline()
    chapter_id = hashlib.sha256(f"ai-text:{url}:{time.time_ns()}".encode()).hexdigest()[:8]
    started = time.perf_counter()
    manifest = pipeline.download_chapter(url, chapter_id, workers=workers)
    pipeline.process_pages(chapter_id, list(range(len(manifest["pages"]))), workers=workers)
    _seed_text_objects(pipeline, chapter_id)
    service = OCRService(MultiLangOCR(), pipeline)
    targets = service.plan_chapter(chapter_id)
    errors = []
    for page_index, box_id in targets:
        try:
            service.inspect_box_id(chapter_id, page_index, box_id, "en", force=True)
        except Exception as exc:
            errors.append({"slice": page_index + 1, "box": box_id, "error": f"{type(exc).__name__}: {exc}"[:200]})
    lines = []
    for index, page in enumerate(load_manifest_raw(chapter_id)["pages"]):
        boxes = [box for box in page.get("boxes") or [] if isinstance(box, dict) and not box.get("removed") and str(box.get("ocr_text") or "").strip()]
        boxes.sort(key=lambda box: (int(box.get("y1") or 0), int(box.get("x1") or 0)))
        for box in boxes:
            lines.append({"id": f"s{index + 1}-{box.get('id')}", "slice": index + 1, "text": str(box["ocr_text"]).strip()})
    out.mkdir(parents=True, exist_ok=True)
    (out / "ocr.json").write_text(json.dumps({
        "url": url, "slices": len(manifest["pages"]), "targets": len(targets), "lines": lines,
        "errors": errors, "seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OCR: {len(lines)} lines from {len(manifest['pages'])} slices, {len(errors)} errors")


VI_ADDRESS_WORDS = frozenset("""
tôi ta tao mình tớ em anh chị cậu mày ngươi con cha bố mẹ má ba ông bà cô chú bác thầy ngài người nó hắn
bọn chúng bạn thần bệ hạ thuộc hạ huynh đệ tỷ muội sư phụ đồ nhi lão tiểu nhóc cháu dì dượng thím mợ
nàng chàng quý tộc điện chủ nhân thiếu gia tiểu thư đại nhân các vị ấy kia này
""".split())

TUNED_RULES = """
TUNING
- Do not insert line breaks; return each translation as one line. The renderer wraps text to the bubble.
- Write normal Vietnamese sentence case even when the source is ALL CAPS. Keep capitals only for SFX, system window titles and names.
- If you cannot tell who is speaking, infer it from the surrounding lines before choosing pronouns; a parent speaks to a child as ta/con or cha/con, never tôi.
- Spell Vietnamese carefully: every word must be a real Vietnamese word with correct diacritics.
""".strip()


def _valid_address(term: str) -> bool:
    words = term.lower().split()
    return bool(words) and all(word in VI_ADDRESS_WORDS for word in words)


def _ask(base: str, key: str, model: str, system: str, user: str, temperature: float | None = None) -> tuple[str, dict, float]:
    started = time.perf_counter()
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS, "stream": False,
               **({"temperature": temperature} if temperature is not None else {})}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{base.rstrip('/')}/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=(10, 600))
    if response.status_code in (400, 422) and "response_format" in response.text:
        payload.pop("response_format")
        response = requests.post(url, headers=headers, json=payload, timeout=(10, 600))
    elapsed = time.perf_counter() - started
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    choice = body["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    if not content.strip():
        raise RuntimeError(f"empty reply (finish_reason={choice.get('finish_reason')})")
    return content, body.get("usage") or {}, elapsed


def _line(item: dict) -> str:
    return f"[{item['id']}] (slice {item['slice']}) {item['text']}"


def translate(out: Path, target_lang: str, notes: str, tag: str = "", tuned: bool = False,
              temperature: float | None = None) -> None:
    base, key, model = (os.environ[name] for name in ("GATEWAY_UPSTREAM_BASE", "GATEWAY_UPSTREAM_KEY", "GATEWAY_UPSTREAM_MODEL"))
    data = json.loads((out / "ocr.json").read_text(encoding="utf-8"))
    lines = data["lines"]
    system = text_system_prompt(_language_name(target_lang), target_lang)
    if tuned:
        system = "\n".join(line for line in system.splitlines() if not line.startswith("- Break lines yourself"))
        system = system.replace("<text, lines split with \\n>", "<text on one line>") + "\n\n" + TUNED_RULES
    chapter = "\n".join(_line(item) for item in lines)
    memory = ChapterMemory(notes)
    done: dict[str, dict] = {}
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    calls, errors = [], []
    started = time.perf_counter()

    def run(chunk: list[dict], label: str) -> None:
        ids = {item["id"] for item in chunk}
        recent = [{"id": i, "source": v["source"], "vi": v["translated_text"], "speaker": v.get("speaker")}
                  for i, v in list(done.items())[-CONTEXT_DONE:]]
        sheet = memory.snapshot()
        user = (
            f"STORY NOTES: {notes or '(none)'}\n\n"
            f"FULL CHAPTER (read-only context):\n{chapter}\n\n"
            f"CHARACTER SHEET AND FORMS OF ADDRESS (read-only): "
            f"{json.dumps({'characters': sheet['characters'], 'address': sheet['address']}, ensure_ascii=False)}\n\n"
            f"ALREADY TRANSLATED (read-only): {json.dumps(recent, ensure_ascii=False)}\n\n"
            f"TRANSLATE NOW:\n" + "\n".join(_line(item) for item in chunk)
            + '\n\nAnswer with one JSON object that starts with {"translations":[ and contains every id under TRANSLATE NOW.'
        )
        try:
            content, usage, elapsed = _ask(base, key, model, system, user, temperature)
            translations = parse_vision_translation(content, ids, allow_missing=True)
            reply = json.loads(content.strip().removeprefix("```json").removesuffix("```"))
        except Exception as exc:
            errors.append({"chunk": label, "error": str(exc)[:400]})
            calls.append({"chunk": label, "ok": False})
            return
        for name in usage_total:
            usage_total[name] += int(usage.get(name) or 0)
        extra = {e["id"]: e for e in reply.get("translations") or [] if isinstance(e, dict) and e.get("id") in ids}
        speakers = {i: str(e.get("speaker") or "") for i, e in extra.items()}
        if tuned:
            reply["address"] = [
                {k: v for k, v in entry.items() if k not in ("self", "other") or _valid_address(str(v))}
                for entry in reply.get("address") or [] if isinstance(entry, dict)
            ]
        memory.update(0, {**reply, "speakers": speakers}, translations, [item["id"] for item in chunk])
        for item in chunk:
            value = translations.get(item["id"], "")
            if value:
                done[item["id"]] = {"source": item["text"], "translated_text": value,
                                    "role": extra.get(item["id"], {}).get("role"),
                                    "speaker": speakers.get(item["id"]),
                                    "review": bool(extra.get(item["id"], {}).get("review"))}
        calls.append({"chunk": label, "ok": True, "seconds": round(elapsed, 1), "usage": usage,
                      "missing": sorted(ids - {i for i in ids if translations.get(i)})})

    for start in range(0, len(lines), CHUNK):
        run(lines[start:start + CHUNK], f"{start + 1}-{min(start + CHUNK, len(lines))}")
    missing = [item for item in lines if item["id"] not in done]
    if missing:
        run(missing, "retry-missing")

    sheet = memory.snapshot()
    report = {
        "model": model, "target_lang": target_lang, "lines": len(lines),
        "translated": sum(1 for item in lines if item["id"] in done),
        "empty_or_missing": [item["id"] for item in lines if item["id"] not in done],
        "seconds": round(time.perf_counter() - started, 1), "usage": usage_total,
        "calls": calls, "errors": errors,
        "characters": sheet["characters"], "address": sheet["address"],
        "table": [{"id": item["id"], "slice": item["slice"], "source": item["text"],
                   **{k: done.get(item["id"], {}).get(k) for k in ("translated_text", "role", "speaker", "review")}}
                  for item in lines],
    }
    suffix = f"-{tag}" if tag else ""
    report["variant"] = {"tag": tag, "tuned": tuned, "temperature": temperature}
    (out / f"translation{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = ["| # | Lát | Gốc | Dịch | Vai | Người nói |", "|---|---|---|---|---|---|"]
    for n, row in enumerate(report["table"], start=1):
        cells = [str(n), str(row["slice"]), row["source"], row.get("translated_text") or "—", row.get("role") or "", row.get("speaker") or ""]
        rows.append("| " + " | ".join(c.replace("|", "/").replace("\n", " / ") for c in cells) + " |")
    (out / f"translation{suffix}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("lines", "translated", "seconds", "usage", "errors")}, ensure_ascii=False, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("ocr")
    o.add_argument("url")
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--workers", type=int, default=2)
    t = sub.add_parser("translate")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--target", default="vi")
    t.add_argument("--notes", default="")
    t.add_argument("--tag", default="")
    t.add_argument("--tuned", action="store_true")
    t.add_argument("--temperature", type=float)
    args = parser.parse_args()
    if args.command == "ocr":
        ocr(args.url, confined(args.out), args.workers)
    else:
        translate(confined(args.out), args.target, args.notes, args.tag, args.tuned, args.temperature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
