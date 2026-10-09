"""Render Seed session evidence as PNG images (exam-rerun style).

Reads the session transcript, AUDIT.md and the audit JSONL, then renders:
  - 01_dashboard.png : session overview + capabilities + stats
  - 02_audit_pN.png  : AUDIT.md paginated as images

Usage: python render_evidence.py <transcript> <audit_md> <audit_jsonl> <out_dir>
All inputs are optional; missing files are skipped gracefully.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W = 1200
BG = (15, 18, 26)
CARD = (24, 29, 41)
BORDER = (45, 55, 75)
FG = (225, 230, 240)
DIM = (140, 150, 168)
GREEN = (110, 215, 130)
CYAN = (110, 195, 235)
YELLOW = (235, 195, 115)
PURPLE = (175, 140, 235)

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FSB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def F(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def wrap(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    text = " ".join(text.split())  # normalize all whitespace incl. newlines
    words, lines, cur = text.split(" "), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if d.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines or [""]


def parse_transcript(path: Path) -> dict:
    info = {"caps": [], "turns": 0, "final": "", "audit_n": 0, "session": ""}
    if not path.exists():
        return info
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Seed session (\S+)", text)
    if m:
        info["session"] = m.group(1)
    m = re.search(r"Capabilities defined:\s*(\[.*?\])", text)
    if m:
        try:
            info["caps"] = json.loads(m.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            pass
    m = re.search(r"Audit entries:\s*(\d+)", text)
    if m:
        info["audit_n"] = int(m.group(1))
    m = re.search(r"--- final ---\s*(.*?)(?:Capabilities defined:|\Z)", text, re.S)
    if m:
        info["final"] = m.group(1).strip()[:1200]
    return info


def count_turns(audit_path: Path) -> int:
    if not audit_path.exists():
        return 0
    n = 0
    for line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            if json.loads(line).get("event") == "assistant.text":
                n += 1
        except (json.JSONDecodeError, AttributeError):
            pass
    return n


def dashboard(info: dict, out: Path, model: str, run_id: str) -> None:
    H = 760
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_h, f_b, f_m, f_s = F(FSB, 26), F(FSB, 17), F(FS, 15), F(FR, 14), F(FS, 13)
    f_mb = F(FB, 14)

    d.text((30, 22), "Seed Backend Audit", font=f_title, fill=FG)
    d.text((30, 56), f"{model}  ·  session {info['session']}  ·  run {run_id}", font=f_s, fill=DIM)
    d.rounded_rectangle([W - 170, 24, W - 30, 58], radius=18, fill=(25, 60, 40))
    d.text((W - 148, 32), "● SUCCESS", font=f_mb, fill=GREEN)

    # capabilities card
    x, y, cw = 30, 100, 560
    d.rounded_rectangle([x, y, x + cw, y + 300], radius=12, fill=CARD, outline=BORDER, width=1)
    d.text((x + 18, y + 14), "Capabilities the model invented", font=f_h, fill=YELLOW)
    sy = y + 52
    for name in info["caps"][:8]:
        d.rounded_rectangle([x + 18, sy, x + cw - 18, sy + 44], radius=8, fill=BG, outline=BORDER)
        d.text((x + 30, sy + 12), name, font=f_mb, fill=YELLOW)
        sy += 54
    if not info["caps"]:
        d.text((x + 18, sy), "(no capabilities recorded)", font=f_s, fill=DIM)

    # stats card
    x2, cw2 = 620, W - 30 - 620
    d.rounded_rectangle([x2, y, x2 + cw2, y + 300], radius=12, fill=CARD, outline=BORDER, width=1)
    d.text((x2 + 18, y + 14), "Session stats", font=f_h, fill=GREEN)
    stats = [("Model", model), ("Capabilities defined", str(len(info["caps"]))),
             ("Audit entries", str(info["audit_n"])), ("Session", info["session"][:18])]
    sy = y + 52
    for k, v in stats:
        d.text((x2 + 18, sy), k, font=f_s, fill=DIM)
        d.text((x2 + 18, sy + 22), v, font=f_m, fill=FG)
        sy += 58

    # final summary card
    y3 = 430
    d.rounded_rectangle([30, y3, W - 30, H - 30], radius=12, fill=CARD, outline=BORDER, width=1)
    d.text((48, y3 + 14), "Model's final report (excerpt)", font=f_h, fill=CYAN)
    sy = y3 + 52
    for para in info["final"].split("\n\n")[:4]:
        for line in wrap(d, re.sub(r"\*+", "", para).strip(), f_m, W - 120)[:5]:
            if sy > H - 60:
                break
            d.text((48, sy), line, font=f_m, fill=FG)
            sy += 24
        sy += 10
    img.save(out)
    print("saved", out)


def audit_pages(audit_md: Path, out_dir: Path) -> int:
    if not audit_md.exists():
        print("no AUDIT.md, skipping pages")
        return 0
    raw = audit_md.read_text(encoding="utf-8", errors="replace")
    # strip markdown noise lightly
    text = re.sub(r"#{1,6}\s*", "", raw)
    text = re.sub(r"\*+", "", text)
    d = ImageDraw.Draw(Image.new("RGB", (W, 10), BG))
    f_h, f_m, f_s = F(FSB, 20), F(FR, 15), F(FS, 13)
    lines: list[tuple[str, object, tuple]] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append(("", f_m, FG))
            continue
        is_head = len(para) < 90 and (para.isupper() or para.endswith(":") or re.match(r"^\d+\.", para))
        font, col = (f_h, CYAN) if is_head else (f_m, FG)
        for wl in wrap(d, para, font, W - 120):
            lines.append((wl, font, col))
    per_page, n = 52, 0
    for i in range(0, len(lines), per_page):
        chunk = lines[i:i + per_page]
        H = 90 + 30 * len(chunk)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 24), f"AUDIT.md  —  page {i // per_page + 1}", font=f_s, fill=DIM)
        d.line([40, 58, W - 40, 58], fill=BORDER, width=1)
        sy = 76
        for wl, font, col in chunk:
            if wl:
                d.text((60, sy), wl, font=font, fill=col)
            sy += 30
        p = out_dir / f"02_audit_p{i // per_page + 1}.png"
        img.save(p)
        n += 1
        print("saved", p)
    return n


def _p(s: str | None, default: str) -> Path:
    return Path(s) if s else Path(default)


def main() -> int:
    args = sys.argv[1:]
    transcript = _p(args[0] if len(args) > 0 else "", "/tmp/seed-run.log")
    audit_md = _p(args[1] if len(args) > 1 else "", "/tmp/seed-ws/AUDIT.md")
    audit_jsonl_arg = args[2] if len(args) > 2 else ""
    out_dir = _p(args[3] if len(args) > 3 else "", "/tmp/seed-evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = sys.argv[5] if len(sys.argv) > 5 else "agnes-3.0-flash"
    run_id = sys.argv[6] if len(sys.argv) > 6 else "local"

    info = parse_transcript(transcript)
    # find newest audit jsonl
    audit_jsonl: Path | None = _p(audit_jsonl_arg, "") if audit_jsonl_arg else None
    seed_dir = Path.home() / ".seed"
    if seed_dir.exists():
        cands = sorted(seed_dir.glob("audit-*.jsonl"), key=lambda p: p.stat().st_mtime)
        if cands:
            audit_jsonl = cands[-1]
    if audit_jsonl is None:
        audit_jsonl = seed_dir / "audit.jsonl"
    info["turns"] = count_turns(audit_jsonl)
    if not info["audit_n"] and audit_jsonl.exists():
        info["audit_n"] = sum(1 for _ in audit_jsonl.read_text(encoding="utf-8", errors="replace").splitlines() if _.strip())

    dashboard(info, out_dir / "01_dashboard.png", model, run_id)
    audit_pages(audit_md, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
