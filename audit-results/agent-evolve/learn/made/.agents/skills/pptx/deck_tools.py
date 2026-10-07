"""make_deck / check_deck implementation (imported by deck_mcp_server.py and
usable standalone). Keeps SKILL.md helper recipes as the single source of truth."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / ".agents" / "skills" / "pptx"))

import helpers  # noqa: E402
from helpers import (new_prs, fade, add_slide, txbox, slide_cover, slide_agenda,  # noqa: E402
                     slide_cards, slide_twocol, slide_panels, slide_process, slide_barchart,
                     slide_quote, slide_closing, slide_diagram, slide_picture, slide_table, slide_bento, slide_timeline, slide_kpi,
                     BLUE, TEAL, MUTED, LINEC, LIGHT, NAVY)
from helpers import slide_title, source_line  # noqa: E402


def _hex_to_rgb(hexstr: str):
    from helpers import RGBColor
    h = hexstr.lstrip('#')
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def make_deck(params: dict) -> tuple[str, bool]:
    name = params.get("name")
    slides = params.get("slides")
    if not name or not isinstance(name, str):
        return "lỗi: tham số 'name' thiếu hoặc không phải chuỗi (tên file .pptx trong deck-lab/)", False
    if not slides or not isinstance(slides, list):
        return "lỗi: tham số 'slides' thiếu hoặc không phải list (cần ít nhất 1 slide, xem inputSchema)", False
    accent = _hex_to_rgb(params.get("accent", "#0B5ED7"))

    prs = new_prs()
    for i, spec in enumerate(slides):
        kind = spec.get("kind")
        try:
            if kind == "cover":
                slide_cover(prs, spec.get("title", ""), spec.get("subtitle", ""), spec.get("footer", ""))
            elif kind == "agenda":
                slide_agenda(prs, spec.get("title", "Nội dung chính"), spec.get("items", []), accent=accent)
            elif kind == "cards":
                slide_cards(prs, spec.get("title", ""), spec.get("cards", []), accent=accent)
            elif kind == "twocol":
                # spec 'columns' là dạng phẳng [[header, p1, p2], ...]; helper mong (header, [pts])
                cols_nested = [(row[0], row[1:]) for row in spec.get("columns", [])]
                slide_twocol(prs, spec.get("title", ""), cols_nested, source=spec.get("subtitle"))
            elif kind == "panels":
                cols_nested = [(row[0], row[1:]) for row in spec.get("columns", [])]
                slide_panels(prs, spec.get("title", ""), cols_nested, source=spec.get("subtitle"))
            elif kind == "process":
                slide_process(prs, spec.get("title", ""), spec.get("items", []), accent=accent)
            elif kind == "barchart":
                data = [(row[0], int(float(row[1]))) for row in spec.get("data", [])]
                slide_barchart(prs, spec.get("title", ""), data, accent=accent, source=spec.get("subtitle"))
            elif kind == "quote":
                slide_quote(prs, spec.get("lead", ""), spec.get("title", ""), spec.get("attribution", ""))
            elif kind == "closing":
                slide_closing(prs, spec.get("title", ""), spec.get("subtitle", ""))
            elif kind == "diagram":
                slide_diagram(prs, spec.get("title", ""), spec.get("top", ""),
                              spec.get("branches", []), spec.get("evidence", []),
                              source=spec.get("subtitle"), accent=accent)
            elif kind == "picture":
                slide_picture(prs, spec.get("title", ""), spec.get("caption", ""),
                              source=spec.get("subtitle"))
            elif kind == "bento":
                slide_bento(prs, spec.get("title", ""), spec.get("cells", []), accent=accent)
            elif kind == "kpi":
                slide_kpi(prs, spec.get("title", ""), spec.get("kpis", []), accent=accent, source=spec.get("subtitle"))
            elif kind == "timeline":
                events = [(e[0], e[1], e[2]) for e in spec.get("events", [])]
                slide_timeline(prs, spec.get("title", ""), events, accent=accent, source=spec.get("subtitle"))
            elif kind == "table":
                slide_table(prs, spec.get("title", ""), spec.get("header", []),
                            spec.get("rows", []), source=spec.get("subtitle"))
            else:
                return f"lỗi: slide thứ {i+1} có kind {kind!r} không hợp lệ (chọn trong cover/agenda/cards/twocol/panels/process/barchart/quote/closing/diagram/picture/table/bento/timeline)", False
        except Exception as exc:
            return f"lỗi: slide thứ {i+1} (kind={kind!r}) — {type(exc).__name__}: {exc}", False

    out_dir = ROOT / "deck-lab"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.pptx"
    prs.save(str(out_path))

    from check_deck import check as deck_check
    hard, soft, n = deck_check(str(out_path))
    msg = f"Tạo {out_path.name} ({n} slide).\n"
    if hard:
        msg += "LỖI CỨNG (phải sửa):\n" + "\n".join(" - " + h for h in hard) + "\n"
    else:
        msg += "Không có lỗi cứng theo check_deck (mục 8 SKILL.md).\n"
    if soft:
        msg += f"Cảnh báo cần render xác nhận: {len(soft)} (chưa liệt kê, chạy check_deck để xem chi tiết)."
    return msg, True


def check_deck(params: dict) -> tuple[str, bool]:
    path = params.get("path")
    if not path or not isinstance(path, str):
        return "lỗi: tham số 'path' thiếu hoặc không phải chuỗi (đường dẫn .pptx)", False
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return f"lỗi: file {path!r} không tồn tại", False
    from check_deck import check
    hard, soft, n = check(str(p))
    out = [f"{p.name} ({n} slide)"]
    if hard:
        out.append("LỖI CỨNG (phải sửa trước khi báo xong):")
        out += [f"  - {h}" for h in hard]
    else:
        out.append("LỖI CỨNG: không")
    if soft:
        out.append("CẢNH BÁO (cần render + view_image xác nhận, không loại bộ):")
        out += [f"  - {s}" for s in soft]
    else:
        out.append("CẢNH BÁO: không")
    return "\n".join(out), not hard
