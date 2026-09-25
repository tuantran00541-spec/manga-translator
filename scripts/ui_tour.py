from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, expect, sync_playwright

DESKTOP = {"width": 1440, "height": 960}
MOBILE = {"width": 390, "height": 844}


class Tour:
    def __init__(self, base_url: str, out: Path) -> None:
        self.base = base_url.rstrip("/")
        self.out = out
        self.shots: list[dict] = []
        self.checks: dict[str, object] = {}
        self.errors: list[str] = []
        self.step = 0

    def shot(self, page: Page, name: str, note: str, *, full: bool = False) -> None:
        self.step += 1
        path = self.out / f"{self.step:02d}-{name}.png"
        page.wait_for_timeout(400)
        page.screenshot(path=str(path), full_page=full)
        self.shots.append({"file": path.name, "note": note})
        print(f"[shot] {path.name}: {note}", flush=True)

    def api(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())

    def image(self, url: str) -> np.ndarray:
        with urllib.request.urlopen(self.base + url.split("?")[0] + f"?t={time.time()}", timeout=120) as response:
            return np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB")).astype(np.int16)

    def manifest(self, chapter_id: str) -> dict:
        return self.api(f"/api/chapter/{chapter_id}")


def wait_review(page: Page, timeout: float = 120_000) -> None:
    page.wait_for_selector(".review-workspace-shell.review-single-document", timeout=timeout)
    page.wait_for_function(
        "() => document.querySelector('.review-stitched-image .review-strip-slice img')",
        timeout=timeout,
    )
    page.wait_for_timeout(800)


def landing(tour: Tour, page: Page) -> None:
    page.goto(tour.base, wait_until="networkidle")
    expect(page.locator("#home-view")).to_be_visible()
    tour.shot(page, "home", "Trang chủ: lời chào, chương gần đây")
    page.locator("#start-action").click()
    expect(page.locator("#import-view")).to_be_visible()
    tour.shot(page, "import", "Nhập chương: link, tải ảnh lên, A.I mode", full=True)
    page.locator("#settings-toggle").click()
    expect(page.locator("#settings-drawer")).to_be_visible()
    page.wait_for_timeout(800)
    tour.shot(page, "settings", "Cài đặt: dịch vụ và model AI")
    first_row = page.locator("#settings-drawer .ai-provider-row, #settings-drawer details, #settings-drawer [aria-expanded]").first
    if first_row.count():
        first_row.click()
        page.wait_for_timeout(400)
        tour.shot(page, "settings-provider", "Cài đặt: mở một provider")
    page.locator("#settings-close").click()
    page.locator('.sidebar-link[data-route="home"]').click()
    page.locator("#theme-select").select_option("dark")
    tour.shot(page, "home-dark", "Giao diện tối")
    page.locator("#theme-select").select_option("light")


def import_chapter(tour: Tour, page: Page, url: str) -> str:
    page.locator("#start-action").click()
    page.locator("#chapter-url").fill(url)
    tour.shot(page, "import-url", "Dán link chương")
    page.locator("#load-btn").click()
    page.wait_for_selector("body[data-app-stage='preview']", timeout=600_000)
    page.wait_for_selector(".preview-canvas-surface img", timeout=120_000)
    chapter_id = page.evaluate("() => window.currentChapterId")
    tour.checks["chapter_id"] = chapter_id
    return chapter_id


def open_preview(tour: Tour, page: Page, chapter_id: str) -> None:
    page.goto(f"{tour.base}/#{chapter_id}", wait_until="networkidle")
    page.reload(wait_until="networkidle")
    wait_review(page)
    goto_preview(page)


def goto_preview(page: Page) -> None:
    link = page.locator('.sidebar-link[data-stage="preview"]')
    expect(link).to_be_enabled(timeout=60_000)
    link.click()
    page.wait_for_selector(".preview-canvas-surface img", timeout=60_000)


def select_preview_page(page: Page, index: int) -> None:
    page.evaluate("(i) => { previewActivePageIndex = i; renderPreview(); }", index)
    page.wait_for_selector(f'.preview-canvas-surface[data-page-index="{index}"] img')
    page.wait_for_function(
        "() => { const i = document.querySelector('.preview-canvas-surface img'); return i && i.complete && i.naturalWidth > 0; }"
    )
    page.wait_for_timeout(300)




def skip_slices(tour: Tour, page: Page, chapter_id: str, keep: list[int]) -> None:
    total = len(tour.manifest(chapter_id)["pages"])
    tour.shot(page, "preview", f"Xem trước: {total} lát, trước khi chọn")
    ui_skips = [i for i in range(total) if i not in keep][:2]
    for index in ui_skips:
        select_preview_page(page, index)
        page.locator(".skip-btn").click()
        expect(page.locator(".skip-btn")).to_have_text("Đã bỏ qua · Chọn để khôi phục")
    if ui_skips:
        tour.shot(page, "preview-skipped", f"Bỏ qua lát ảnh bằng nút (lát {', '.join(str(i + 1) for i in ui_skips)})")
    rest = [i for i in range(total) if i not in keep and i not in ui_skips]
    if rest:
        tour.api("/api/skip_pages", {"chapter_id": chapter_id, "page_indices": rest, "skipped": True})
        page.reload(wait_until="networkidle")
        goto_preview(page)
    manifest = tour.manifest(chapter_id)
    skipped = [i for i, p in enumerate(manifest["pages"]) if p.get("skipped")]
    tour.checks["skipped_pages"] = len(skipped)
    tour.checks["kept_pages"] = [i for i in range(total) if i not in skipped]
    select_preview_page(page, keep[0])
    tour.shot(page, "preview-selected", f"{total - len(skipped)}/{total} lát được chọn để xử lý")


def draw_preserve(tour: Tour, page: Page, index: int, target: list[int]) -> None:
    select_preview_page(page, index)
    page.locator(".excluded-toggle-btn").click()
    img = page.locator(".preview-canvas-surface .preview-image-wrap img")
    natural_w = img.evaluate("i => i.naturalWidth")
    tx1, ty1, tx2, ty2 = target
    scale = img.bounding_box()["width"] / natural_w
    img.evaluate(
        """(i, y) => {
          const marker = document.createElement('div');
          marker.style.cssText = `position:absolute;left:0;top:${y}px;width:1px;height:1px`;
          i.parentElement.appendChild(marker);
          marker.scrollIntoView({block: 'center'});
          marker.remove();
        }""",
        (ty1 + ty2) / 2 * scale,
    )
    page.wait_for_timeout(400)
    img_box = img.bounding_box()
    x1, y1 = img_box["x"] + tx1 * scale, img_box["y"] + ty1 * scale
    x2, y2 = img_box["x"] + tx2 * scale, img_box["y"] + ty2 * scale
    page.mouse.move(x1, y1)
    page.mouse.down()
    page.mouse.move((x1 + x2) / 2, (y1 + y2) / 2, steps=6)
    page.mouse.move(x2, y2, steps=6)
    page.mouse.up()
    page.locator(".excluded-toggle-btn").click()
    page.wait_for_timeout(800)
    page.evaluate("() => window.flushPreserveRegionSaves?.(window.currentChapterId)")


def process(tour: Tour, page: Page, name: str, note: str) -> float:
    page.locator(".preview-primary-action").first.click()
    page.wait_for_timeout(1500)
    tour.shot(page, name, note)
    started = time.time()
    page.wait_for_selector("body[data-app-stage='review']", timeout=1_800_000)
    wait_review(page, timeout=600_000)
    return round(time.time() - started, 1)


def scroll_to_source(page: Page, index: int, y: int) -> tuple[dict, float]:
    page.evaluate(
        "(i) => document.querySelector(`.review-strip-slice[data-page-index=\"${i}\"]`)?.scrollIntoView({block: 'start'})",
        index,
    )
    page.wait_for_timeout(400)
    slice_img = page.locator(f'.review-strip-slice[data-page-index="{index}"] img')
    scale = slice_img.bounding_box()["width"] / slice_img.evaluate("i => i.naturalWidth")
    view = page.locator(".review-document-viewport")
    vb = view.bounding_box()
    view.evaluate("(v, d) => { v.scrollTop += d; }", slice_img.bounding_box()["y"] + y * scale - (vb["y"] + 120))
    page.wait_for_timeout(1000)
    return slice_img.bounding_box(), scale


def crop_strip(images: list[np.ndarray], region: dict, path: Path) -> None:
    y1, y2, x1, x2 = region["y1"], region["y2"], region["x1"], region["x2"]
    gap = np.full((y2 - y1, 12, 3), 255, np.int16)
    parts = []
    for image in images:
        parts += [image[y1:y2, x1:x2], gap]
    Image.fromarray(np.concatenate(parts[:-1], axis=1).clip(0, 255).astype(np.uint8)).save(path)


def diff_pixels(a: np.ndarray, b: np.ndarray, region: dict, threshold: int = 8) -> int:
    y1, y2, x1, x2 = region["y1"], region["y2"], region["x1"], region["x2"]
    return int((np.abs(a[y1:y2, x1:x2] - b[y1:y2, x1:x2]).max(axis=2) > threshold).sum())


def review_views(tour: Tour, page: Page) -> None:
    tour.shot(page, "review", "Review: bản đã làm sạch, các lát ghép liền")
    overlays = page.locator(".review-text-object-overlay")
    tour.checks["text_objects_visible"] = overlays.count()
    page.get_by_role("button", name="Ảnh gốc", exact=True).click()
    page.wait_for_timeout(1000)
    tour.shot(page, "review-original", "Review: xem ảnh gốc")
    page.get_by_role("button", name="Sau inpaint", exact=True).click()
    wait_review(page)
    if overlays.count():
        overlays.first.click()
        page.wait_for_timeout(600)
        tour.shot(page, "review-text-inspector", "Chọn vùng chữ: bảng thuộc tính, font, bản dịch")
        page.keyboard.press("Escape")
    page.locator(".review-more-toggle").click()
    page.wait_for_timeout(400)
    tour.shot(page, "review-more-menu", "Menu ⋯: OCR, dịch, QC toàn chương, ngôn ngữ gốc")
    page.locator(".review-more-toggle").click()


def repaint(tour: Tour, page: Page, chapter_id: str, index: int, region: dict) -> None:
    pages = tour.manifest(chapter_id)["pages"]
    original = tour.image(pages[index]["original"])
    before = tour.image(pages[index]["clean"])
    tour.checks["repaint_region"] = {"page": index, **region}
    tour.checks["repaint_text_present_before"] = diff_pixels(original, before, region) == 0
    page.locator('.review-rail-tool[data-tool="brush"]').click()
    expect(page.locator(".review-brush-bar")).to_be_visible()
    page.locator(".brush-size-slider").fill("30")
    sb, scale = scroll_to_source(page, index, region["y1"])
    tour.shot(page, "review-repaint-target", "Chữ bị bộ phát hiện bỏ sót, còn nguyên sau khi xử lý")
    vb = page.locator(".review-document-viewport").bounding_box()
    cx1, cx2 = sb["x"] + region["x1"] * scale, sb["x"] + region["x2"] * scale
    y, bottom, strokes = sb["y"] + region["y1"] * scale + 20, sb["y"] + region["y2"] * scale, 0
    while y <= bottom - 10 and y < vb["y"] + vb["height"] - 10:
        page.mouse.move(cx1 + 8, y)
        page.mouse.down()
        page.mouse.move(cx2 - 8, y, steps=30)
        page.mouse.up()
        strokes += 1
        y += 36
    tour.checks["brush_strokes"] = strokes
    page.wait_for_timeout(500)
    tour.shot(page, "review-brush", "Tô cọ lên chữ còn sót")
    page.locator(".repaint-btn").click()
    page.wait_for_selector(".repaint-mode-dialog")
    tour.shot(page, "review-repaint-dialog", "Chọn chế độ repaint: mặc định hoặc LaMa")
    page.locator(".repaint-mode-option").nth(1).click()
    page.locator(".repaint-mode-confirm").click()
    started = time.time()
    page.wait_for_function(
        "() => [...document.querySelectorAll('.toast, [class*=toast]')].some(t => /Đã làm sạch|Không thể xử lý|Không có vùng/.test(t.textContent))",
        timeout=600_000,
    )
    tour.checks["repaint_s"] = round(time.time() - started, 1)
    tour.checks["repaint_toast"] = page.locator(".toast, [class*=toast]").last.inner_text()
    wait_review(page)
    page.locator('.review-rail-tool[data-tool="select"]').click()
    scroll_to_source(page, index, region["y1"])
    tour.shot(page, "review-after-repaint", "Sau repaint bằng LaMa")
    after = tour.image(tour.manifest(chapter_id)["pages"][index]["clean"])
    changed = diff_pixels(before, after, region)
    tour.checks["repaint_changed_pixels"] = changed
    if not changed:
        tour.errors.append("repaint changed nothing in the brushed region")
    crop_strip([original, before, after], region, tour.out / "zz-repaint-original-before-after.png")


def pick_bubble(tour: Tour, chapter_id: str, keep: list[int], avoid: int) -> tuple[int, dict] | None:
    pages = tour.manifest(chapter_id)["pages"]
    best = None
    for index in keep:
        if index == avoid:
            continue
        for box in pages[index].get("boxes") or []:
            if not isinstance(box, dict) or box.get("removed") or box.get("manual"):
                continue
            area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            if best is None or area > best[0]:
                best = (area, index, box)
    if best is None:
        return None
    _, index, box = best
    height, width = tour.image(pages[index]["original"]).shape[:2]
    pad = 12
    return index, {
        "x1": max(0, box["x1"] - pad), "y1": max(0, box["y1"] - pad),
        "x2": min(width, box["x2"] + pad), "y2": min(height, box["y2"] + pad),
    }


def preserve(tour: Tour, page: Page, chapter_id: str, keep: list[int], avoid: int) -> None:
    picked = pick_bubble(tour, chapter_id, keep, avoid)
    if picked is None:
        tour.errors.append("no detected bubble to preserve")
        return
    index, region = picked
    pages = tour.manifest(chapter_id)["pages"]
    original = tour.image(pages[index]["original"])
    cleaned = tour.image(pages[index]["clean"])
    tour.checks["preserve_page"] = index
    tour.checks["preserve_cleaned_pixels_before"] = diff_pixels(original, cleaned, region)
    scroll_to_source(page, index, region["y1"])
    tour.shot(page, "review-bubble-cleaned", "Bong bóng đã được làm sạch ở lần xử lý đầu")
    goto_preview(page)
    draw_preserve(tour, page, index, [region["x1"], region["y1"], region["x2"], region["y2"]])
    saved = tour.manifest(chapter_id)["pages"][index].get("preserve_regions") or []
    tour.checks["preserve_regions"] = saved
    if not saved:
        tour.errors.append("preserve region was not saved")
        tour.shot(page, "preview-preserve-missing", "Không lưu được vùng giữ nguyên")
        page.locator('.sidebar-link[data-stage="review"]').click()
        wait_review(page)
        return
    tour.shot(page, "preview-preserve", f"Đánh dấu vùng giữ nguyên quanh bong bóng (lát {index + 1})")
    tour.checks["reprocess_s"] = process(tour, page, "reprocessing", "Xử lý lại với vùng giữ nguyên")
    kept = tour.image(tour.manifest(chapter_id)["pages"][index]["clean"])
    region = saved[0]
    tour.checks["preserve_changed_pixels_after"] = diff_pixels(original, kept, region, threshold=0)
    if tour.checks["preserve_changed_pixels_after"]:
        tour.errors.append("preserve region differs from the original after reprocessing")
    scroll_to_source(page, index, region["y1"])
    tour.shot(page, "review-bubble-preserved", "Sau khi xử lý lại: bong bóng giữ nguyên chữ gốc")
    crop_strip([original, cleaned, kept], region, tour.out / "zz-preserve-original-cleaned-kept.png")


def extra_tools(tour: Tour, page: Page) -> None:
    page.get_by_role("button", name="Có chữ", exact=True).click()
    page.wait_for_timeout(1500)
    tour.shot(page, "review-rendered", "Xem bản có chữ (đã kết xuất)")
    page.get_by_role("button", name="Sau inpaint", exact=True).click()
    wait_review(page)
    page.locator('.review-rail-tool[data-tool="rectangle"]').click()
    page.wait_for_timeout(300)
    tour.shot(page, "review-rect-tool", "Công cụ vẽ vùng chữ mới")
    page.locator('.review-rail-tool[data-tool="select"]').click()
    page.locator(".review-more-toggle").click()
    qc = page.get_by_role("button", name="Kiểm tra AI toàn chương", exact=True)
    if qc.count():
        qc.click()
        page.wait_for_timeout(800)
        tour.shot(page, "review-chapter-qc", "Bảng kiểm tra AI toàn chương")
        page.keyboard.press("Escape")
    else:
        page.locator(".review-more-toggle").click()
    goto_preview(page)
    tour.shot(page, "preview-after", "Xem trước sau khi xử lý: trạng thái từng lát")
    page.locator('.sidebar-link[data-stage="review"]').click()
    wait_review(page)


def mobile(tour: Tour, browser, chapter_id: str) -> None:
    page = browser.new_page(viewport=MOBILE, device_scale_factor=2)
    page.goto(tour.base, wait_until="networkidle")
    tour.shot(page, "mobile-home", "Điện thoại: trang chủ")
    page.goto(f"{tour.base}/#{chapter_id}", wait_until="networkidle")
    page.reload(wait_until="networkidle")
    wait_review(page)
    tour.shot(page, "mobile-review", "Điện thoại: Review")
    page.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ui-tour"))
    parser.add_argument("--chapter-url")
    parser.add_argument("--chapter-id")
    parser.add_argument("--keep", default="1,2,3,4")
    parser.add_argument("--repaint", help="x1,y1,x2,y2 in source pixels of the first kept slice")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tour = Tour(args.base_url, args.out)
    keep = [int(v) for v in args.keep.split(",")]

    with sync_playwright() as playwright:
        executable = os.getenv("UI_TOUR_CHROMIUM") or None
        browser = playwright.chromium.launch(executable_path=executable)
        page = browser.new_page(viewport=DESKTOP, device_scale_factor=1)
        page.on("pageerror", lambda exc: tour.errors.append(f"page error: {exc}"))
        try:
            landing(tour, page)
            if args.chapter_url:
                chapter_id = import_chapter(tour, page, args.chapter_url)
            else:
                chapter_id = args.chapter_id
                open_preview(tour, page, chapter_id)
            skip_slices(tour, page, chapter_id, keep)
            tour.checks["process_s"] = process(tour, page, "processing", "Đang xử lý các lát đã chọn")
            review_views(tour, page)
            if args.repaint:
                x1, y1, x2, y2 = (int(v) for v in args.repaint.split(","))
                repaint(tour, page, chapter_id, keep[0], {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            preserve(tour, page, chapter_id, keep, keep[0] if args.repaint else -1)
            extra_tools(tour, page)
            mobile(tour, browser, chapter_id)
        except Exception as exc:
            tour.errors.append(f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
            try:
                tour.shot(page, "failure", "Màn hình lúc lỗi")
            except Exception:
                pass
        finally:
            browser.close()

    report = {"shots": tour.shots, "checks": tour.checks, "errors": tour.errors}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"checks": tour.checks, "errors": tour.errors}, ensure_ascii=False, indent=1))
    return 1 if tour.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
