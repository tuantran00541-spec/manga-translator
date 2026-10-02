"""Walks the app like a first-time user and records what they see."""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import requests
from playwright.sync_api import Page, sync_playwright

APP = "http://127.0.0.1:8000"
GATEWAY = "http://127.0.0.1:8100"
TOAST_HOOK = """
window.__toasts = [];
new MutationObserver((records) => records.forEach((r) => r.addedNodes.forEach((n) => {
  if (n.nodeType === 1 && n.classList.contains('ui-toast')) window.__toasts.push(n.textContent.trim());
}))).observe(document, { childList: true, subtree: true });
"""


class Journal:
    def __init__(self, out: Path):
        self.out = out
        self.rows: list[dict] = []
        self.errors: list[str] = []
        self.shots = 0

    def shot(self, page: Page, name: str, full: bool = False) -> str:
        self.shots += 1
        file = f"{self.shots:02d}-{name}.png"
        page.screenshot(path=str(self.out / file), full_page=full)
        return file

    @contextmanager
    def step(self, page: Page, name: str):
        row = {"step": name, "ok": True, "notes": [], "shots": []}
        page.evaluate("window.__toasts = []")
        started = time.perf_counter()
        try:
            yield row
        except Exception as exc:  # noqa: BLE001
            row["ok"] = False
            row["notes"].append(f"FAILED: {type(exc).__name__}: {str(exc)[:400]}")
            try:
                row["shots"].append(self.shot(page, f"{name}-failed"))
            except Exception:  # noqa: BLE001
                pass
        row["seconds"] = round(time.perf_counter() - started, 1)
        try:
            row["toasts"] = page.evaluate("window.__toasts || []")
        except Exception:  # noqa: BLE001
            row["toasts"] = []
        row["console_errors"] = self.errors[:]
        self.errors.clear()
        self.rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        self.save()

    def save(self) -> None:
        (self.out / "journal.json").write_text(json.dumps(self.rows, ensure_ascii=False, indent=1))
        lines = ["# First-use walkthrough", ""]
        for row in self.rows:
            lines.append(f"## {'OK' if row['ok'] else 'FAIL'} · {row['step']} · {row['seconds']}s")
            lines += [f"- {n}" for n in row["notes"]]
            lines += [f"- toast: {t}" for t in row.get("toasts", [])]
            lines += [f"- console error: {e}" for e in row.get("console_errors", [])]
            lines += [f"![]({s})" for s in row["shots"]]
            lines.append("")
        (self.out / "journal.md").write_text("\n".join(lines))


def visible_text(page: Page, selector: str) -> str:
    return " | ".join(t.strip() for t in page.locator(selector).all_inner_texts() if t.strip())[:600]


def wait_stage(page: Page, stage: str, timeout_s: float, journal: Journal, row: dict, tag: str) -> None:
    deadline = time.time() + timeout_s
    next_shot = time.time() + 20
    while time.time() < deadline:
        if page.evaluate("document.body.dataset.appStage") == stage and (
            stage != "review" or page.locator(".review-stitched-image").count()
        ):
            return
        if time.time() >= next_shot:
            row["shots"].append(journal.shot(page, f"{tag}-progress"))
            row["notes"].append("progress text: " + visible_text(page, "#workbench-status, .process-progress, .ui-state, .review-busy"))
            next_shot = time.time() + 60
        page.wait_for_timeout(1000)
    raise TimeoutError(f"stage {stage} not reached in {timeout_s}s")


def manual_path(page: Page, journal: Journal, url: str, out: Path) -> None:
    with journal.step(page, "home") as row:
        page.goto(APP, wait_until="networkidle")
        row["notes"].append("home text: " + visible_text(page, "#home-view"))
        row["shots"].append(journal.shot(page, "home"))

    with journal.step(page, "open-import") as row:
        page.get_by_role("button", name="Bắt đầu dự án đầu tiên", exact=True).click()
        page.wait_for_timeout(500)
        row["notes"].append("import text: " + visible_text(page, "#import-view .page-intro, #import-view h2"))
        row["shots"].append(journal.shot(page, "import", full=True))

    with journal.step(page, "load-chapter-from-link") as row:
        page.locator("#chapter-url").fill(url)
        row["notes"].append(f"'Tải chương' buttons on screen: {page.get_by_role('button', name='Tải chương', exact=True).count()}")
        page.locator("#load-btn").click()
        wait_stage(page, "preview", 600, journal, row, "download")
        page.wait_for_timeout(1500)
        row["notes"].append("preview header: " + visible_text(page, ".preview-workspace-title, #app-context"))
        row["notes"].append(f"slices listed: {page.locator('.page-navigator-item').count()}")
        row["shots"].append(journal.shot(page, "preview"))
    if not journal.rows[-1]["ok"]:
        return

    with journal.step(page, "preview-browse-and-skip") as row:
        items = page.locator(".page-navigator-item")
        if items.count() > 2:
            items.nth(2).click()
            page.wait_for_timeout(800)
        page.get_by_role("button", name="Bỏ qua lát ảnh", exact=True).click()
        page.wait_for_timeout(800)
        row["notes"].append("after skip: " + visible_text(page, ".preview-workspace-title"))
        row["shots"].append(journal.shot(page, "preview-skip"))
        page.get_by_role("button", name="Đã bỏ qua · Chọn để khôi phục", exact=True).click()
        page.wait_for_timeout(800)
        row["notes"].append("after restore: " + visible_text(page, ".preview-workspace-title"))

    with journal.step(page, "preview-preserve-region") as row:
        page.get_by_role("button", name="Đánh dấu vùng giữ nguyên", exact=True).click()
        box = page.locator(".preview-canvas-surface img").first.bounding_box()
        if box:
            x, y = box["x"] + box["width"] * 0.2, box["y"] + 40
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 200, y + 120, steps=8)
            page.mouse.up()
        page.wait_for_timeout(800)
        row["shots"].append(journal.shot(page, "preview-preserve"))
        page.get_by_role("button", name="Xóa vùng giữ nguyên", exact=True).click()
        page.wait_for_timeout(800)
        row["notes"].append("inspector: " + visible_text(page, ".context-inspector"))

    with journal.step(page, "process-chapter") as row:
        page.locator("#start-action").click()
        page.wait_for_timeout(1000)
        confirm = page.get_by_role("button", name="Bắt đầu xử lý", exact=True)
        if confirm.count() and confirm.first.is_visible():
            row["notes"].append("a confirm step appeared: " + visible_text(page, "[role=dialog], .ui-dialog, .review-confirm"))
            row["shots"].append(journal.shot(page, "process-confirm"))
            confirm.first.click()
        wait_stage(page, "review", 1800, journal, row, "processing")
        page.wait_for_timeout(3000)
        row["notes"].append(f"text regions on screen: {page.locator('.review-text-object-overlay').count()}")
        row["shots"].append(journal.shot(page, "review-first-look"))
    if not journal.rows[-1]["ok"]:
        return

    with journal.step(page, "review-views") as row:
        for name in ("Ảnh gốc", "Sau inpaint"):
            page.get_by_role("button", name=name, exact=True).click()
            page.wait_for_timeout(2500)
            row["shots"].append(journal.shot(page, f"view-{name}"))

    with journal.step(page, "tool-tooltips") as row:
        rail = page.locator(".review-rail-tool")
        row["notes"].append(f"tools in rail: {rail.count()}")
        rail.nth(3).hover()
        page.wait_for_timeout(600)
        row["shots"].append(journal.shot(page, "tooltip-brush"))

    with journal.step(page, "select-bubble") as row:
        overlay = page.locator(".review-text-object-overlay").first
        overlay.scroll_into_view_if_needed()
        overlay.click()
        page.wait_for_timeout(800)
        ocr = page.locator(".review-floating-inspector textarea").first.input_value()
        row["notes"].append(f"OCR text of first region: {ocr[:120]!r}")
        row["shots"].append(journal.shot(page, "inspector"))

    with journal.step(page, "type-translation") as row:
        area = page.locator(".review-floating-inspector .translation-textarea")
        area.fill("Thử dịch câu đầu tiên")
        page.wait_for_timeout(1500)
        page.keyboard.press("Escape")
        row["notes"].append("save status: " + visible_text(page, ".editor-save-status"))

    with journal.step(page, "ocr-whole-chapter") as row:
        page.get_by_role("button", name="Thêm thao tác", exact=True).click()
        page.wait_for_timeout(500)
        row["shots"].append(journal.shot(page, "more-menu"))
        run = page.get_by_role("button", name="OCR toàn chương", exact=True)
        if run.count() and run.first.is_enabled():
            run.first.click()
            page.wait_for_timeout(1000)
            deadline = time.time() + 600
            while time.time() < deadline and not any("OCR" in t for t in page.evaluate("window.__toasts")):
                page.wait_for_timeout(2000)
            row["notes"].append("OCR panel: " + visible_text(page, ".chapter-ocr-panel"))
            row["shots"].append(journal.shot(page, "ocr-done"))
        else:
            row["notes"].append("OCR toàn chương button missing or disabled")
        page.keyboard.press("Escape")

    with journal.step(page, "proof-panel") as row:
        page.get_by_role("button", name="Soát chữ toàn chương", exact=True).click()
        page.wait_for_timeout(1000)
        rows = page.locator(".review-proof-row")
        row["notes"].append(f"rows in proof panel: {rows.count()}")
        fields = page.locator(".review-proof-translation")
        for i in range(min(4, fields.count())):
            fields.nth(i).fill(f"Câu dịch số {i + 1}")
            page.wait_for_timeout(300)
        page.wait_for_timeout(1500)
        row["shots"].append(journal.shot(page, "proof-panel"))
        page.locator(".review-proof-find").fill("số")
        page.wait_for_timeout(500)
        row["notes"].append(f"rows matching 'số': {rows.count()}")
        page.locator(".review-proof-find").fill("")
        page.locator(".review-proof-close").click()

    with journal.step(page, "draw-new-region") as row:
        page.locator(".review-document-viewport").focus()
        page.get_by_role("button", name="Vùng chữ nhật: Kéo quanh bong bóng để tạo vùng OCR.").click()
        before = page.locator(".review-text-object-overlay").count()
        box = page.locator(".review-document-viewport").bounding_box()
        x, y = box["x"] + box["width"] * 0.35, box["y"] + box["height"] * 0.4
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 160, y + 90, steps=8)
        page.mouse.up()
        page.wait_for_timeout(2500)
        row["notes"].append(f"regions before/after drawing: {before}/{page.locator('.review-text-object-overlay').count()}")
        row["shots"].append(journal.shot(page, "new-region"))

    with journal.step(page, "undo-redo") as row:
        page.locator(".review-document-viewport").focus()
        page.keyboard.press("Escape")
        page.locator(".review-undo-btn").click()
        page.wait_for_timeout(1500)
        row["notes"].append(f"regions after undo: {page.locator('.review-text-object-overlay').count()}")
        page.locator(".review-redo-btn").click()
        page.wait_for_timeout(1500)
        row["notes"].append(f"regions after redo: {page.locator('.review-text-object-overlay').count()}")

    with journal.step(page, "style-preset") as row:
        page.locator(".review-text-object-overlay").first.click()
        page.wait_for_timeout(600)
        page.locator(".review-floating-inspector summary", has_text="Kiểu chữ").click()
        page.locator(".bold-toggle-btn").click()
        page.locator(".style-preset-name").fill("Lời thoại đậm")
        page.locator(".style-preset-store").click()
        page.wait_for_timeout(800)
        row["shots"].append(journal.shot(page, "preset"))
        page.keyboard.press("Escape")

    with journal.step(page, "brush-clean") as row:
        page.locator(".review-document-viewport").focus()
        page.get_by_role("button", name="Cọ Inpaint", exact=False).first.click()
        box = page.locator(".review-document-viewport").bounding_box()
        x, y = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.3
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 120, y + 10, steps=10)
        page.mouse.up()
        page.wait_for_timeout(500)
        row["shots"].append(journal.shot(page, "brush-painted"))
        page.get_by_role("button", name="Làm sạch vùng", exact=True).click()
        deadline = time.time() + 300
        while time.time() < deadline and page.locator(".review-busy").count():
            page.wait_for_timeout(1000)
        page.wait_for_timeout(2000)
        row["shots"].append(journal.shot(page, "brush-cleaned"))
        page.keyboard.press("Escape")

    with journal.step(page, "lettered-view") as row:
        started = time.perf_counter()
        page.get_by_role("button", name="Có chữ", exact=True).click()
        page.wait_for_timeout(500)
        deadline = time.time() + 600
        while time.time() < deadline and page.locator(".review-busy").count():
            page.wait_for_timeout(1000)
        page.wait_for_timeout(2500)
        row["notes"].append(f"lettered view ready after {time.perf_counter() - started:.1f}s")
        first = page.locator(".review-text-object-overlay").first
        if first.count():
            first.scroll_into_view_if_needed()
        row["shots"].append(journal.shot(page, "lettered"))

    with journal.step(page, "export") as row:
        with page.expect_download(timeout=600_000) as info:
            page.get_by_role("button", name="Xuất chương (.zip)", exact=True).click()
        path = out / "manual-export.zip"
        info.value.save_as(str(path))
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        row["notes"].append(f"zip has {len(names)} files: {names[:5]}")
        path.unlink()

    with journal.step(page, "keyboard-help") as row:
        page.locator(".review-document-viewport").focus()
        page.keyboard.press("?")
        page.wait_for_timeout(500)
        row["shots"].append(journal.shot(page, "shortcut-help"))
        page.keyboard.press("?")

    with journal.step(page, "settings") as row:
        page.get_by_role("button", name="Cài đặt", exact=True).click()
        page.wait_for_timeout(1500)
        row["notes"].append("settings: " + visible_text(page, "#settings-drawer"))
        row["shots"].append(journal.shot(page, "settings"))
        page.keyboard.press("Escape")

    with journal.step(page, "dark-theme") as row:
        page.locator("#theme-select").select_option("dark")
        page.wait_for_timeout(800)
        row["shots"].append(journal.shot(page, "dark"))
        page.locator("#theme-select").select_option("system")

    with journal.step(page, "back-home-recent") as row:
        page.get_by_role("button", name="Trang chủ", exact=True).click()
        page.wait_for_timeout(1000)
        row["notes"].append("home text: " + visible_text(page, "#home-view"))
        row["shots"].append(journal.shot(page, "home-recent"))


def upload_path(page: Page, journal: Journal, out: Path) -> None:
    with journal.step(page, "upload-zip") as row:
        raw = sorted(p for p in Path("data/raw").rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})[:3]
        archive = out / "upload.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for i, p in enumerate(raw):
                z.write(p, f"{i:03d}{p.suffix}")
        page.get_by_role("button", name="Nhập nội dung", exact=True).click()
        page.wait_for_timeout(500)
        with page.expect_file_chooser() as chooser:
            page.locator("#upload-dropzone").click()
        chooser.value.set_files(str(archive))
        wait_stage(page, "preview", 300, journal, row, "upload")
        page.wait_for_timeout(1500)
        row["notes"].append(f"uploaded {len(raw)} images; preview: " + visible_text(page, ".preview-workspace-title"))
        row["shots"].append(journal.shot(page, "upload-preview"))
        archive.unlink()


def ai_mode_path(page: Page, journal: Journal, url: str, admin: str) -> None:
    with journal.step(page, "ai-mode-first-look") as row:
        page.goto(APP, wait_until="networkidle")
        page.get_by_role("button", name="Nhập nội dung", exact=True).click()
        page.wait_for_timeout(1500)
        row["notes"].append("A.I panel: " + visible_text(page, "#ai-mode-panel"))
        row["shots"].append(journal.shot(page, "ai-mode", full=True))

    with journal.step(page, "ai-mode-sign-in") as row:
        page.locator("#ai-mode-email").fill("nguoi-moi@example.com")
        page.locator("#ai-mode-login-submit").click()
        page.wait_for_timeout(1500)
        toasts = page.evaluate("window.__toasts")
        code = next((t.split(":")[-1].strip() for t in toasts if "Mã thử nghiệm" in t), "")
        row["notes"].append(f"code toast: {toasts}")
        page.locator("#ai-mode-code").fill(code)
        page.locator("#ai-mode-login-submit").click()
        page.wait_for_timeout(2000)
        row["notes"].append("plan bar: " + visible_text(page, "#ai-mode-plan"))
        row["shots"].append(journal.shot(page, "ai-signed-in"))

    with journal.step(page, "ai-mode-top-up-simulated") as row:
        account_id = requests.get(f"{GATEWAY}/v1/admin/accounts", params={"email": "nguoi-moi@example.com"},
                                  headers={"X-Admin-Key": admin}, timeout=10).json()["account_id"]
        requests.post(f"{GATEWAY}/v1/admin/accounts/{account_id}/credit", json={"amount_usd": 1.0, "note": "walkthrough"},
                      headers={"X-Admin-Key": admin}, timeout=10).raise_for_status()
        page.reload(wait_until="networkidle")
        page.get_by_role("button", name="Nhập nội dung", exact=True).click()
        page.wait_for_timeout(1500)
        row["notes"].append("plan bar after $1 credit: " + visible_text(page, "#ai-mode-plan"))

    with journal.step(page, "ai-mode-run") as row:
        page.locator("#ai-mode-url").fill(url)
        row["notes"].append("provider picked: " + page.locator("#ai-mode-provider").input_value())
        page.locator("#ai-mode-run").click()
        started = time.time()
        next_shot = started + 30
        while time.time() - started < 3600:
            if page.locator("#ai-mode-open").is_visible():
                break
            if time.time() >= next_shot:
                row["shots"].append(journal.shot(page, "ai-progress"))
                row["notes"].append(f"{time.time() - started:.0f}s: " + visible_text(page, "#ai-mode-status"))
                next_shot = time.time() + 120
            page.wait_for_timeout(3000)
        row["notes"].append(f"finished after {time.time() - started:.0f}s: " + visible_text(page, "#ai-mode-status"))
        row["shots"].append(journal.shot(page, "ai-done", full=True))
    if not journal.rows[-1]["ok"]:
        return

    with journal.step(page, "ai-mode-open-result") as row:
        page.locator("#ai-mode-open").click()
        wait_stage(page, "review", 120, journal, row, "ai-open")
        page.wait_for_timeout(4000)
        row["shots"].append(journal.shot(page, "ai-result-top"))
        viewport = page.locator(".review-document-viewport")
        height = viewport.evaluate("v => v.scrollHeight")
        for i, frac in enumerate((0.25, 0.5, 0.75)):
            viewport.evaluate(f"v => v.scrollTop = {int(height * frac)}")
            page.wait_for_timeout(2500)
            row["shots"].append(journal.shot(page, f"ai-result-{i + 1}"))
        page.locator(".review-document-viewport").focus()
        page.keyboard.press("Control+f")
        page.wait_for_timeout(1000)
        row["notes"].append(f"translated rows: {page.locator('.review-proof-row').count()}")
        row["shots"].append(journal.shot(page, "ai-proof-panel"))


def mobile_pass(browser, journal: Journal) -> None:
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    page.add_init_script(TOAST_HOOK)
    page.on("pageerror", lambda e: journal.errors.append(str(e)))
    with journal.step(page, "mobile-home") as row:
        page.goto(APP, wait_until="networkidle")
        row["shots"].append(journal.shot(page, "mobile-home"))
        page.get_by_role("button", name="Mở menu", exact=True).click()
        page.wait_for_timeout(500)
        row["shots"].append(journal.shot(page, "mobile-menu"))
    with journal.step(page, "mobile-open-recent") as row:
        page.keyboard.press("Escape")
        page.locator(".recent-card").first.click()
        page.wait_for_timeout(5000)
        row["shots"].append(journal.shot(page, "mobile-chapter"))
        overlay = page.locator(".review-text-object-overlay").first
        if overlay.count():
            overlay.scroll_into_view_if_needed()
            overlay.tap()
            page.wait_for_timeout(800)
            row["shots"].append(journal.shot(page, "mobile-inspector"))
    page.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--ai-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    journal = Journal(args.out)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page.add_init_script(TOAST_HOOK)
        page.on("pageerror", lambda e: journal.errors.append(str(e)))
        page.on("console", lambda m: journal.errors.append(m.text) if m.type == "error" else None)
        manual_path(page, journal, args.url, args.out)
        upload_path(page, journal, args.out)
        mobile_pass(browser, journal)
        ai_mode_path(page, journal, args.ai_url, os.environ["GATEWAY_ADMIN_KEY"])
        browser.close()
    failed = [r["step"] for r in journal.rows if not r["ok"]]
    print("failed steps:", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
