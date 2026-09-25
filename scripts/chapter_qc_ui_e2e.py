from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GEMINI_API_KEY", "chapter-qc-e2e")

import uvicorn  # noqa: E402
from playwright.sync_api import Page, expect, sync_playwright  # noqa: E402

from app.visual_qc.batch_protocol import RegionBatchDecision, RegionBatchIssue  # noqa: E402
import app.routers.visual_qc as visual_qc_router  # noqa: E402

CHAPTER_ID = "f00d0001"
FLAGGED_PAGE = 1
FLAGGED_BBOX = (240, 360, 720, 640)
FAILING_PAGE = 2


class FakeRegionQC:
    delay_seconds = 1.2
    fail_page = True
    calls: list[tuple[str, ...]] = []

    def __init__(self, model: str = "fake-qc", timeout_seconds: int = 30):
        self.model = model

    def inspect(self, sheet, regions_by_id, api_key, *, mode):
        FakeRegionQC.calls.append(tuple(sorted(regions_by_id)))
        time.sleep(FakeRegionQC.delay_seconds)
        if FakeRegionQC.fail_page and any(r.page_index == FAILING_PAGE for r in regions_by_id.values()):
            raise RuntimeError("simulated provider outage")
        decisions = []
        for region_id, region in regions_by_id.items():
            if region.page_index == FLAGGED_PAGE:
                issue = RegionBatchIssue(
                    page_index=region.page_index,
                    region_id=region_id,
                    issue_type="residual_text",
                    confidence=0.9,
                    bbox=FLAGGED_BBOX,
                    reason="letters left in the bubble",
                    recommended_action="repaint",
                )
                decisions.append(RegionBatchDecision(region.page_index, region_id, "flagged", (issue,)))
            else:
                decisions.append(RegionBatchDecision(region.page_index, region_id, "pass", ()))
        return decisions


def start_server(port: int) -> uvicorn.Server:
    visual_qc_router.GeminiRegionQC = FakeRegionQC
    from app.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.1)
    return server


def open_review(page: Page, base_url: str) -> None:
    page.goto(base_url, wait_until="networkidle")
    page.locator(f'.recent-card[data-chapter-id="{CHAPTER_ID}"]').click()
    page.wait_for_selector(".review-document-shell")
    page.wait_for_function("() => document.querySelector('.review-stitched-image .review-strip-slice')")


def open_panel(page: Page) -> None:
    page.locator(".review-more-toggle").click()
    opener = page.locator(".review-more-actions .chapter-qc-open")
    expect(opener).to_have_count(1)
    opener.click()
    expect(page.locator(".chapter-qc-floating")).to_be_visible()
    expect(page.locator(".review-more-menu")).not_to_have_attribute("open", "")


def wait_finished(page: Page) -> None:
    page.wait_for_function(
        "() => !document.querySelector('.review-workspace-shell.review-chapter-qc-running')"
        " && /Hoàn tất|Đã hủy/.test(document.querySelector('.chapter-qc-summary')?.textContent || '')",
        timeout=30000,
    )


def exercise_desktop(page: Page, base_url: str, artifacts: Path) -> None:
    open_review(page, base_url)
    page.screenshot(path=str(artifacts / "desktop-review.png"))
    open_panel(page)
    panel = page.locator(".chapter-qc-floating")
    expect(panel.locator(".chapter-qc-summary")).to_have_text("Chưa chạy kiểm tra toàn chương.")
    expect(panel.locator(".chapter-qc-cancel")).to_be_hidden()
    expect(panel.locator(".chapter-qc-retry")).to_be_hidden()

    with page.expect_request(lambda r: r.url.endswith("/api/visual_qc/chapter") and r.method == "POST") as started:
        panel.locator(".chapter-qc-run").click()
    request = started.value.post_data_json
    assert request["chapter_id"] == CHAPTER_ID and request["provider"] == "gemini", request
    expect(page.locator(".review-workspace-shell")).to_have_class(re.compile("review-chapter-qc-running"))
    expect(page.locator(".repaint-btn")).to_be_disabled()
    expect(panel.locator(".chapter-qc-run")).to_be_disabled()
    expect(page.locator(".chapter-qc-open")).to_have_text("AI đang kiểm tra toàn chương…")
    expect(panel.locator(".chapter-qc-summary")).to_contain_text("Đang kiểm tra")
    veiled = page.evaluate(
        """() => [...document.querySelectorAll('.review-stitched-image .review-image-wrap')]
          .filter((wrap) => getComputedStyle(wrap, '::after').content !== 'none').length"""
    )
    assert veiled == 0, f"{veiled} page(s) are covered while chapter QC runs"
    overlay = page.locator('.review-text-object-overlay[data-page-index="0"]').first
    overlay.click(force=True)
    expect(overlay).not_to_have_class(re.compile("selected"))
    page.screenshot(path=str(artifacts / "desktop-running.png"))
    panel.locator(".chapter-qc-cancel").click()
    wait_finished(page)
    expect(panel.locator(".chapter-qc-summary")).to_contain_text("Đã hủy")
    expect(page.locator(".repaint-btn")).to_be_enabled()

    panel.locator(".chapter-qc-run").click()
    wait_finished(page)
    summary = panel.locator(".chapter-qc-summary")
    expect(summary).to_contain_text("Hoàn tất")
    expect(summary).to_contain_text("1 cần xem")
    expect(summary).to_contain_text("1 lỗi")
    expect(panel.locator(".chapter-qc-retry")).to_be_visible()
    results = panel.locator(".chapter-qc-result")
    expect(results).to_have_count(1)
    expect(results.first).to_contain_text(f"Trang {FLAGGED_PAGE + 1}")
    expect(results.first).to_contain_text("Còn sót chữ · 90%")
    expect(page.locator(".chapter-qc-open")).to_have_text("Kết quả kiểm tra AI")

    page.locator(".review-document-viewport").evaluate("element => element.scrollTop = 0")
    results.first.click()
    marker = page.locator(".review-stitched-image > .review-qc-highlight")
    expect(marker).to_have_count(1)
    geometry = page.evaluate(
        """(page) => {
          const shell = document.querySelector('.review-document-shell');
          const desc = shell._descriptors.find((d) => Number(d.item.canonicalIndex) === page);
          const marker = document.querySelector('.review-qc-highlight');
          return { sourceY1: desc.sourceY1, localY1: desc.localY1,
                   left: parseFloat(marker.style.left), top: parseFloat(marker.style.top),
                   width: parseFloat(marker.style.width), height: parseFloat(marker.style.height) };
        }""",
        FLAGGED_PAGE,
    )
    x1, y1, x2, y2 = FLAGGED_BBOX
    expected_top = geometry["sourceY1"] + (y1 - geometry["localY1"])
    assert geometry["left"] == x1 and geometry["width"] == x2 - x1, geometry
    assert geometry["top"] == expected_top and geometry["height"] == y2 - y1, geometry
    page.wait_for_function(
        """() => {
          const m = document.querySelector('.review-qc-highlight').getBoundingClientRect();
          const v = document.querySelector('.review-document-viewport').getBoundingClientRect();
          return m.top >= v.top && m.bottom <= v.bottom;
        }""",
        timeout=5000,
    )
    page.screenshot(path=str(artifacts / "desktop-result.png"))

    FakeRegionQC.fail_page = False
    panel.locator(".chapter-qc-retry").click()
    wait_finished(page)
    expect(summary).to_contain_text("0 lỗi")
    expect(panel.locator(".chapter-qc-retry")).to_be_hidden()

    panel.locator(".chapter-qc-close").click()
    expect(panel).to_be_hidden()
    expect(marker).to_have_count(0)


def exercise_mobile(page: Page, base_url: str, artifacts: Path) -> None:
    open_review(page, base_url)
    open_panel(page)
    box = page.locator(".chapter-qc-floating").bounding_box()
    viewport = page.viewport_size
    assert box and box["x"] >= 0 and box["x"] + box["width"] <= viewport["width"], box
    expect(page.locator(".chapter-qc-floating .chapter-qc-run")).to_be_visible()
    page.screenshot(path=str(artifacts / "mobile-panel.png"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive the whole-chapter AI QC panel against the real app with a fake AI client.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "chapter-qc-e2e")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "create_ui_smoke_fixture.py")], check=True, cwd=ROOT)
    server = start_server(args.port)
    base_url = f"http://127.0.0.1:{args.port}"
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            for name, viewport, exercise in (
                ("desktop", {"width": 1440, "height": 960}, exercise_desktop),
                ("mobile", {"width": 390, "height": 844}, exercise_mobile),
            ):
                context = browser.new_context(viewport=viewport)
                page = context.new_page()
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                exercise(page, base_url, args.artifacts)
                context.close()
                print(f"{name}: PASS")
            browser.close()
    finally:
        server.should_exit = True
    if errors:
        raise SystemExit(f"page errors: {errors}")
    print(f"AI calls: {len(FakeRegionQC.calls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
