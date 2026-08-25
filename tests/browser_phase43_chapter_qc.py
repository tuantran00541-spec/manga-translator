from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "app/static/js/chapter-qc.js"


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        calls = {"status": 0}

        def handle(route):
            url = route.request.url
            method = route.request.method
            if url.endswith("/api/visual_qc/chapter") and method == "POST":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"job_id":"job1","chapter_id":"deadbeef","status":"running","total_regions":2,"completed_regions":0,"passed":0,"flagged":0,"failed":0,"results":[]}',
                )
                return
            if url.endswith("/api/visual_qc/chapter/job1") and method == "GET":
                calls["status"] += 1
                if calls["status"] == 1:
                    body = '{"job_id":"job1","chapter_id":"deadbeef","status":"running","total_regions":2,"completed_regions":1,"passed":1,"flagged":0,"failed":0,"results":[]}'
                else:
                    body = '{"job_id":"job1","chapter_id":"deadbeef","status":"completed","total_regions":2,"completed_regions":2,"passed":1,"flagged":1,"failed":0,"results":[{"page_index":0,"region_id":"P0001-R01","status":"flagged","issues":[{"issue_type":"residual_text","confidence":0.91,"bbox":[100,120,280,260],"reason":"glyph remains","recommended_action":"review"}]}]}'
                route.fulfill(status=200, content_type="application/json", body=body)
                return
            route.abort()

        page.route("**/api/visual_qc/**", handle)
        page.set_content(
            """
            <html><body>
              <div class="review-workspace-shell">
                <div class="review-sticky-toolbar">
                  <div class="review-actions-group"><button class="review-primary-action">Editor</button></div>
                </div>
                <nav class="review-page-nav">
                  <input class="workspace-nav-jump-input" value="1">
                </nav>
                <div class="review-card" data-page-index="0">
                  <div class="review-image-wrap" style="position:relative;width:500px;height:700px">
                    <img width="500" height="700">
                    <canvas class="brush-canvas"></canvas>
                  </div>
                  <button class="brush-toggle-btn">Brush</button>
                  <button class="clear-brush-btn">Clear</button>
                  <button class="repaint-btn">Repaint</button>
                  <button class="reset-manual-btn">Reset</button>
                  <button class="ai-qc-btn">Page AI</button>
                  <input class="brush-size-slider">
                </div>
              </div>
            </body></html>
            """
        )
        page.evaluate(
            """
            window.currentChapterId = 'deadbeef';
            window.currentManifest = {pages:[{skipped:false,width:1000,height:1400}]};
            window.parseApiResponse = async (response) => response.json();
            window.getErrorMessage = (status, data) => data.detail || `HTTP ${status}`;
            window.showToast = () => {};
            const jump = document.querySelector('.workspace-nav-jump-input');
            jump.addEventListener('change', () => { window.__jumped = jump.value; });
            """
        )
        page.add_script_tag(path=str(SCRIPT))

        page.wait_for_selector(".chapter-qc-run")
        page.click(".chapter-qc-run")
        page.wait_for_function("document.querySelector('.review-workspace-shell').classList.contains('review-chapter-qc-running')")
        assert page.locator(".repaint-btn").is_disabled()

        page.wait_for_selector(".chapter-qc-result", timeout=6000)
        summary = page.locator(".chapter-qc-summary").inner_text()
        assert "Hoàn tất" in summary
        assert not page.locator(".repaint-btn").is_disabled()

        page.click(".chapter-qc-result")
        page.wait_for_function("window.__jumped === '1'")
        page.wait_for_selector(".review-qc-highlight")
        assert page.locator(".review-qc-highlight").count() == 1
        assert page.evaluate("document.querySelector('.brush-canvas')._reviewDirty === undefined")

        browser.close()


if __name__ == "__main__":
    main()
