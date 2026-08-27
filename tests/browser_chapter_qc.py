from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "app/static/js/chapter-qc.js"


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        calls = {"status": 0, "start": None}

        def handle(route):
            url = route.request.url
            method = route.request.method
            if url.endswith("/api/visual_qc/chapter") and method == "POST":
                calls["start"] = route.request.post_data_json
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"job_id":"job1","chapter_id":"deadbeef","provider":"deepseek","model":"deepseek-v4-flash-vision-exp","status":"running","total_regions":2,"completed_regions":0,"passed":0,"flagged":0,"failed":0,"results":[],"usage":{"requests":1,"estimated_cost_usd":0.0012,"budget_usd":0.02}}',
                )
                return
            if url.endswith("/api/visual_qc/chapter/job1") and method == "GET":
                calls["status"] += 1
                if calls["status"] == 1:
                    body = '{"job_id":"job1","chapter_id":"deadbeef","provider":"deepseek","model":"deepseek-v4-flash-vision-exp","status":"running","total_regions":2,"completed_regions":1,"passed":1,"flagged":0,"failed":0,"results":[],"usage":{"requests":2,"estimated_cost_usd":0.0024,"budget_usd":0.02}}'
                else:
                    body = '{"job_id":"job1","chapter_id":"deadbeef","provider":"deepseek","model":"deepseek-v4-flash-vision-exp","status":"completed","total_regions":2,"completed_regions":2,"passed":1,"flagged":1,"failed":0,"results":[{"page_index":0,"region_id":"P0001-R01","status":"flagged","issues":[{"issue_type":"residual_text","confidence":0.91,"bbox":[100,120,280,260],"reason":"glyph remains","recommended_action":"review"}]}],"usage":{"requests":2,"estimated_cost_usd":0.0024,"budget_usd":0.02}}'
                route.fulfill(status=200, content_type="application/json", body=body)
                return
            route.abort()

        page.route("**/api/visual_qc/**", handle)
        page.set_content(
            """
            <html><head><base href="http://qc.test/"></head><body>
              <div class="review-workspace-shell">
                <div class="review-sticky-toolbar">
                  <div class="review-actions-group"><button class="review-primary-action">Editor</button></div>
                </div>
                <div class="workbench-stage-grid review-workbench-grid">
                  <aside class="page-navigator"></aside>
                  <div class="review-canvas-host">
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
                  <aside class="context-inspector review-inspector"></aside>
                </div>
              </div>
            </body></html>
            """
        )
        page.evaluate(
            """
            window.currentChapterId = 'deadbeef';
            window.currentManifest = {pages:[{skipped:false,width:1000,height:1400}]};
            window.deepseekVisualQCConfigured = true;
            window.parseApiResponse = async (response) => response.json();
            window.getErrorMessage = (status, data) => data.detail || `HTTP ${status}`;
            window.showToast = () => {};
            const workspace = document.querySelector('.review-workspace-shell');
            workspace._pageNavigator = {
              selectByKey(key) { window.__jumped = String(key); }
            };
            """
        )
        page.add_script_tag(path=str(SCRIPT))

        page.wait_for_selector(".chapter-qc-run")
        page.wait_for_selector(".review-inspector .chapter-qc-panel")
        page.select_option(".chapter-qc-provider", "deepseek")
        page.fill(".chapter-qc-budget-input", "0.02")
        page.click(".chapter-qc-run")
        page.wait_for_function("document.querySelector('.review-workspace-shell').classList.contains('review-chapter-qc-running')")
        assert calls["start"]["provider"] == "deepseek"
        assert calls["start"]["budget_usd"] == 0.02
        assert page.locator(".repaint-btn").is_disabled()
        assert page.locator(".chapter-qc-provider").is_disabled()

        page.wait_for_selector(".chapter-qc-result", timeout=6000)
        summary = page.locator(".chapter-qc-summary").inner_text()
        assert "DeepSeek" in summary
        assert "Hoàn tất" in summary
        assert "$0.0024 / $0.020" in page.locator(".chapter-qc-usage").inner_text()
        assert not page.locator(".repaint-btn").is_disabled()

        page.click(".chapter-qc-result")
        page.wait_for_function("window.__jumped === '0'")
        page.wait_for_selector(".review-qc-highlight")
        assert page.locator(".review-qc-highlight").count() == 1
        assert page.evaluate("document.querySelector('.brush-canvas')._reviewDirty === undefined")

        browser.close()


if __name__ == "__main__":
    main()
