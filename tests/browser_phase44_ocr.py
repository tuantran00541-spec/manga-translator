from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "app/static/js/chapter-ocr.js"


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        calls = {
            "start": [],
            "status_job1": 0,
            "retry": 0,
            "cancel": 0,
            "box": 0,
            "allow_complete": False,
        }

        def handle(route):
            url = route.request.url
            method = route.request.method
            if url.endswith("/api/ocr_box") and method == "POST":
                calls["box"] += 1
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"page_index":0,"box_id":"box_a","text":"FRESH","lang":"en","engine":"phase44-v1:test","cached":false,"committed":true,"stale":false}',
                )
                return
            if url.endswith("/api/ocr/chapter") and method == "POST":
                payload = route.request.post_data_json
                calls["start"].append(payload)
                job_id = "job3" if len(calls["start"]) > 1 else "job1"
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=(
                        '{"job_id":"%s","chapter_id":"deadbeef","lang":"en","concurrency":1,'
                        '"force":false,"status":"running","total":2,"completed":0,"recognized":0,'
                        '"empty":0,"cached":0,"stale":0,"failed":0,"errors":[]}' % job_id
                    ),
                )
                return
            if url.endswith("/api/ocr/chapter/job1") and method == "GET":
                calls["status_job1"] += 1
                if calls["allow_complete"]:
                    body = '{"job_id":"job1","chapter_id":"deadbeef","lang":"en","concurrency":1,"force":false,"status":"completed","total":2,"completed":1,"recognized":1,"empty":0,"cached":1,"stale":1,"failed":0,"errors":[]}'
                else:
                    body = '{"job_id":"job1","chapter_id":"deadbeef","lang":"en","concurrency":1,"force":false,"status":"running","total":2,"completed":1,"recognized":1,"empty":0,"cached":1,"stale":0,"failed":0,"errors":[]}'
                route.fulfill(status=200, content_type="application/json", body=body)
                return
            if url.endswith("/api/ocr/chapter/job1/retry") and method == "POST":
                calls["retry"] += 1
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"job_id":"job2","chapter_id":"deadbeef","lang":"en","concurrency":1,"force":false,"status":"running","total":1,"completed":0,"recognized":0,"empty":0,"cached":0,"stale":0,"failed":0,"errors":[]}',
                )
                return
            if url.endswith("/api/ocr/chapter/job2") and method == "GET":
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"job_id":"job2","chapter_id":"deadbeef","lang":"en","concurrency":1,"force":false,"status":"completed","total":1,"completed":1,"recognized":1,"empty":0,"cached":0,"stale":0,"failed":0,"errors":[]}',
                )
                return
            if url.endswith("/api/ocr/chapter/job3/cancel") and method == "POST":
                calls["cancel"] += 1
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"job_id":"job3","chapter_id":"deadbeef","lang":"en","concurrency":1,"force":false,"status":"cancelled","total":2,"completed":0,"recognized":0,"empty":0,"cached":0,"stale":0,"failed":0,"errors":[]}',
                )
                return
            route.abort()

        # Playwright 1.47 glob matching is stricter than newer versions around
        # a trailing **. Keep the chapter and legacy box routes explicit so the
        # smoke test validates browser behaviour instead of glob semantics.
        page.route("**/api/ocr/**", handle)
        page.route("**/api/ocr_box", handle)
        page.set_content(
            """
            <html><head><base href="http://ocr.test/"></head><body>
              <select id="lang-select"><option value="ja">JA</option><option value="en" selected>EN</option></select>
              <div class="review-workspace-shell">
                <div class="review-sticky-toolbar">
                  <div class="review-actions-group"><button class="review-primary-action">Editor</button></div>
                </div>
                <nav class="review-page-nav"><button class="nav-next">Next</button></nav>
                <div class="chapter-qc-panel"></div>
              </div>
              <span id="box-original"></span>
            </body></html>
            """
        )
        page.evaluate(
            """
            window.currentChapterId = 'deadbeef';
            window.currentManifest = {pages:[{boxes:[{id:'box_a',ocr_text:'STALE-CACHED',ocr_lang:'en'}]}]};
            window.parseApiResponse = async (response) => response.json();
            window.getErrorMessage = (status, data) => data.detail || `HTTP ${status}`;
            window.showToast = () => {};
            window.fetchOcr = async () => { throw new Error('legacy fetchOcr should have been replaced'); };
            undefined;
            """
        )
        page.add_script_tag(path=str(SCRIPT))

        page.wait_for_selector(".chapter-ocr-run")
        assert not page.locator(".nav-next").is_disabled()

        page.evaluate("window.fetchOcr(0, 0, document.getElementById('box-original'))")
        expect(page.locator("#box-original")).to_have_text("FRESH")
        assert calls["box"] == 1
        assert page.evaluate("window.currentManifest.pages[0].boxes[0].ocr_text") == "FRESH"

        page.click(".chapter-ocr-run")
        page.wait_for_selector(".chapter-ocr-cancel:visible")
        expect(page.locator(".chapter-ocr-summary")).to_contain_text("Đang nhận dạng")
        assert calls["start"][0] == {
            "chapter_id": "deadbeef",
            "lang": "en",
            "concurrency": 1,
            "force": False,
        }
        assert not page.locator(".nav-next").is_disabled()

        calls["allow_complete"] = True
        expect(page.locator(".chapter-ocr-summary")).to_contain_text("stale 1", timeout=6000)
        expect(page.locator(".chapter-ocr-retry")).to_be_visible()
        expect(page.locator(".chapter-ocr-progress")).to_have_attribute("value", "2")

        page.click(".chapter-ocr-retry")
        expect(page.locator(".chapter-ocr-summary")).to_contain_text("Hoàn tất", timeout=4000)
        assert calls["retry"] == 1
        expect(page.locator(".chapter-ocr-retry")).to_be_hidden()

        page.click(".chapter-ocr-run")
        page.wait_for_selector(".chapter-ocr-cancel:visible")
        page.click(".chapter-ocr-cancel")
        expect(page.locator(".chapter-ocr-summary")).to_contain_text("Đã hủy")
        assert calls["cancel"] == 1
        assert not page.locator(".nav-next").is_disabled()

        browser.close()


if __name__ == "__main__":
    main()
