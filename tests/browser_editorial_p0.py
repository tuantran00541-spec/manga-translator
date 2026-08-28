import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_JS = ROOT / "app/static/js/script-workspace.js"
FINAL_QC_JS = ROOT / "app/static/js/final-qc.js"


def main() -> None:
    updates: list[dict] = []
    approvals: list[dict] = []

    manifest = {
        "chapter_id": "deadbeef",
        "pages": [
            {
                "width": 600,
                "height": 800,
                "clean": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
                "original": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
                "rendered": True,
                "text_objects": [
                    {
                        "id": "text_1",
                        "region": {"x1": 100, "y1": 120, "x2": 400, "y2": 240},
                        "ocr_text": "Hello",
                        "translation": "Xin chào",
                        "script_status": "draft",
                    }
                ],
            }
        ],
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        def route_api(route):
            url = route.request.url
            if url.endswith("/api/text_objects/ensure"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps(manifest))
                return
            if url.endswith("/api/text_object/update"):
                payload = route.request.post_data_json
                updates.append(payload)
                route.fulfill(status=200, content_type="application/json", body=json.dumps(manifest))
                return
            if url.endswith("/api/script/review"):
                payload = route.request.post_data_json
                approvals.append({"script": payload})
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({
                        "chapter_id": "deadbeef",
                        "page_index": 0,
                        "object_id": "text_1",
                        "status": payload["status"],
                        "script_review_fingerprint": "test-fingerprint" if payload["status"] == "reviewed" else None,
                    }),
                )
                return
            if url.endswith("/api/final_qc/deadbeef"):
                report = {
                    "chapter_id": "deadbeef",
                    "ready_for_export": False,
                    "summary": {"pages_total": 1, "pages_required": 1, "pages_approved": 0, "blocking_issues": 0},
                    "pages": [{"page_index": 0, "skipped": False, "approved": False, "render_revision": 1, "approved_render_revision": 0, "issues": []}],
                }
                route.fulfill(status=200, content_type="application/json", body=json.dumps(report))
                return
            if url.endswith("/api/final_qc/page"):
                payload = route.request.post_data_json
                approvals.append(payload)
                report = {
                    "chapter_id": "deadbeef",
                    "ready_for_export": True,
                    "summary": {"pages_total": 1, "pages_required": 1, "pages_approved": 1, "blocking_issues": 0},
                    "pages": [{"page_index": 0, "skipped": False, "approved": True, "render_revision": 1, "approved_render_revision": 1, "issues": []}],
                }
                route.fulfill(status=200, content_type="application/json", body=json.dumps(report))
                return
            route.fulfill(status=404, content_type="application/json", body='{"detail":"not mocked"}')

        page.route("**/api/**", route_api)
        page.set_content(
            """
            <html><head><base href="http://editorial.test/"></head>
            <body data-app-stage="script"><main id="page-view"></main></body></html>
            """
        )
        page.evaluate(
            f"""
            window.currentChapterId = 'deadbeef';
            window.currentManifest = {json.dumps(manifest)};
            window.showToast = () => {{}};
            window.setWorkflowCheckpoint = async () => {{}};
            window.pageLabel = (_pages, index) => `Trang ${{index + 1}}`;
            window.parseApiResponse = async (response) => response.json();
            window.getErrorMessage = (status, data) => data?.detail || `HTTP ${{status}}`;
            undefined;
            """
        )
        page.add_script_tag(path=str(SCRIPT_JS))
        page.evaluate("window.renderScript()")
        page.wait_for_selector(".script-translation")
        page.locator(".script-translation").focus()
        page.locator(".script-translation").press("Control+Enter")
        page.wait_for_function("document.querySelector('.script-status-badge')?.textContent === 'Đã soát'")
        assert updates and updates[-1]["translation"] == "Xin chào"
        assert any(item.get("script", {}).get("status") == "reviewed" for item in approvals)

        page.evaluate(
            """
            document.body.dataset.appStage = 'final_qc';
            window.createPageNavigator = ({items, activeIndex, onSelect}) => {
              const element = document.createElement('div');
              element.className = 'fake-navigator';
              return { element, setActive: () => {}, setBusy: () => {}, selectByKey: onSelect };
            };
            undefined;
            """
        )
        page.add_script_tag(path=str(FINAL_QC_JS))
        page.evaluate("window.renderFinalQC()")
        page.wait_for_selector("text=Duyệt trang này")
        assert page.locator(".final-qc-export").is_disabled()
        page.get_by_text("Duyệt trang này", exact=True).click()
        page.wait_for_function("!document.querySelector('.final-qc-export').disabled")
        assert approvals and approvals[-1]["approved"] is True

        browser.close()


if __name__ == "__main__":
    main()
