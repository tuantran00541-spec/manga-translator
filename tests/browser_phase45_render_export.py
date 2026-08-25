from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
API_SCRIPT = ROOT / "app/static/js/api.js"


def main() -> None:
    calls = {"render": 0}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        def handle_render(route):
            calls["render"] += 1
            if calls["render"] == 1:
                route.fulfill(
                    status=409,
                    content_type="application/json",
                    body='{"detail":"stale render discarded"}',
                )
                return
            route.fulfill(
                status=200,
                content_type="application/json",
                body=(
                    '{"output":"/api/image/deadbeef/0/rendered",'
                    '"committed":true,"render_revision":5}'
                ),
            )

        page.route("**/api/render", handle_render)
        page.set_content(
            """
            <html><head><base href="http://render.test/"></head><body>
              <button class="editor-render-btn">Kết xuất bản dịch</button>
            </body></html>
            """
        )
        page.evaluate(
            """
            window.currentChapterId = 'deadbeef';
            window.currentManifest = {
              pages: [{
                rendered: false,
                render_revision: 1,
                text_objects: [{
                  id: 'obj1',
                  translation: 'HELLO',
                  style: {
                    color: 'auto', font: 'default', fontSize: 'auto', bold: false,
                    strokeWidth: 'auto', strokeColor: 'auto', bgColor: 'transparent',
                    cornerRadius: '0', horizontalAlign: 'center', verticalAlign: 'middle'
                  }
                }]
              }]
            };
            window.availableFonts = [];
            window.flushAllPendingPersists = async () => {};
            window._phase45Toasts = [];
            window._phase45Results = [];
            window.showToast = (message, kind) => window._phase45Toasts.push({message, kind});
            window.showRenderResult = (pageIndex, output) => window._phase45Results.push({pageIndex, output});
            undefined;
            """
        )
        page.add_script_tag(path=str(API_SCRIPT))

        page.evaluate("window.renderTranslations(0)")
        page.wait_for_function("window._phase45Toasts.length === 1")
        assert page.evaluate("window.currentManifest.pages[0].rendered") is False
        assert page.evaluate("window._phase45Results.length") == 0
        assert "stale render discarded" in page.evaluate("window._phase45Toasts[0].message")

        page.evaluate("window.renderTranslations(0)")
        page.wait_for_function("window._phase45Results.length === 1")
        assert page.evaluate("window.currentManifest.pages[0].rendered") is True
        assert page.evaluate("window._phase45Results[0].output") == "/api/image/deadbeef/0/rendered"
        assert calls["render"] == 2

        browser.close()


if __name__ == "__main__":
    main()
