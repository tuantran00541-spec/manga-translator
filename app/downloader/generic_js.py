from playwright.sync_api import sync_playwright

from app.downloader.base import BaseAdapter
from app.downloader.image_urls import best_srcset_candidate, resolve_image_candidate
from app.security import browser_request_allowed, validate_url


class GenericJsAdapter(BaseAdapter):
    img_selector = "img"
    min_width = 300
    max_scroll_rounds = 30
    stable_rounds_required = 3

    def can_handle(self, url: str) -> bool:
        return True

    def _scroll_until_stable(self, page) -> None:
        last_height = -1
        last_count = -1
        stable_rounds = 0
        for _ in range(self.max_scroll_rounds):
            height = int(page.evaluate("Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"))
            count = int(page.locator(self.img_selector).count())
            page.evaluate("window.scrollTo(0, Math.max(document.body.scrollHeight, document.documentElement.scrollHeight))")
            page.wait_for_timeout(400)
            next_height = int(page.evaluate("Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"))
            next_count = int(page.locator(self.img_selector).count())
            if next_height == height == last_height and next_count == count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
            last_height = next_height
            last_count = next_count
            if stable_rounds >= self.stable_rounds_required:
                break

    def extract_image_urls(self, chapter_url: str) -> list[str]:
        validate_url(chapter_url)
        urls = []
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=[
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-zygote",
                    "--disable-extensions",
                    "--disable-background-networking",
                ]
            )
            try:
                context = browser.new_context(service_workers="block")

                def guard_route(route):
                    if browser_request_allowed(route.request.url):
                        route.continue_()
                    else:
                        route.abort()

                context.route("**/*", guard_route)
                context.add_init_script(
                    "Object.defineProperty(globalThis, 'WebSocket', "
                    "{value: undefined, writable: false, configurable: false});"
                )
                page = context.new_page()
                page.goto(chapter_url, wait_until="domcontentloaded", timeout=60000)
                self._scroll_until_stable(page)

                elements = page.query_selector_all(self.img_selector)
                for el in elements:
                    box = el.bounding_box()
                    natural_width = 0
                    try:
                        natural_width = int(el.evaluate("img => img.naturalWidth || 0"))
                    except Exception:
                        natural_width = 0
                    visible_width = float(box["width"]) if box else 0.0
                    if max(visible_width, natural_width) < self.min_width:
                        continue
                    current_src = None
                    try:
                        current_src = el.evaluate("img => img.currentSrc || ''")
                    except Exception:
                        current_src = None
                    srcset = best_srcset_candidate(
                        el.get_attribute("data-srcset") or el.get_attribute("srcset")
                    )
                    src = resolve_image_candidate(
                        chapter_url,
                        current_src,
                        srcset,
                        el.get_attribute("data-src"),
                        el.get_attribute("data-original"),
                        el.get_attribute("data-lazy"),
                        el.get_attribute("src"),
                    )
                    if src and browser_request_allowed(src):
                        urls.append(src)
            finally:
                browser.close()
        return self._dedupe(urls)
