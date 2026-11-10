from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright
from playwright_stealth import Stealth

from app.downloader.base import BaseAdapter
from app.downloader.image_urls import best_srcset_candidate, resolve_image_candidate
from app.parameters import (
    DOWNLOAD_JS_MAX_SCROLL_ROUNDS,
    DOWNLOAD_JS_MIN_IMAGE_WIDTH,
    DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS,
    DOWNLOAD_JS_SCROLL_WAIT_MS,
    DOWNLOAD_JS_STABLE_ROUNDS,
)
from app.security import browser_request_allowed, validate_url

# A desktop Chrome fingerprint the stealth evasions match (sec-ch-ua and
# userAgentData are derived from this UA, platform defaults to Win32).
REAL_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class GenericJsAdapter(BaseAdapter):
    img_selector = "img"
    min_width = DOWNLOAD_JS_MIN_IMAGE_WIDTH
    max_scroll_rounds = DOWNLOAD_JS_MAX_SCROLL_ROUNDS
    stable_rounds_required = DOWNLOAD_JS_STABLE_ROUNDS

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
            page.wait_for_timeout(DOWNLOAD_JS_SCROLL_WAIT_MS)
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
                # A real desktop browser's context, not automation defaults:
                # desktop UA + viewport, locale and timezone to match it.
                context = browser.new_context(
                    service_workers="block",
                    user_agent=REAL_USER_AGENT,
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                    timezone_id="America/New_York",
                )

                def guard_route(route):
                    if browser_request_allowed(route.request.url):
                        route.continue_()
                    else:
                        route.abort()

                context.route("**/*", guard_route)
                page = context.new_page()
                # Patch the headless tells (webdriver, plugins, languages,
                # vendor, WebGL) before any page script runs, so a bot check
                # sees a desktop Chrome instead of automation.
                Stealth(chrome_runtime=True).apply_stealth_sync(page)
                # route() does not intercept WebSocket handshakes; keep them disabled so a
                # page cannot open an SSRF channel the HTTP guard never sees.
                context.add_init_script(
                    "Object.defineProperty(globalThis, 'WebSocket', "
                    "{value: undefined, writable: false, configurable: false});"
                )
                page.goto(
                    chapter_url,
                    wait_until="domcontentloaded",
                    timeout=DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS,
                )
                # A bot challenge serves a near-empty page first and only swaps
                # in the real reader once its check clears; reading the images
                # immediately would return the challenge's (empty) set.
                try:
                    page.wait_for_function(
                        "([selector, minWidth]) => {"
                        "  for (const img of document.querySelectorAll(selector)) {"
                        "    if ((img.naturalWidth || 0) >= minWidth) return true;"
                        "  }"
                        "  return false;"
                        "}",
                        arg=[self.img_selector, self.min_width],
                        timeout=DOWNLOAD_JS_NAVIGATION_TIMEOUT_MS,
                    )
                except PlaywrightTimeoutError:
                    pass
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
