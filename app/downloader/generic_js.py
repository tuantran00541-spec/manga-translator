from playwright.sync_api import sync_playwright
from app.downloader.base import BaseAdapter


class GenericJsAdapter(BaseAdapter):
    img_selector = "img"
    min_width = 400

    def can_handle(self, url: str) -> bool:
        return True

    def extract_image_urls(self, chapter_url: str) -> list[str]:
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
            page = browser.new_page()
            page.goto(chapter_url, wait_until="networkidle", timeout=60000)
            page.mouse.wheel(0, 20000)
            page.wait_for_timeout(1500)
            elements = page.query_selector_all(self.img_selector)
            for el in elements:
                box = el.bounding_box()
                if box and box["width"] < self.min_width:
                    continue
                src = el.get_attribute("src") or el.get_attribute("data-src")
                if src:
                    urls.append(src)
            browser.close()
        return self._dedupe(urls)

    @staticmethod
    def _dedupe(urls: list[str]) -> list[str]:
        seen = set()
        result = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                result.append(u)
        return result
