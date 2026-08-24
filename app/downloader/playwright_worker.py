import sys

import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.downloader.http import safe_download_file
from app.security import browser_request_allowed, validate_url

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
MIN_WIDTH = 300


def guess_ext(url: str) -> str:
    clean = url.lower().split("?")[0].split("#")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if clean.endswith(ext):
            return ext
    return ".jpg"


def dedupe(urls: list) -> list:
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def main():
    if len(sys.argv) < 3:
        print("[]")
        return

    chapter_url = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        validate_url(chapter_url)
    except Exception:
        print("[]")
        return

    from playwright.sync_api import sync_playwright

    saved = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            service_workers="block",
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
        context.add_init_script(
            "Object.defineProperty(globalThis, 'WebSocket', "
            "{value: undefined, writable: false, configurable: false});"
        )

        def guard_route(route):
            if browser_request_allowed(route.request.url):
                route.continue_()
            else:
                route.abort()

        context.route("**/*", guard_route)
        page = context.new_page()

        try:
            try:
                page.goto(chapter_url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                print(f"[Worker] Warning loading page: {e}", file=sys.stderr)

            for _ in range(6):
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(600)

            elements = page.query_selector_all("img")
            image_urls = []
            for el in elements:
                box = el.bounding_box()
                if box and box["width"] < MIN_WIDTH:
                    continue
                src = (
                    el.get_attribute("data-src")
                    or el.get_attribute("data-original")
                    or el.get_attribute("data-lazy")
                    or el.get_attribute("src")
                )
                if src and src.startswith("http") and browser_request_allowed(src):
                    image_urls.append(src)

            image_urls = dedupe(image_urls)

            for i, img_url in enumerate(image_urls):
                ext = guess_ext(img_url)
                out_path = output_dir / f"{i:03d}{ext}"
                try:
                    safe_download_file(
                        img_url,
                        out_path,
                        headers={"User-Agent": USER_AGENT, "Referer": chapter_url},
                        timeout=30,
                    )
                    saved.append(str(out_path))
                except Exception:
                    pass
        finally:
            browser.close()

    print(json.dumps(saved))


if __name__ == "__main__":
    main()
