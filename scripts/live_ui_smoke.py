"""Exercise the real FastAPI server and browser runtime at desktop and mobile widths."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

from create_ui_smoke_fixture import CHAPTER_ID


def _wait_for_editor(page: Page) -> None:
    page.wait_for_selector(".translation-workspace")
    page.wait_for_function(
        """() => {
          const image = document.querySelector('.translation-canvas-host img');
          return image && image.naturalWidth > 0 && image.naturalHeight > 0;
        }"""
    )
    expect(page.locator(".text-object-overlay").first).to_be_visible()


def _exercise_desktop(page: Page) -> None:
    _wait_for_editor(page)
    page.locator(".text-object-overlay").first.click()
    expect(page.locator(".translation-textarea")).to_be_visible()

    page.locator(".page-navigator-item").nth(1).click()
    page.wait_for_function(
        """() => document.querySelector('.translation-canvas-host .page-block')?.dataset.pageIndex === '1'"""
    )
    page.locator(".translation-canvas-host img").click(position={"x": 20, "y": 20})
    expect(page.locator(".translation-canvas-host .page-block")).to_have_attribute(
        "data-page-index", "1"
    )


def _exercise_mobile(page: Page) -> None:
    _wait_for_editor(page)
    expect(page.locator("#workbench-panel-controls")).to_be_visible()
    expect(page.locator(".page-navigator")).to_be_hidden()

    page.locator("#toggle-page-panel").click()
    expect(page.locator(".page-navigator")).to_be_visible()
    page.locator(".page-navigator-item").nth(1).click()
    page.wait_for_function(
        """() => document.querySelector('.translation-canvas-host .page-block')?.dataset.pageIndex === '1'"""
    )

    page.locator("#toggle-inspector-panel").click()
    expect(page.locator(".translation-panel-host")).to_be_visible()
    page.locator(".translation-canvas-host img").click(position={"x": 20, "y": 20})
    expect(page.locator(".translation-canvas-host .page-block")).to_have_attribute(
        "data-page-index", "1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/ui-smoke"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    target = f"{args.base_url.rstrip('/')}/#{CHAPTER_ID}"
    failures: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for name, viewport, exercise in (
                ("desktop", {"width": 1440, "height": 960}, _exercise_desktop),
                ("mobile", {"width": 390, "height": 844}, _exercise_mobile),
            ):
                page = browser.new_page(viewport=viewport, device_scale_factor=1)
                page.on("pageerror", lambda exc, n=name: failures.append(f"{n} page error: {exc}"))
                page.on(
                    "console",
                    lambda message, n=name: failures.append(f"{n} console error: {message.text}")
                    if message.type == "error"
                    else None,
                )
                page.goto(target, wait_until="networkidle")
                exercise(page)
                page.screenshot(path=str(args.artifacts / f"{name}.png"), full_page=True)
                print(f"{name}: PASS")
                page.close()
        finally:
            browser.close()

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main()
