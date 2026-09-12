"""Exercise the real FastAPI server and browser runtime at desktop and mobile widths."""

from __future__ import annotations

import argparse
import json
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


def _exercise_landing(page: Page, base_url: str, name: str, artifacts: Path) -> None:
    page.goto(base_url, wait_until="networkidle")
    expect(page.locator("#home-view")).to_be_visible()
    expect(page.get_by_role("heading", name="Xin chào!")).to_be_visible()
    expect(page.locator(".recent-card")).to_have_count(1)
    expect(page.locator("#theme-select")).to_have_value("system")

    if name == "mobile":
        page.locator("#sidebar-toggle").click()
        expect(page.locator("body")).to_have_class("sidebar-open")
        page.locator('.sidebar-link[data-route="import"]').click()
    else:
        page.locator("#start-action").click()
    expect(page.locator("#import-view")).to_be_visible()
    expect(page.locator("#home-view")).to_be_hidden()
    expect(page.locator("#chapter-url")).to_be_focused()
    page.screenshot(path=str(artifacts / f"{name}-import.png"), full_page=True)

    if name == "mobile":
        page.locator("#sidebar-toggle").click()
        page.locator('.sidebar-link[data-route="home"]').click()
    else:
        page.locator('.sidebar-link[data-route="home"]').click()
        page.locator("#theme-select").select_option("dark")
        expect(page.locator("html")).to_have_attribute("data-resolved-theme", "dark")
        page.locator("#theme-select").select_option("light")
        expect(page.locator("html")).to_have_attribute("data-resolved-theme", "light")
        page.locator("#theme-select").select_option("system")
    expect(page.locator("#home-view")).to_be_visible()
    page.screenshot(path=str(artifacts / f"{name}-home.png"), full_page=True)


def _open_stage(page: Page, stage: str, mobile: bool = False) -> None:
    if mobile:
        page.locator("#sidebar-toggle").click()
        expect(page.locator("body")).to_have_class("sidebar-open")
    page.locator(f'.sidebar-link[data-stage="{stage}"]').click()
    expect(page.locator("body")).to_have_attribute("data-app-stage", stage)
    expect(page.locator("body")).not_to_have_class("sidebar-open")


def _exercise_desktop(page: Page) -> None:
    _wait_for_editor(page)
    _open_stage(page, "preview")
    expect(page.locator(".preview-workspace")).to_be_visible()
    expect(page.locator("#start-action.preview-primary-action")).to_be_visible()
    _open_stage(page, "review")
    expect(page.locator(".review-workspace-shell")).to_be_visible()
    _open_stage(page, "editor")
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

    # Repeated stage mounts used to retain Review observers and global input
    # handlers. Exercise the real renderer repeatedly and require one live
    # workspace before returning through the product action.
    page.evaluate("() => { for (let i = 0; i < 8; i += 1) window.renderReview(); }")
    expect(page.locator(".review-workspace-shell")).to_have_count(1)
    expect(page.locator(".review-card")).to_be_visible()
    page.locator(".review-primary-action").click()
    _wait_for_editor(page)


def _exercise_mobile(page: Page) -> None:
    _wait_for_editor(page)
    canvas_box = page.locator(".translation-canvas-host").bounding_box()
    if not canvas_box or canvas_box["height"] < 160:
        raise AssertionError(f"mobile editor canvas is not usable: {canvas_box}")
    expect(page.locator("#workbench-panel-controls")).to_be_visible()
    expect(page.locator(".page-navigator")).to_be_hidden()

    _open_stage(page, "preview", mobile=True)
    expect(page.locator(".preview-workspace")).to_be_visible()
    _open_stage(page, "review", mobile=True)
    expect(page.locator(".review-workspace-shell")).to_be_visible()
    _open_stage(page, "editor", mobile=True)
    _wait_for_editor(page)

    page.locator("#toggle-page-panel").click()
    expect(page.locator(".page-navigator")).to_be_visible()
    page.locator(".page-navigator-item").nth(1).click()
    page.wait_for_function(
        """() => document.querySelector('.translation-canvas-host .page-block')?.dataset.pageIndex === '1'"""
    )

    page.locator("#toggle-inspector-panel").click()
    expect(page.locator(".translation-panel-host")).to_be_visible()
    # The inspector is intentionally a modal-sized drawer on a narrow screen,
    # so close it before exercising a canvas click.  This confirms both the
    # drawer lifecycle and that the canvas remains responsive once uncovered.
    page.locator("#toggle-inspector-panel").click()
    expect(page.locator(".translation-panel-host")).to_be_hidden()
    page.locator(".translation-canvas-host img").click(position={"x": 20, "y": 20})
    expect(page.locator(".translation-canvas-host .page-block")).to_have_attribute(
        "data-page-index", "1"
    )


def _canvas_metrics(page: Page) -> dict[str, float | int]:
    return page.evaluate(
        """() => {
          const canvas = document.querySelector('.translation-canvas-host');
          const image = canvas?.querySelector('img');
          const rect = image?.getBoundingClientRect();
          return {
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
            bodyScrollWidth: document.documentElement.scrollWidth,
            canvasHeight: Math.round(canvas?.getBoundingClientRect().height || 0),
            canvasScrollLeft: Math.round(canvas?.scrollLeft || 0),
            canvasScrollTop: Math.round(canvas?.scrollTop || 0),
            imageLeft: Math.round(rect?.left || 0),
            imageTop: Math.round(rect?.top || 0),
            imageWidth: Math.round(rect?.width || 0),
            imageHeight: Math.round(rect?.height || 0),
          };
        }"""
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
                _exercise_landing(page, args.base_url.rstrip("/"), name, args.artifacts)
                page.goto(target, wait_until="networkidle")
                exercise(page)
                page.locator(".translation-canvas-host img").scroll_into_view_if_needed()
                metrics = _canvas_metrics(page)
                print(f"{name} canvas: {json.dumps(metrics, sort_keys=True)}")
                if metrics["imageWidth"] <= 0 or metrics["imageLeft"] >= metrics["viewportWidth"]:
                    raise AssertionError(f"{name} image is outside the active canvas: {metrics}")
                if name == "mobile" and metrics["imageWidth"] < metrics["viewportWidth"] * 0.6:
                    raise AssertionError(f"mobile image is unexpectedly collapsed: {metrics}")
                page.screenshot(path=str(args.artifacts / f"{name}.png"), full_page=True)
                print(f"{name}: PASS")
                page.close()
        finally:
            browser.close()

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main()
