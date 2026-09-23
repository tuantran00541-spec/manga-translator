"""Exercise the real FastAPI server and unified Review browser runtime."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, expect, sync_playwright


def _wait_for_review(page: Page) -> None:
    page.wait_for_selector(".review-workspace-shell.review-single-document")
    page.wait_for_selector(".review-document-shell")
    page.wait_for_function(
        """() => {
          const image = document.querySelector('.review-stitched-image');
          return image
            && Number(image.dataset.sourceWidth || 0) > 0
            && Number(image.dataset.sourceHeight || 0) > 0
            && image.querySelector('.review-strip-slice');
        }"""
    )
    expect(page.locator("body")).to_have_attribute("data-app-stage", "review")


def _wait_for_text_overlay(page: Page) -> None:
    page.wait_for_selector(".review-text-object-overlay")
    expect(page.locator(".review-text-object-overlay").first).to_be_visible()


def _review_layout_chain(page: Page) -> list[dict]:
    return page.evaluate(
        """() => {
          const viewport = document.querySelector('.review-document-viewport');
          const nodes = [];
          let node = viewport;
          while (node && nodes.length < 8) {
            const style = getComputedStyle(node);
            const rect = node.getBoundingClientRect();
            nodes.push({
              tag: node.tagName,
              id: node.id || '',
              cls: node.className || '',
              hidden: !!node.hidden,
              display: style.display,
              position: style.position,
              height: rect.height,
              width: rect.width,
              gridTemplateRows: style.gridTemplateRows,
              minHeight: style.minHeight,
              maxHeight: style.maxHeight,
              overflow: style.overflow,
            });
            node = node.parentElement;
          }
          return nodes;
        }"""
    )


def _exercise_landing(page: Page, base_url: str, name: str, artifacts: Path) -> None:
    page.goto(base_url, wait_until="networkidle")
    expect(page.locator("#home-view")).to_be_visible()
    expect(page.get_by_role("heading", name="Xin chào!")).to_be_visible()
    expect(page.locator('.recent-card[data-chapter-id="f00d0001"]')).to_have_count(1)
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


def _select_text_object(page: Page) -> None:
    _wait_for_text_overlay(page)
    overlay = page.locator(".review-text-object-overlay").first
    expect(overlay).to_be_visible()
    expect(page.locator(".review-inline-translation")).to_have_count(0)
    expect(page.locator(".review-inline-ocr")).to_have_count(0)
    overlay.click()


def _exercise_font_picker(page: Page) -> None:
    """Verify the live editor exposes the catalog and automatic matcher contract."""

    font_select = page.locator(".font-style-toolbar select")
    expect(font_select).to_have_count(1)
    expect(font_select.locator('option[value="auto"]')).to_have_count(1)
    expect(page.get_by_role("button", name="Gợi ý gần nhất", exact=True)).to_have_count(0)

    catalog = page.evaluate(
        """async () => {
          const response = await fetch('/api/fonts');
          const payload = await response.json();
          return {
            status: response.status,
            count: Array.isArray(payload) ? payload.length : 0,
            grouped: Array.isArray(payload) && payload.some((item) => item && item.category),
          };
        }"""
    )
    if catalog["status"] != 200 or catalog["count"] < 60 or not catalog["grouped"]:
        raise AssertionError(f"font catalog contract failed: {catalog}")

    match = page.evaluate(
        """async () => {
          const response = await fetch('/api/fonts/match', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              chapter_id: 'f00d0001',
              page_index: 0,
              object_id: 'bubble-a',
              top_k: 3,
            }),
          });
          const payload = await response.json();
          return {
            status: response.status,
            count: Array.isArray(payload.matches) ? payload.matches.length : 0,
            first: payload.matches?.[0]?.font_id || null,
          };
        }"""
    )
    if match["status"] != 200 or match["count"] != 3 or not match["first"]:
        raise AssertionError(f"font matcher contract failed: {match}")


def _exercise_desktop(page: Page) -> None:
    _wait_for_review(page)
    expect(page.locator(".chapter-translate-controls")).to_be_visible()
    viewport_box = page.locator(".review-document-viewport").bounding_box()
    if not viewport_box or viewport_box["height"] < 160:
        chain = _review_layout_chain(page)
        raise AssertionError(
            f"desktop Review canvas is not usable: {viewport_box}; chain={chain}"
        )
    # The fixture intentionally stores the legacy "editor" checkpoint. Opening
    # it must migrate into the single Review/lettering workspace.
    expect(page.locator('.sidebar-link[data-stage="editor"]')).to_have_count(0)
    expect(page.locator(".review-stitched-select")).to_have_count(0)
    image = page.locator(".review-stitched-image")
    expect(image).to_have_attribute("data-strip-slices", "3")
    expect(image).to_have_attribute("data-source-height", "4800")
    expect(image).to_have_attribute("data-source-width", "1200")
    first_image = page.locator('.review-strip-slice[data-page-index="0"] img')
    expect(first_image).to_have_js_property("naturalWidth", 1200)
    narrow_slice = page.locator('.review-strip-slice[data-page-index="2"]')
    narrow_slice.scroll_into_view_if_needed()
    expect(narrow_slice).to_have_js_property("clientWidth", 1000)
    expect(narrow_slice.locator("img")).to_have_js_property("naturalWidth", 1000)
    page.locator(".review-document-viewport").evaluate("element => element.scrollTop = 0")
    expect(page.locator('.review-text-object-overlay[data-page-index="0"]')).to_be_visible()
    expect(page.locator('.review-text-object-overlay[data-page-index="1"]')).to_be_visible()

    _select_text_object(page)
    expect(page.locator(".review-floating-inspector")).to_be_visible()
    _exercise_font_picker(page)
    expect(page.get_by_role("button", name="OCR toàn chương", exact=True)).to_be_visible()
    expect(page.locator("#site-header")).to_be_hidden()

    second = page.locator('.review-text-object-overlay[data-page-index="1"]').first
    second.click()
    translation = page.locator(".review-floating-inspector .translation-textarea").first
    expect(translation).to_have_value("Bong bóng thứ hai")

    page.get_by_role("button", name="Original", exact=True).click()
    page.wait_for_function(
        "() => document.querySelector('.review-workspace-shell')?.classList.contains('review-readonly-document')"
    )
    page.get_by_role("button", name="Clean", exact=True).click()
    page.wait_for_function(
        "() => !document.querySelector('.review-workspace-shell')?.classList.contains('review-readonly-document')"
    )
    _wait_for_text_overlay(page)

    page.get_by_role("button", name="100%", exact=True).click()
    expect(page.locator(".review-zoom-value")).to_have_text("100%")

    expect(page.locator(".review-tool-rail")).to_be_visible()
    expect(page.locator('.sidebar-link[data-route="home"]')).to_be_visible()
    expect(page.locator('.sidebar-link[data-stage="preview"]')).to_be_visible()
    expect(page.locator('.sidebar-link[data-stage="preview"]')).to_be_enabled()

    toolbar_overflow = page.evaluate(
        """() => {
          const bar = document.querySelector('.review-document-toolbar-compact');
          if (!bar) return [{ missing: true }];
          const frame = bar.getBoundingClientRect();
          return [...bar.querySelectorAll('button,select,input')]
            .map((el) => {
              const r = el.getBoundingClientRect();
              return {
                tag: el.tagName,
                cls: el.className,
                text: (el.textContent || el.value || "").trim().slice(0, 80),
                left: r.left,
                right: r.right,
                top: r.top,
                bottom: r.bottom,
                width: r.width,
                height: r.height,
                frameLeft: frame.left,
                frameRight: frame.right,
                frameTop: frame.top,
                frameBottom: frame.bottom,
              };
            })
            .filter((r) => r.width > 0 && r.height > 0)
            .filter((r) => r.left < r.frameLeft - 1
              || r.right > r.frameRight + 1
              || r.top < r.frameTop - 1
              || r.bottom > r.frameBottom + 1);
        }"""
    )
    if toolbar_overflow:
        raise AssertionError(
            "desktop Review toolbar controls overflow their frame: "
            + repr(toolbar_overflow)
        )

    page.locator('.sidebar-link[data-route="home"]').click()
    expect(page.locator("#home-view")).to_be_visible()
    page.locator('.sidebar-link[data-stage="review"]').click()
    nav_state = page.evaluate(
        """() => {
          const pv = document.getElementById('page-view');
          const landing = document.getElementById('landing-view');
          const ws = document.querySelector('.review-workspace-shell');
          const chain = [];
          let node = ws;
          while (node && chain.length < 6) {
            const style = getComputedStyle(node);
            chain.push({
              tag: node.tagName,
              id: node.id || '',
              cls: node.className || '',
              hidden: !!node.hidden,
              display: style.display,
              visibility: style.visibility,
              width: node.getBoundingClientRect().width,
              height: node.getBoundingClientRect().height,
            });
            node = node.parentElement;
          }
          return {
            stage: document.body.dataset.appStage || '',
            hash: location.hash,
            pageViewHidden: !!pv?.hidden,
            pageViewDisplay: pv ? getComputedStyle(pv).display : null,
            landingHidden: !!landing?.hidden,
            workspaceConnected: !!ws?.isConnected,
            chain,
          };
        }"""
    )
    print("desktop navigation:", json.dumps(nav_state, sort_keys=True))
    _wait_for_review(page)

    # Repeated mounts must not retain duplicate observers/workspaces.
    page.evaluate("() => { for (let i = 0; i < 8; i += 1) window.renderReview(); }")
    _wait_for_review(page)
    expect(page.locator(".review-workspace-shell")).to_have_count(1)
    expect(page.locator(".review-document-shell")).to_have_count(1)
    expect(page.locator(".review-primary-action")).to_have_count(0)


def _exercise_mobile(page: Page) -> None:
    _wait_for_review(page)
    viewport_box = page.locator(".review-document-viewport").bounding_box()
    if not viewport_box or viewport_box["height"] < 160:
        chain = _review_layout_chain(page)
        raise AssertionError(
            f"mobile Review canvas is not usable: {viewport_box}; chain={chain}"
        )

    expect(page.locator("#workbench-panel-controls")).to_be_hidden()
    expect(page.locator("#site-header")).to_be_hidden()
    expect(page.locator(".page-navigator")).to_have_count(0)
    expect(page.locator(".review-stitched-select")).to_have_count(0)
    image = page.locator(".review-stitched-image")
    expect(image).to_have_attribute("data-strip-slices", "3")
    expect(image).to_have_attribute("data-source-height", "4800")

    second = page.locator('.review-text-object-overlay[data-page-index="1"]').first
    expect(second).to_be_visible()
    second.click()
    expect(page.locator(".review-floating-inspector")).to_be_visible()
    _exercise_font_picker(page)
    translation = page.locator(".review-floating-inspector .translation-textarea").first
    expect(translation).to_be_visible()
    expect(translation).to_have_value("Bong bóng thứ hai")
    expect(page.get_by_role("button", name="OCR toàn chương", exact=True)).to_be_visible()

    expect(page.locator(".review-tool-rail")).to_be_visible()
    expect(page.locator('.sidebar-link[data-route="home"]')).to_be_visible()
    expect(page.locator('.sidebar-link[data-stage="preview"]')).to_be_visible()
    expect(page.locator('.sidebar-link[data-stage="preview"]')).to_be_enabled()

    toolbar_overflow = page.evaluate(
        """() => {
          const bar = document.querySelector('.review-document-toolbar-compact');
          if (!bar) return [{ missing: true }];
          const frame = bar.getBoundingClientRect();
          return [...bar.querySelectorAll('button,select,input')]
            .map((el) => {
              const r = el.getBoundingClientRect();
              return {
                tag: el.tagName,
                cls: el.className,
                text: (el.textContent || el.value || "").trim().slice(0, 80),
                left: r.left,
                right: r.right,
                top: r.top,
                bottom: r.bottom,
                width: r.width,
                height: r.height,
                frameLeft: frame.left,
                frameRight: frame.right,
                frameTop: frame.top,
                frameBottom: frame.bottom,
              };
            })
            .filter((r) => r.width > 0 && r.height > 0)
            .filter((r) => r.left < r.frameLeft - 1
              || r.right > r.frameRight + 1
              || r.top < r.frameTop - 1
              || r.bottom > r.frameBottom + 1);
        }"""
    )
    if toolbar_overflow:
        raise AssertionError(
            "mobile Review toolbar controls overflow their frame: "
            + repr(toolbar_overflow)
        )


def _canvas_metrics(page: Page) -> dict[str, float | int]:
    return page.evaluate(
        """() => {
          const viewport = document.querySelector('.review-document-viewport');
          const image = document.querySelector('.review-stitched-image');
          const rect = image?.getBoundingClientRect();
          return {
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
            bodyScrollWidth: document.documentElement.scrollWidth,
            canvasHeight: Math.round(viewport?.getBoundingClientRect().height || 0),
            canvasScrollLeft: Math.round(viewport?.scrollLeft || 0),
            canvasScrollTop: Math.round(viewport?.scrollTop || 0),
            imageLeft: Math.round(rect?.left || 0),
            imageTop: Math.round(rect?.top || 0),
            imageWidth: Math.round(rect?.width || 0),
            imageHeight: Math.round(rect?.height || 0),
          };
        }"""
    )


def _exercise_long_image(page: Page, base_url: str, artifacts: Path, name: str) -> None:
    page.goto(f"{base_url}/#f00d0042", wait_until="networkidle")
    page.reload(wait_until="networkidle")
    _wait_for_review(page)
    page.wait_for_function("() => !!document.querySelector('.review-strip-slice,.review-stitched-error')")
    error = page.locator(".review-stitched-error")
    if error.count():
        raise AssertionError(f"{name}: {error.inner_text()}")
    image = page.locator('.review-strip-slice[data-page-index="0"] img')
    if not image.count():
        state = page.evaluate("""async () => {
          const url = window.currentManifest?.pages?.[0]?.clean;
          const response = url ? await fetch(url) : null;
          return {url, status: response?.status, type: response?.headers.get('content-type'),
            slice: document.querySelector('.review-strip-slice')?.textContent};
        }""")
        raise AssertionError(f"{name}: long image element missing: {state}")
    expect(page.locator('.review-stitched-image')).to_have_attribute("data-strip-slices", "1")
    expect(page.locator('.review-stitched-image')).to_have_attribute("data-source-height", "41000")
    expect(image).to_have_js_property("naturalWidth", 1200)
    expect(image).to_have_js_property("naturalHeight", 41000)
    viewport = page.locator('.review-document-viewport')

    def has_band(color: str, output: Path) -> None:
        shot = viewport.screenshot(path=str(output))
        with Image.open(BytesIO(shot)) as screenshot:
            pixels = screenshot.convert("RGB")
            x = pixels.width // 2
            visible = [pixels.getpixel((x, y)) for y in range(pixels.height)]
        if color == "red":
            found = any(r > 170 and g < 90 and b < 90 for r, g, b in visible)
        else:
            found = any(b > 170 and r < 90 and g < 110 for r, g, b in visible)
        if not found:
            raise AssertionError(f"{name}: long image {color} band is absent from visible canvas")

    has_band("red", artifacts / f"{name}-long-top.png")
    viewport.evaluate("element => element.scrollTop = element.scrollHeight")
    page.wait_for_function(
        "() => { const v = document.querySelector('.review-document-viewport'); return v.scrollTop > 10000 && v.scrollTop + v.clientHeight >= v.scrollHeight - 2; }"
    )
    has_band("blue", artifacts / f"{name}-long-bottom.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/ui-smoke"))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
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
                page.locator('.recent-card[data-chapter-id="f00d0001"]').click()
                _wait_for_review(page)
                exercise(page)
                metrics = _canvas_metrics(page)
                print(f"{name} canvas: {json.dumps(metrics, sort_keys=True)}")
                if metrics["imageWidth"] <= 0 or metrics["imageLeft"] >= metrics["viewportWidth"]:
                    raise AssertionError(f"{name} image is outside the active canvas: {metrics}")
                if name == "mobile" and metrics["imageWidth"] < metrics["viewportWidth"] * 0.6:
                    raise AssertionError(f"mobile image is unexpectedly collapsed: {metrics}")
                page.screenshot(path=str(args.artifacts / f"{name}.png"), full_page=True)

                # Hard reload of the chapter hash must restore the unified Review.
                page.reload(wait_until="networkidle")
                _wait_for_review(page)
                expect(page.locator("body")).to_have_attribute("data-app-stage", "review")
                _exercise_long_image(page, args.base_url.rstrip("/"), args.artifacts, name)
                print(f"{name}: PASS")
                page.close()
        finally:
            browser.close()

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main()
