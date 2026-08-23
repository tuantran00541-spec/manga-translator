import json
import sys

from playwright.sync_api import sync_playwright

PAGE_0 = {
    "source_page": 0, "slice_index": 0, "width": 800, "height": 600,
    "original": "/api/image/probe_ch/0/original", "clean": "/api/image/probe_ch/0/clean",
    "rendered": False, "skipped": False, "boxes": [],
    "text_objects": [
        {
            "id": "obj_1", "shape": "rectangle",
            "region": {"x1": 100, "y1": 100, "x2": 300, "y2": 200},
            "source_boxes": [0], "ocr_text": "original jp text", "translation": "translated",
            "style": {"color": "auto", "font": "default", "fontSize": "auto", "bold": False,
                      "strokeWidth": "auto", "strokeColor": "auto", "bgColor": "transparent",
                      "cornerRadius": "0", "horizontalAlign": "center", "verticalAlign": "middle"},
        },
        {
            "id": "obj_2", "shape": "ellipse",
            "region": {"x1": 400, "y1": 300, "x2": 600, "y2": 400},
            "source_boxes": [], "ocr_text": "second", "translation": "hai",
            "style": {"color": "#000000", "font": "default", "fontSize": "22", "bold": True,
                      "strokeWidth": "3", "strokeColor": "#000000", "bgColor": "#ffffff",
                      "cornerRadius": "4", "horizontalAlign": "left", "verticalAlign": "top"},
        },
    ],
}

FAKE_MANIFEST = {"chapter_id": "probe_ch", "pages": [PAGE_0]}

BOOTSTRAP = r"""
window.__probeCalls = [];
window.__failNextUpdate = false;
window.__serverManifest = JSON.parse(JSON.stringify(%s));
window.fetch = (url, opts) => {
  const key = String(url);
  if (key.includes("/api/fonts")) {
    return Promise.resolve(new Response(JSON.stringify([{ id: "default", name: "Mặc định (Comic)" }, { id: "shadow", name: "Shadow" }]), { status: 200, headers: { "Content-Type": "application/json" } }));
  }
  if (key.includes("/api/text_object/") || key.includes("/api/render")) {
    const body = opts && opts.body ? JSON.parse(opts.body) : {};
    window.__probeCalls.push({ url: key, body });
    if (key.includes("/api/text_object/update") && window.__failNextUpdate) {
      window.__failNextUpdate = false;
      return Promise.resolve(new Response(JSON.stringify({ detail: "boom" }), { status: 500, headers: { "Content-Type": "application/json" } }));
    }
    const m = JSON.parse(JSON.stringify(window.__serverManifest));
    const page = m.pages[body.page_index];
    if (key.includes("/api/text_object/create")) {
      const created = { id: "new_" + window.__probeCalls.length, shape: body.shape || "rectangle", region: body.region, source_boxes: [], ocr_text: "", translation: "", style: Object.assign({}, window.DEFAULT_TEXT_OBJECT_STYLE) };
      page.text_objects.push(created);
    } else if (key.includes("/api/text_object/delete")) {
      page.text_objects = page.text_objects.filter(o => o.id !== body.id);
    } else if (key.includes("/api/text_object/update")) {
      const obj = page.text_objects.find(o => o.id === body.id);
      if (obj) {
        if (body.region) obj.region = body.region;
        if (body.ocr_text !== undefined) obj.ocr_text = body.ocr_text;
        if (body.translation !== undefined) obj.translation = body.translation;
        if (body.style) obj.style = Object.assign({}, obj.style || {}, body.style);
      }
    } else if (key.includes("/api/text_object/ocr")) {
      const delay = window.__ocrDelay || 0;
      return new Promise(resolve => setTimeout(() => {
        const m2 = JSON.parse(JSON.stringify(window.__serverManifest));
        const obj = m2.pages[body.page_index].text_objects.find(o => o.id === body.id);
        if (obj) { obj.ocr_text = "ocr done"; obj.source_boxes = [0]; }
        window.__serverManifest = JSON.parse(JSON.stringify(m2));
        resolve(new Response(JSON.stringify(m2), { status: 200, headers: { "Content-Type": "application/json" } }));
      }, delay));
    } else if (key.includes("/api/render")) {
      return Promise.resolve(new Response(JSON.stringify({ output: "/api/image/probe_ch/0/rendered" }), { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    window.__serverManifest = JSON.parse(JSON.stringify(m));
    return Promise.resolve(new Response(JSON.stringify(m), { status: 200, headers: { "Content-Type": "application/json" } }));
  }
  return Promise.resolve(new Response("", { status: 404 }));
};
window.currentChapterId = "probe_ch";
window.currentManifest = JSON.parse(JSON.stringify(window.__serverManifest));
availableFonts = [{ id: "default", name: "Mặc định (Comic)" }, { id: "shadow", name: "Shadow" }];
"""

PNG_BYTES = __import__("base64").b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

PASS = 0
FAIL = 0
RESULTS = []


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")
    RESULTS.append((name, ok, detail))


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        page.route("**/api/image/**", lambda route: route.fulfill(status=200, content_type="image/png", body=PNG_BYTES))
        page.add_init_script(script=BOOTSTRAP % json.dumps(FAKE_MANIFEST))
        page.goto("http://127.0.0.1:8123/", wait_until="domcontentloaded")
        page.evaluate("renderEditor();")
        page.wait_for_timeout(500)

        def js(expr):
            return page.evaluate(expr)

        def calls(action):
            return js(f"window.__probeCalls.filter(c => c.url.includes('/api/text_object/{action}')).length")

        def obj_count():
            return js("currentManifest.pages[0].text_objects.length")

        print("UI-08: Properties panel organization")
        check("workspace renders", js("!!document.querySelector('.translation-workspace')"))
        check("panel host renders", js("!!document.querySelector('.translation-panel-host')"))
        page.wait_for_selector(".text-object-overlay", timeout=5000)
        check("overlays render over image", js("document.querySelectorAll('.text-object-overlay').length") == 2)

        page.click(".text-object-overlay >> nth=0")
        page.wait_for_timeout(100)
        check("click selects object", js("editorState.selectedTextObjectId") == "obj_1")
        check("properties shows selected object", js("document.querySelector('.text-editor-panel').dataset.objectId") == "obj_1")

        sections = js("[...document.querySelectorAll('.text-editor-section')].map(s => ({ title: s.querySelector('.text-editor-section-header span').textContent, open: s.classList.contains('open') }))")
        check("TEXT section exists", any(s["title"] == "Văn bản" for s in sections), str(sections))
        check("APPEARANCE section exists", any(s["title"] == "Kiểu dáng" for s in sections), str(sections))
        check("BACKGROUND section exists", any(s["title"] == "Nền" for s in sections), str(sections))
        text_open = next((s for s in sections if s["title"] == "Văn bản"), {}).get("open")
        appearance_open = next((s for s in sections if s["title"] == "Kiểu dáng"), {}).get("open")
        background_open = next((s for s in sections if s["title"] == "Nền"), {}).get("open")
        check("TEXT expanded by default", text_open is True, str(sections))
        check("APPEARANCE collapsed by default", appearance_open is False, str(sections))
        check("BACKGROUND collapsed by default", background_open is False, str(sections))
        page.click(".text-editor-section-header >> nth=1")
        check("APPEARANCE toggles open", js("document.querySelectorAll('.text-editor-section')[1].classList.contains('open')"))
        page.select_option(".font-family-select", "shadow")
        page.evaluate("flushTextObjectPersist()")
        page.wait_for_timeout(150)
        check("font change updates style", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.font") == "shadow")
        check("font change updates dataset", js("document.querySelector('.text-editor-panel').dataset.font") == "shadow")
        check("font change persists via canonical path", calls("update") >= 1)
        page.click(".bold-toggle-btn")
        check("bold toggle updates style", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.bold") is True)
        page.click(".size-auto-btn")
        check("font size leaves auto", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.fontSize") != "auto")
        page.click(".align-btn >> nth=2")
        check("horizontal align updates", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.horizontalAlign") == "right")
        page.click(".color-btn >> nth=1")
        check("text color updates", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.color") == "#ffffff")
        check("no stroke style control", js("document.querySelectorAll('.stroke-style-select').length") == 0)
        page.click(".text-editor-section-header >> nth=2")
        page.check(".bg-toggle-checkbox")
        check("background enable sets color", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.bgColor") == "#ffffff")
        page.uncheck(".bg-toggle-checkbox")
        check("background disable sets transparent", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style.bgColor") == "transparent")

        before_b = js("JSON.stringify(currentManifest.pages[0].text_objects.find(o=>o.id==='obj_2').style)")
        check("object B unchanged after A edits", js("JSON.stringify(currentManifest.pages[0].text_objects.find(o=>o.id==='obj_2').style)") == before_b)
        check("obj_1 region unchanged by style edits", js("JSON.stringify(currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').region)") == '{"x1":100,"y1":100,"x2":300,"y2":200}')

        print("UI-09: Editing / selection")
        page.dblclick(".text-object-overlay >> nth=0")
        focused = js("document.activeElement.className")
        check("double-click focuses translation textarea", "translation-textarea" in focused, focused)
        before_val = js("document.querySelector('.translation-textarea').value")
        page.keyboard.press("ArrowLeft")
        page.keyboard.type("X")
        after_val = js("document.querySelector('.translation-textarea').value")
        expected = before_val[:-1] + "X" + before_val[-1:]
        check("typing edits text", after_val == expected, f"{before_val} -> {after_val} (expected {expected})")
        check("arrow keys behave as text nav", js("document.activeElement === document.querySelector('.translation-textarea')"))
        page.keyboard.press("Delete")
        check("Delete edits text not object", js("currentManifest.pages[0].text_objects.length") == 2 and calls("delete") == 0)
        page.keyboard.press("Backspace")
        check("Backspace edits text not object", calls("delete") == 0)

        print("UI-08/09: Persistence failure handling")
        js("window.__failNextUpdate = true")
        page.fill(".translation-textarea", "will fail")
        page.wait_for_timeout(950)
        check("failed save shows error toast", js("!!document.querySelector('.toast-error')"))
        check("failed save keeps pending state", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').translation") == "will fail")
        before_updates = calls("update")
        page.evaluate("flushTextObjectPersist()")
        page.wait_for_timeout(200)
        check("failed save is retried on next flush", calls("update") > before_updates)

        print("UI-08: Reload persistence")
        persisted_style = js("JSON.stringify(window.__serverManifest.pages[0].text_objects.find(o=>o.id==='obj_1').style)")
        check("style persisted to server (survives reload)", '"font":"shadow"' in persisted_style, persisted_style)

        print("UI-09: Duplicate")
        page.click(".text-object-action-btn >> nth=1")
        page.wait_for_timeout(200)
        check("duplicate creates new object", obj_count() == 3)
        dup_id = js("editorState.selectedTextObjectId")
        check("duplicate has new id", dup_id.startswith("new_"), dup_id)
        dup = js(f"currentManifest.pages[0].text_objects.find(o=>o.id==='{dup_id}')")
        check("duplicate preserves translation (effective state)", dup["translation"] == "will fail", json.dumps(dup))
        check("duplicate preserves ocr text", dup["ocr_text"] == "original jp text")
        check("duplicate preserves style", dup["style"]["font"] == "shadow")
        check("duplicate copies no OCR association", dup["source_boxes"] == [], str(dup.get("source_boxes")))
        r = dup["region"]
        check("duplicate offset inside image", r["x1"] == 124 and r["y1"] == 124, json.dumps(r))
        check("duplicate preserves size", (r["x2"] - r["x1"]) == 200 and (r["y2"] - r["y1"]) == 100)
        check("duplicate is selected", js("editorState.selectedTextObjectId") == dup_id)
        page.fill(".translation-textarea", "dup edited")
        page.wait_for_timeout(50)
        check("duplicate edits independently", js(f"currentManifest.pages[0].text_objects.find(o=>o.id==='{dup_id}').translation") == "dup edited")
        check("original translation untouched by dup edit", js("currentManifest.pages[0].text_objects.find(o=>o.id==='obj_1').translation") == "will fail")

        print("UI-09: Add Box")
        before_count = obj_count()
        page.click(".editor-tool-btn >> text=Thêm ô chữ")
        page.wait_for_timeout(200)
        check("add box creates object", obj_count() == before_count + 1)
        box_id = js("editorState.selectedTextObjectId")
        box = js(f"currentManifest.pages[0].text_objects.find(o=>o.id==='{box_id}')")
        check("add box selects new object", box_id.startswith("new_"), box_id)
        check("add box valid placement inside image", 0 <= box["region"]["x1"] and box["region"]["x2"] <= 800 and 0 <= box["region"]["y1"] and box["region"]["y2"] <= 600)
        check("add box respects min size", (box["region"]["x2"] - box["region"]["x1"]) >= 10 and (box["region"]["y2"] - box["region"]["y1"]) >= 10)
        check("add box persists via create", calls("create") >= 1)
        check("add box opens properties", js("document.querySelector('.text-editor-panel').dataset.objectId") == box_id)

        print("UI-09: Delete")
        page.click(".text-object-action-btn >> nth=2")
        check("delete requires second click", obj_count() == before_count + 1)
        page.click(".text-object-action-btn >> nth=2")
        page.wait_for_timeout(200)
        check("delete removes object", obj_count() == before_count)
        check("delete clears selection", js("editorState.selectedTextObjectId") is None)
        check("delete persisted", calls("delete") >= 1)

        print("UI-09: Pending-state safety")
        before_pending = obj_count()
        page.click(".text-object-overlay >> nth=0")
        page.wait_for_timeout(50)
        page.fill(".translation-textarea", "pending text")
        page.wait_for_timeout(50)
        page.click(".text-object-action-btn >> nth=2")
        page.click(".text-object-action-btn >> nth=2")
        page.wait_for_timeout(300)
        gone = js(f"!currentManifest.pages[0].text_objects.some(o => o.id === 'obj_1')")
        check("delete after pending mutation removes object", gone)
        check("no resurrection after delete", obj_count() == before_pending - 1)

        print("UI-09: OCR stale-response safety")
        js("window.__ocrDelay = 300")
        page.click(".text-object-overlay >> nth=0")
        page.wait_for_timeout(50)
        page.click(".text-object-action-btn >> text=Nhóm OCR lại")
        page.wait_for_timeout(50)
        page.click(".text-object-action-btn >> nth=2")
        page.click(".text-object-action-btn >> nth=2")
        page.wait_for_timeout(600)
        js("window.__ocrDelay = 0")
        check("stale OCR response does not resurrect deleted object", js("!currentManifest.pages[0].text_objects.some(o => o.id === 'obj_2')"))

        page.click(".text-object-overlay >> nth=0")
        page.wait_for_timeout(50)
        js("document.activeElement && document.activeElement.blur()")
        count_before_kb = obj_count()
        page.keyboard.press("Delete")
        page.wait_for_timeout(200)
        check("keyboard Delete removes selected object when not editing", obj_count() == count_before_kb - 1)

        print("UI-08: Render flow")
        page.click(".editor-render-btn")
        page.wait_for_timeout(300)
        check("render result renders in panel", js("!!document.querySelector('.translation-panel-host .render-result img')"))

        real_errors = [e for e in errors if "Failed to load resource" not in e]
        check("no console errors in affected flows", len(real_errors) == 0, str(real_errors))

        browser.close()

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
