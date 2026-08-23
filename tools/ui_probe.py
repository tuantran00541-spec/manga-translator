import json
from playwright.sync_api import sync_playwright

FAKE_MANIFEST = {
    "chapter_id": "probe_ch",
    "pages": [
        {
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
                }
            ],
        }
    ],
}

PROBE_JS = r"""
window.__probeCalls = [];
window.fetch = (url, opts) => {
  const key = String(url);
  if (key.includes("/api/text_object/") || key.includes("/api/render") || key.includes("/api/fonts")) {
    const body = opts && opts.body ? JSON.parse(opts.body) : {};
    window.__probeCalls.push({ url: key, body });
    if (key.includes("/api/fonts")) {
      return Promise.resolve(new Response(JSON.stringify([{ id: "default", name: "Mặc định (Comic)" }]), { status: 200, headers: { "Content-Type": "application/json" } }));
    }
    const m = JSON.parse(JSON.stringify(currentManifest));
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
      const obj = page.text_objects.find(o => o.id === body.id);
      if (obj) { obj.ocr_text = "ocr done"; obj.source_boxes = [0]; }
    }
    return Promise.resolve(new Response(JSON.stringify(m), { status: 200, headers: { "Content-Type": "application/json" } }));
  }
  return Promise.resolve(new Response("", { status: 400 }));
};
window.currentChapterId = "probe_ch";
window.currentManifest = JSON.parse(JSON.stringify(%s));
"""

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        page.goto("http://127.0.0.1:8123/", wait_until="domcontentloaded")
        page.add_script_tag(content=PROBE_JS % json.dumps(FAKE_MANIFEST))
        page.evaluate("renderEditor();")
        page.wait_for_timeout(400)
        structure = page.evaluate("""() => {
          const out = { classes: document.getElementById('page-view').className,
                        workspaces: document.querySelectorAll('.translation-workspace').length,
                        panelHosts: document.querySelectorAll('.translation-panel-host').length,
                        panels: document.querySelectorAll('.text-editor-panel').length,
                        overlays: document.querySelectorAll('.text-object-overlay').length,
                        sections: [...document.querySelectorAll('.text-editor-section')].map(s => ({ cls: s.className, title: s.querySelector('.text-editor-section-header span')?.textContent })),
                        actions: [...document.querySelectorAll('.text-object-action-btn')].map(b => b.textContent),
                        toolbarBtns: [...document.querySelectorAll('.editor-tool-btn')].map(b => b.textContent) };
          return out;
        }""")
        print(json.dumps({"structure": structure, "errors": errors}, ensure_ascii=False, indent=2))
        browser.close()

if __name__ == "__main__":
    main()
