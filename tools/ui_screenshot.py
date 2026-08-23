import json
from playwright.sync_api import sync_playwright

MANIFEST = {"chapter_id":"probe_ch","pages":[{"source_page":0,"slice_index":0,"width":800,"height":600,
  "original":"/api/image/probe_ch/0/original","clean":"/api/image/probe_ch/0/clean","rendered":False,"skipped":False,"boxes":[],
  "text_objects":[{"id":"obj_1","shape":"rectangle","region":{"x1":100,"y1":100,"x2":300,"y2":200},"source_boxes":[0],
    "ocr_text":"original jp text","translation":"translated",
    "style":{"color":"auto","font":"default","fontSize":"auto","bold":False,"strokeWidth":"auto","strokeColor":"auto","bgColor":"transparent","cornerRadius":"0","horizontalAlign":"center","verticalAlign":"middle"}}]}]}

JS = r"""
window.fetch = (url, opts) => {
  const key = String(url);
  if (key.includes("/api/fonts")) return Promise.resolve(new Response(JSON.stringify([{id:"default",name:"Mặc định (Comic)"}]),{status:200,headers:{"Content-Type":"application/json"}}));
  const m = JSON.parse(JSON.stringify(currentManifest));
  return Promise.resolve(new Response(JSON.stringify(m),{status:200,headers:{"Content-Type":"application/json"}}));
};
window.currentChapterId = "probe_ch";
window.currentManifest = JSON.parse(JSON.stringify(%s));
availableFonts = [{id:"default",name:"Mặc định (Comic)"}];
"""
PNG = __import__("base64").b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 950})
    page.route("**/api/image/**", lambda r: r.fulfill(status=200, content_type="image/png", body=PNG))
    page.add_init_script(script=JS % json.dumps(MANIFEST))
    page.goto("http://127.0.0.1:8123/", wait_until="domcontentloaded")
    page.evaluate("renderEditor();")
    page.wait_for_timeout(600)
    page.click(".text-object-overlay >> nth=0")
    page.wait_for_timeout(200)
    page.screenshot(path="tools/ui_panel_default.png")
    page.click(".text-editor-section-header >> nth=1")
    page.click(".text-editor-section-header >> nth=2")
    page.wait_for_timeout(150)
    page.screenshot(path="tools/ui_panel_expanded.png")
    b.close()
print("screenshots saved")
