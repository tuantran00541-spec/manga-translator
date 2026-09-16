(() => {
  "use strict";
  const S = { zoom: "fit", variant: "clean", peek: null, space: false, sync: false };
  const steps = [25, 50, 75, 100, 125, 150, 200, 300, 400];
  const $ = (q, r = document) => r.querySelector(q);
  const $$ = (q, r = document) => [...r.querySelectorAll(q)];
  const click = (q) => { const n = $(q); if (!n || n.disabled) return false; n.click(); return true; };
  const button = (text, cls = "") => Object.assign(document.createElement("button"), { type: "button", className: cls, textContent: text });

  function activeReview() {
    const card = $("#page-view .review-canvas-host .review-card");
    const index = Number.parseInt(card?.dataset.pageIndex || "", 10);
    return { card, index, page: Number.isInteger(index) ? window.currentManifest?.pages?.[index] : null };
  }
  function imageUrl(page, variant) {
    if (!page) return null;
    if (variant === "original") return page.original || page.clean || null;
    return typeof window.pageImageUrl === "function" ? window.pageImageUrl(page) : (page.clean || page.original || null);
  }
  function setReadonly(on) {
    const ws = $("#page-view .review-workspace-shell");
    if (!ws) return;
    ws.classList.toggle("pro-review-original", on);
    if (on) {
      const canvas = $("canvas.brush-canvas", ws);
      if (typeof canvas?._stopBrush === "function") canvas._stopBrush();
    }
    $$(".brush-toggle-btn,.clear-brush-btn,.repaint-btn,.reset-manual-btn,.ai-qc-btn,.brush-size-slider", ws).forEach((c) => {
      if (on) { c.dataset.proDisabled ??= c.disabled ? "1" : "0"; c.disabled = true; }
      else if (c.dataset.proDisabled !== undefined) {
        const old = c.dataset.proDisabled === "1"; delete c.dataset.proDisabled;
        c.disabled = old || ws.classList.contains("review-busy") || ws.classList.contains("review-chapter-qc-running");
      }
    });
  }
  function badge(variant) {
    const wrap = activeReview().card?.querySelector(".review-image-wrap");
    if (!wrap) return;
    let n = $(".pro-canvas-badge", wrap);
    if (!n) { n = document.createElement("span"); n.className = "pro-canvas-badge"; wrap.append(n); }
    n.textContent = variant === "original" ? "ORIGINAL" : "CLEAN";
    n.dataset.variant = variant;
  }
  function syncVariantButtons() {
    $$('[data-pro-review-variant]').forEach((b) => {
      const on = b.dataset.proReviewVariant === S.variant;
      b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on));
    });
  }
  function showVariant(variant, transient = false) {
    const { card, page } = activeReview();
    const img = card?.querySelector(".review-image-wrap img");
    const url = imageUrl(page, variant);
    if (!img || !url) return false;
    if (!transient) S.variant = variant;
    img.dataset.proViewVariant = variant;
    const absolute = new URL(url, location.href).href;
    if (img.src !== absolute) img.src = url;
    setReadonly(variant === "original"); badge(variant);
    if (!transient) syncVariantButtons();
    setTimeout(() => zoom(S.zoom), 0); syncStatus(); return true;
  }
  function peekStart() { if (document.body.dataset.appStage === "review" && S.peek === null) { S.peek = S.variant; showVariant("original", true); } }
  function peekEnd() { if (S.peek === null) return; const v = S.peek; S.peek = null; showVariant(v, true); setReadonly(v === "original"); badge(v); syncVariantButtons(); syncStatus(); }

  function scaleFor(value, img, host) {
    if (!img?.naturalWidth) return 1;
    if (value !== "fit") return Number(value) / 100;
    return Math.min(1, Math.max(120, (host?.clientWidth || innerWidth) - 72) / img.naturalWidth);
  }
  function zoom(value = S.zoom) {
    const { card } = activeReview();
    const host = $("#page-view .review-canvas-host");
    const wrap = card?.querySelector(".review-image-wrap");
    const img = wrap?.querySelector("img");
    if (!card || !host || !wrap || !img?.naturalWidth || !img.naturalHeight) return false;
    S.zoom = value === "fit" ? "fit" : Math.max(10, Math.min(800, Number(value) || 100));
    const scale = scaleFor(S.zoom, img, host), w = Math.max(1, Math.round(img.naturalWidth * scale)), h = Math.max(1, Math.round(img.naturalHeight * scale));
    Object.assign(img.style, { width: `${w}px`, height: `${h}px`, maxWidth: "none" });
    Object.assign(wrap.style, { width: `${w}px`, height: `${h}px` });
    Object.assign(card.style, { width: `${w}px`, maxWidth: "none" });
    const canvas = $("canvas.brush-canvas", wrap); if (canvas) Object.assign(canvas.style, { width: `${w}px`, height: `${h}px` });
    $$('[data-pro-zoom]').forEach((b) => { const on = String(b.dataset.proZoom) === String(S.zoom); b.classList.toggle("active", on); b.setAttribute("aria-pressed", String(on)); });
    const out = $(".pro-zoom-readout"); if (out) out.textContent = S.zoom === "fit" ? `Fit · ${Math.round(scale * 100)}%` : `${Math.round(scale * 100)}%`;
    syncStatus(); return true;
  }
  function stepZoom(dir) {
    const { card } = activeReview(), img = card?.querySelector(".review-image-wrap img"), host = $("#page-view .review-canvas-host");
    if (!img?.naturalWidth || !host) return;
    const current = scaleFor(S.zoom, img, host) * 100;
    let i = steps.findIndex((x) => x >= current - .5); if (i < 0) i = steps.length - 1;
    zoom(steps[Math.max(0, Math.min(steps.length - 1, i + dir))]);
  }

  function mountReviewTools() {
    if (document.body.dataset.appStage !== "review") return;
    const ws = $("#page-view .review-workspace-shell"), bar = $(".review-sticky-toolbar", ws), actions = $(".review-actions-group", bar), { card } = activeReview();
    if (!ws || !bar || !actions || !card) return;
    let tools = $(".pro-review-options", bar);
    if (!tools) {
      tools = document.createElement("div"); tools.className = "pro-review-options";
      const variants = document.createElement("div"); variants.className = "pro-segmented";
      ["original", "clean"].forEach((v) => { const b = button(v === "original" ? "Original" : "Clean", "pro-option-button"); b.dataset.proReviewVariant = v; b.onclick = () => showVariant(v); variants.append(b); });
      const peek = button("Giữ: xem gốc", "pro-option-button pro-peek-button"); peek.title = "Giữ chuột hoặc phím \\ để xem ảnh gốc";
      peek.onpointerdown = (e) => { e.preventDefault(); peek.setPointerCapture?.(e.pointerId); peekStart(); };
      ["pointerup", "pointercancel", "lostpointercapture", "pointerleave"].forEach((n) => peek.addEventListener(n, peekEnd));
      const z = document.createElement("div"); z.className = "pro-zoom-controls";
      const minus = button("−", "pro-option-button pro-square-button"), fit = button("Fit", "pro-option-button"), one = button("100%", "pro-option-button"), plus = button("+", "pro-option-button pro-square-button"), out = document.createElement("span");
      fit.dataset.proZoom = "fit"; one.dataset.proZoom = "100"; out.className = "pro-zoom-readout";
      minus.onclick = () => stepZoom(-1); fit.onclick = () => zoom("fit"); one.onclick = () => zoom(100); plus.onclick = () => stepZoom(1);
      z.append(minus, fit, one, plus, out); tools.append(variants, peek, z); bar.insertBefore(tools, actions);
    }
    const stitched = $(".review-stitched-shell", ws); tools.hidden = Boolean(stitched && !stitched.hidden);
    const img = $(".review-image-wrap img", card);
    if (img && img.dataset.proZoomBound !== "1") { img.dataset.proZoomBound = "1"; img.addEventListener("load", () => { zoom(S.zoom); badge(img.dataset.proViewVariant || S.variant); }); }
    if (!img?.dataset.proViewVariant) showVariant(S.variant); else { badge(img.dataset.proViewVariant); setReadonly(img.dataset.proViewVariant === "original"); syncVariantButtons(); zoom(S.zoom); }
    const host = $(".review-canvas-host", ws);
    if (host && host.dataset.proWheel !== "1") { host.dataset.proWheel = "1"; host.addEventListener("wheel", (e) => { if ((!e.ctrlKey && !e.metaKey) || e.target.closest(".brush-mode")) return; e.preventDefault(); stepZoom(e.deltaY < 0 ? 1 : -1); }, { passive: false }); }
  }

  function command(c) {
    const stage = (s) => $(`.sidebar-link[data-stage="${s}"]`)?.click();
    return ({ home: () => click('.sidebar-link[data-route="home"]'), import: () => click('.sidebar-link[data-route="import"]'), preview: () => stage("preview"), review: () => stage("review"), editor: () => stage("editor"), settings: () => click("#settings-toggle"), pages: () => click("#toggle-page-panel"), tools: () => click("#toggle-inspector-panel"), focus: () => click("#toggle-focus-mode"), original: () => showVariant("original"), clean: () => showVariant("clean"), fit: () => zoom("fit"), one: () => zoom(100), brush: () => click("#page-view .review-controls-slot .brush-toggle-btn"), clear: () => click("#page-view .review-controls-slot .clear-brush-btn"), repaint: () => click("#page-view .review-controls-slot .repaint-btn"), stitched: () => click('[data-review-mode="stitched"]'), slices: () => click('[data-review-mode="slices"]') }[c] || (() => false))();
  }

  function closeMenus() { $$(".pro-menu.open").forEach((m) => { m.classList.remove("open"); $(".pro-menu-trigger", m)?.setAttribute("aria-expanded", "false"); }); }
  function menu(label, items) {
    const root = document.createElement("div"); root.className = "pro-menu";
    const t = button(label, "pro-menu-trigger"); t.setAttribute("aria-haspopup", "menu"); t.setAttribute("aria-expanded", "false");
    const p = document.createElement("div"); p.className = "pro-menu-panel"; p.setAttribute("role", "menu");
    items.forEach((i) => { if (!i) { const s = document.createElement("div"); s.className = "pro-menu-separator"; p.append(s); return; } const b = button(i[0], "pro-menu-item"); b.dataset.proCommand = i[1]; if (i[2]) { const k = document.createElement("kbd"); k.textContent = i[2]; b.append(k); } b.onclick = () => { closeMenus(); command(i[1]); }; p.append(b); });
    t.onclick = (e) => { e.stopPropagation(); const open = !root.classList.contains("open"); closeMenus(); root.classList.toggle("open", open); t.setAttribute("aria-expanded", String(open)); };
    root.append(t, p); return root;
  }
  function mountMenu() {
    const main = $("#app > .app-main"), header = $("#site-header"); if (!main || !header || $("#pro-menubar")) return;
    const bar = document.createElement("nav"); bar.id = "pro-menubar"; bar.className = "pro-menubar";
    const brand = document.createElement("div"); brand.className = "pro-menubar-brand"; brand.textContent = "MT";
    bar.append(brand,
      menu("Tệp", [["Trang chủ", "home"], ["Nhập chương…", "import"], null, ["Cài đặt…", "settings"]]),
      menu("Chỉnh sửa", [["Đánh dấu vùng lỗi", "brush", "B"], ["Xóa nét đánh dấu", "clear"], ["Xử lý vùng đánh dấu…", "repaint"]]),
      menu("Ảnh", [["Xem ảnh gốc", "original", "\\"], ["Xem ảnh đã inpaint", "clean"], null, ["Từng lát", "slices"], ["Ghép như ảnh gốc", "stitched"]]),
      menu("Xem", [["Fit on Screen", "fit", "Ctrl+0"], ["100%", "one", "Ctrl+1"], null, ["Chế độ tập trung", "focus", "Tab"]]),
      menu("Cửa sổ", [["Trang / Lát", "pages"], ["Công cụ", "tools"], null, ["Xem cắt lát", "preview"], ["Xem inpaint", "review"], ["Biên tập", "editor"]]));
    const doc = document.createElement("div"); doc.id = "pro-document-title"; doc.className = "pro-document-title"; doc.textContent = "Manga Translator"; bar.append(doc); main.insertBefore(bar, header);
  }
  function mountStatus() {
    const main = $("#app > .app-main"); if (!main || $("#pro-statusbar")) return;
    const f = document.createElement("footer"); f.id = "pro-statusbar"; f.className = "pro-statusbar"; f.innerHTML = '<span id="pro-status-stage">Ready</span><span id="pro-status-page"></span><span class="pro-status-spacer"></span><span id="pro-status-variant"></span><span id="pro-status-zoom"></span>'; main.append(f);
  }
  function syncRail() { $$("#app-sidebar .sidebar-link").forEach((b) => { const text = $("span", b)?.textContent?.trim() || b.getAttribute("aria-label") || "Công cụ"; b.dataset.proLabel = text; b.title = text; }); }
  function syncStatus() {
    const stage = document.body.dataset.appStage || "landing", names = { landing: "Home", preview: "Slice", review: "Inpaint", editor: "Type" };
    const page = $("#page-view .review-card .page-block-label")?.textContent || $("#page-view .preview-label")?.textContent || $("#page-view .page-block-label")?.textContent || "";
    const set = (id, text) => { const n = $(id); if (n) n.textContent = text; };
    set("#pro-status-stage", names[stage] || stage); set("#pro-status-page", page); set("#pro-status-variant", stage === "review" ? (S.variant === "original" ? "Original" : "Clean") : "");
    const { card } = activeReview(), img = card?.querySelector(".review-image-wrap img"), host = $("#page-view .review-canvas-host"); set("#pro-status-zoom", stage === "review" && img?.naturalWidth ? `${Math.round(scaleFor(S.zoom, img, host) * 100)}%` : "");
    set("#pro-document-title", `${names[stage] || stage}${window.currentChapterId ? ` · ${window.currentChapterId}` : ""}`);
  }
  function panAndKeys() {
    if (document.body.dataset.proInput === "1") return; document.body.dataset.proInput = "1";
    let drag = null;
    addEventListener("keydown", (e) => {
      if (e.target.matches?.("input,textarea,select") || e.isComposing) return;
      if (e.code === "Space" && !e.repeat) { S.space = true; document.body.classList.add("pro-space-pan"); e.preventDefault(); return; }
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "0") { e.preventDefault(); command("fit"); }
      else if ((e.ctrlKey || e.metaKey) && k === "1") { e.preventDefault(); command("one"); }
      else if (e.key === "Tab" && document.body.dataset.appStage !== "landing") { e.preventDefault(); command("focus"); }
      else if (e.key === "\\" && document.body.dataset.appStage === "review") { e.preventDefault(); peekStart(); }
      else if (k === "b" && document.body.dataset.appStage === "review") { e.preventDefault(); command("brush"); }
    });
    addEventListener("keyup", (e) => { if (e.code === "Space") { S.space = false; drag = null; document.body.classList.remove("pro-space-pan", "pro-space-dragging"); } if (e.key === "\\") peekEnd(); });
    addEventListener("blur", () => { S.space = false; drag = null; document.body.classList.remove("pro-space-pan", "pro-space-dragging"); peekEnd(); });
    document.addEventListener("pointerdown", (e) => { if (!S.space || e.button !== 0) return; const host = e.target.closest(".review-canvas-host,.translation-canvas-host,.preview-main"); if (!host) return; e.preventDefault(); e.stopPropagation(); drag = { host, x: e.clientX, y: e.clientY, l: host.scrollLeft, t: host.scrollTop }; document.body.classList.add("pro-space-dragging"); host.setPointerCapture?.(e.pointerId); }, true);
    document.addEventListener("pointermove", (e) => { if (!drag) return; drag.host.scrollLeft = drag.l - (e.clientX - drag.x); drag.host.scrollTop = drag.t - (e.clientY - drag.y); }, true);
    document.addEventListener("pointerup", () => { drag = null; document.body.classList.remove("pro-space-dragging"); }, true);
  }
  function sync() { document.body.classList.add("pro-editor-shell"); mountMenu(); mountStatus(); syncRail(); mountReviewTools(); syncStatus(); }
  function queue() { if (S.sync) return; S.sync = true; requestAnimationFrame(() => { S.sync = false; sync(); }); }
  function boot() {
    panAndKeys(); sync();
    const app = $("#app"); if (app && app.dataset.proObserver !== "1") { app.dataset.proObserver = "1"; new MutationObserver(queue).observe(app, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden", "class", "data-page-index"] }); }
    addEventListener("resize", () => { if (document.body.dataset.appStage === "review" && S.zoom === "fit") zoom("fit"); });
    document.addEventListener("click", (e) => { if (!e.target.closest(".pro-menu")) closeMenus(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenus(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot, { once: true }); else boot();
})();
