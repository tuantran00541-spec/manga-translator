// Takes the chapter images the user's own browser already shows and uploads them to the local app.
const DEFAULT_SERVER = "http://127.0.0.1:8000";
const KEPT_FORMATS = new Set(["image/png", "image/jpeg", "image/webp", "image/bmp"]);
const REFERER_RULE = 1;

async function setStatus(status) {
  await chrome.storage.session.set({ status: { ...status, at: Date.now() } });
}

// Runs inside the page: scrolls to wake lazy images, then lists the wide images and canvases in reading order.
async function collectPage(autoScroll) {
  const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
  if (autoScroll) {
    const start = window.scrollY;
    let last = -1;
    for (let i = 0; i < 400 && window.scrollY !== last; i++) {
      last = window.scrollY;
      window.scrollBy(0, Math.max(400, window.innerHeight * 0.9));
      await sleep(250);
    }
    await sleep(800);
    window.scrollTo(0, start);
  }
  const lazy = ["data-src", "data-lazy-src", "data-original", "data-url"];
  const nodes = [...document.querySelectorAll("img, canvas")].filter((node) => node.getBoundingClientRect().width > 0);
  const widest = Math.max(0, ...nodes.map((node) => node.getBoundingClientRect().width));
  const items = [];
  const seen = new Set();
  for (const node of nodes) {
    if (node.getBoundingClientRect().width < widest * 0.6) continue;
    if (node.tagName === "CANVAS") {
      try { items.push({ data: node.toDataURL("image/png") }); } catch { /* a canvas drawn from another site cannot be read */ }
      continue;
    }
    let src = node.currentSrc || node.src || "";
    if (!src || (src.startsWith("data:") && node.naturalWidth < 300)) src = lazy.map((name) => node.getAttribute(name)).find(Boolean) || src;
    if (!src) continue;
    src = new URL(src, location.href).href;
    if (seen.has(src)) continue;
    seen.add(src);
    if (src.startsWith("blob:")) {
      try {
        const blob = await (await fetch(src)).blob();
        src = await new Promise((done) => { const reader = new FileReader(); reader.onload = () => done(reader.result); reader.readAsDataURL(blob); });
      } catch { continue; }
      items.push({ data: src });
    } else {
      items.push({ url: src });
    }
  }
  return { title: document.title, page: location.href, items };
}

async function findServer() {
  const { server } = await chrome.storage.local.get("server");
  const candidates = [server || DEFAULT_SERVER];
  for (let port = 8000; port < 8020; port++) candidates.push(`http://127.0.0.1:${port}`);
  for (const base of [...new Set(candidates)]) {
    try {
      const reply = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) });
      if (reply.ok && "models_missing" in (await reply.json())) return base;
    } catch { /* not this port */ }
  }
  return null;
}

// Image hosts often refuse requests that do not come from the chapter page.
async function sendAsPage(page) {
  await chrome.declarativeNetRequest.updateSessionRules({
    removeRuleIds: [REFERER_RULE],
    addRules: [{
      id: REFERER_RULE,
      priority: 1,
      action: { type: "modifyHeaders", requestHeaders: [{ header: "Referer", operation: "set", value: page }] },
      condition: { initiatorDomains: [chrome.runtime.id], resourceTypes: ["xmlhttprequest"] },
    }],
  });
}

// Runs inside the page: fetches one image with the page's own origin and cookies.
async function fetchInPage(url) {
  const blob = await (await fetch(url, { credentials: "include" })).blob();
  return new Promise((done) => { const reader = new FileReader(); reader.onload = () => done(reader.result); reader.readAsDataURL(blob); });
}

async function download(tabId, item) {
  try {
    const reply = await fetch(item.url || item.data, { credentials: "include" });
    if (reply.ok) return reply.blob();
  } catch { /* try again from the page */ }
  const [{ result }] = await chrome.scripting.executeScript({ target: { tabId }, func: fetchInPage, args: [item.url] });
  return (await fetch(result)).blob();
}

async function asUploadable(blob) {
  if (KEPT_FORMATS.has(blob.type)) return blob;
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

async function grab(tabId, autoScroll) {
  await setStatus({ state: "busy", text: "Đang tìm app trên máy…" });
  const server = await findServer();
  if (!server) {
    await setStatus({ state: "error", text: "Chưa thấy app. Mở Manga Translator trước rồi bấm lại." });
    return;
  }
  await setStatus({ state: "busy", text: autoScroll ? "Đang cuộn trang để tải hết ảnh…" : "Đang tìm ảnh trên trang…" });
  const [{ result }] = await chrome.scripting.executeScript({ target: { tabId }, func: collectPage, args: [autoScroll] });
  if (!result || !result.items.length) {
    await setStatus({ state: "error", text: "Không thấy ảnh truyện nào trên trang này." });
    return;
  }
  await sendAsPage(result.page);
  const form = new FormData();
  let kept = 0;
  try {
    for (const [index, item] of result.items.entries()) {
      await setStatus({ state: "busy", text: `Đang lấy ảnh ${index + 1}/${result.items.length}…` });
      try {
        const blob = await asUploadable(await download(tabId, item));
        const ext = blob.type.split("/")[1].replace("jpeg", "jpg");
        form.append("files", blob, `${String(index + 1).padStart(3, "0")}.${ext}`);
        kept++;
      } catch { /* one broken image does not stop the chapter */ }
    }
  } finally {
    await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: [REFERER_RULE] });
  }
  if (!kept) {
    await setStatus({ state: "error", text: "Trang có ảnh nhưng không lấy được ảnh nào." });
    return;
  }
  await setStatus({ state: "busy", text: `Đang gửi ${kept} ảnh vào app…` });
  const reply = await fetch(`${server}/api/chapter/upload`, { method: "POST", body: form });
  const body = await reply.json().catch(() => ({}));
  if (!reply.ok || !body.chapter_id) {
    await setStatus({ state: "error", text: `App từ chối: ${body.detail || reply.status}` });
    return;
  }
  const skipped = result.items.length - kept;
  await setStatus({ state: "done", text: `Đã gửi ${kept} ảnh${skipped ? `, bỏ ${skipped} ảnh lỗi` : ""}.` });
  await chrome.tabs.create({ url: `${server}/#${body.chapter_id}` });
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "grab") return;
  grab(message.tabId, message.autoScroll).catch((error) => setStatus({ state: "error", text: `Lỗi: ${error.message}` }));
});
