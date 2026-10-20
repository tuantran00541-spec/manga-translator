// Takes the chapter images the user's own browser already shows and uploads them to the local app.
const DEFAULT_SERVER = "http://127.0.0.1:8000";
const KEPT_FORMATS = new Set(["image/png", "image/jpeg", "image/webp", "image/bmp"]);
const REFERER_RULE = 1;

async function setStatus(status) {
  await chrome.storage.session.set({ status: { ...status, at: Date.now() } });
}

// Runs inside each frame: scrolls to wake lazy images, then lists the images and canvases in reading order.
async function collectPage(autoScroll) {
  const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
  const lazy = ["data-src", "data-lazy-src", "data-original", "data-url"];
  const shown = (node) => node.getBoundingClientRect();
  const lazySrc = (node) => lazy.map((name) => node.getAttribute(name)).find(Boolean);
  // A tall slot still showing a tiny picture is a page that has not loaded yet.
  const waiting = (node) => node.tagName === "IMG" && shown(node).width >= 200 && shown(node).height >= 150
    && !lazySrc(node) && (!node.complete || node.naturalHeight < 100);
  if (autoScroll) {
    // Readers often scroll a box of their own instead of the window.
    const boxes = [...document.querySelectorAll("*")].filter((el) => el.scrollHeight > el.clientHeight + 50
      && /(auto|scroll)/.test(getComputedStyle(el).overflowY));
    const box = [document.scrollingElement || document.documentElement, ...boxes]
      .reduce((best, el) => (el.scrollHeight - el.clientHeight > best.scrollHeight - best.clientHeight ? el : best));
    const start = box.scrollTop;
    let last = -1;
    for (let i = 0; i < 400 && box.scrollTop !== last; i++) {
      last = box.scrollTop;
      box.scrollBy(0, Math.max(400, box.clientHeight * 0.9));
      await sleep(250);
    }
    await sleep(800);
    // Pages that were scrolled past before they loaded get a second look.
    const gaveUp = new Set();
    for (const deadline = Date.now() + 30000; Date.now() < deadline;) {
      const pending = [...document.querySelectorAll("img")].find((node) => waiting(node) && !gaveUp.has(node));
      if (!pending) break;
      pending.scrollIntoView({ block: "center" });
      for (let t = 0; t < 16 && waiting(pending); t++) await sleep(250);
      if (waiting(pending)) gaveUp.add(pending);
    }
    box.scrollTop = start;
  }
  const items = [];
  const seen = new Set();
  const missing = [];
  for (const node of document.querySelectorAll("img, canvas")) {
    const { width, height } = shown(node);
    if (width < 200) continue;
    if (node.tagName === "CANVAS") {
      try { items.push({ data: node.toDataURL("image/png"), width, height }); } catch { /* a canvas drawn from another site cannot be read */ }
      continue;
    }
    let src = node.currentSrc || node.src || "";
    const placeholder = !src || (src.startsWith("data:") && node.naturalWidth < 300) || (height >= 150 && node.complete && node.naturalHeight < 100);
    if (placeholder) src = lazySrc(node) || "";
    if (!src) {
      if (height >= 150) missing.push({ width, height });
      continue;
    }
    // A loaded strip this thin is a divider, never a page.
    if (!placeholder && node.naturalHeight && node.naturalHeight < 100) continue;
    src = new URL(src, location.href).href;
    if (seen.has(src)) continue;
    seen.add(src);
    if (src.startsWith("blob:")) {
      try {
        const blob = await (await fetch(src)).blob();
        src = await new Promise((done) => { const reader = new FileReader(); reader.onload = () => done(reader.result); reader.readAsDataURL(blob); });
      } catch { continue; }
      items.push({ data: src, width, height });
    } else {
      items.push({ url: src, width, height });
    }
  }
  return { title: document.title, page: location.href, items, missing };
}

// Keeps the pages: the width most of the chapter is shown at, minus banners and thumbnails.
function chapterPages(frames) {
  const all = frames.flatMap(({ frameId, result }) => (result?.items || []).map((item) => ({ ...item, frameId, page: result.page })));
  const weight = new Map();
  for (const { width, height } of all) weight.set(Math.round(width / 10), (weight.get(Math.round(width / 10)) || 0) + height);
  const common = 10 * [...weight].reduce((best, entry) => (entry[1] > best[1] ? entry : best), [0, 0])[0];
  const page = ({ width, height }) => width >= common * 0.6 && (width <= common * 1.1 || height >= width * 0.5);
  const missing = frames.flatMap(({ result }) => result?.missing || []).filter(page).length;
  return { items: all.filter(page), missing };
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

// Tells the app which version Chrome loaded; true when the app holds a newer one.
async function outdated(server) {
  try {
    const reply = await fetch(`${server}/api/extension/hello`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ version: chrome.runtime.getManifest().version }),
    });
    return Boolean((await reply.json()).outdated);
  } catch { return false; }
}

// Image hosts often refuse requests that do not come from the chapter page.
async function sendAsPage(page) {
  if (sendAsPage.current === page) return;
  sendAsPage.current = page;
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
  const [{ result }] = await chrome.scripting.executeScript({ target: { tabId, frameIds: [item.frameId] }, func: fetchInPage, args: [item.url] });
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
  const stale = await outdated(server);
  await setStatus({ state: "busy", text: autoScroll ? "Đang cuộn trang để tải hết ảnh…" : "Đang tìm ảnh trên trang…" });
  let frames;
  try {
    frames = await chrome.scripting.executeScript({ target: { tabId, allFrames: true }, func: collectPage, args: [autoScroll] });
  } catch {
    frames = await chrome.scripting.executeScript({ target: { tabId }, func: collectPage, args: [autoScroll] });
  }
  frames.sort((a, b) => a.frameId - b.frameId);
  const { items, missing } = chapterPages(frames);
  if (!items.length) {
    await setStatus({ state: "error", text: "Không thấy ảnh truyện nào trên trang này." });
    return;
  }
  const form = new FormData();
  let kept = 0;
  try {
    for (const [index, item] of items.entries()) {
      await setStatus({ state: "busy", text: `Đang lấy ảnh ${index + 1}/${items.length}…` });
      try {
        await sendAsPage(item.page);
        const blob = await asUploadable(await download(tabId, item));
        const ext = blob.type.split("/")[1].replace("jpeg", "jpg");
        form.append("files", blob, `${String(index + 1).padStart(3, "0")}.${ext}`);
        kept++;
      } catch { /* one broken image does not stop the chapter */ }
    }
  } finally {
    sendAsPage.current = null;
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
  const skipped = items.length - kept;
  const notes = [skipped && `bỏ ${skipped} ảnh lỗi`, missing && `${missing} ảnh chưa tải kịp, thử bấm lại`,
    stale && "tiện ích có bản mới: vào trang Tiện ích bấm Tải lại"].filter(Boolean);
  await setStatus({ state: "done", text: `Đã gửi ${kept} ảnh${notes.length ? `, ${notes.join(", ")}` : ""}.` });
  await chrome.tabs.create({ url: `${server}/#${body.chapter_id}` });
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "grab") return;
  grab(message.tabId, message.autoScroll).catch((error) => setStatus({ state: "error", text: `Lỗi: ${error.message}` }));
});
