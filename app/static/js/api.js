
var currentChapterId = null;
var currentManifest = null;

let availableFonts = [];

async function parseApiResponse(resp) {
  let data = null;
  try {
    data = await resp.json();
  } catch (_) {
    try {
      data = { detail: await resp.text() };
    } catch (_) {
      data = {};
    }
  }
  return data || {};
}
window.parseApiResponse = parseApiResponse;

function getErrorMessage(status, data) {
  if (data && data.detail) {
    if (Array.isArray(data.detail)) {
      return data.detail
        .map((e) => (typeof e === "string" ? e : e.msg || JSON.stringify(e)))
        .join("; ");
    }
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail.trim();
    }
    if (typeof data.detail?.message === "string") return data.detail.message;
  }
  const statusMessages = {
    400: "Yêu cầu không hợp lệ (400)",
    404: "Không tìm thấy dữ liệu (404)",
    422: "Dữ liệu gửi lên không đúng định dạng (422)",
    500: "Lỗi nội bộ máy chủ (500)",
    502: "Máy chủ cổng không phản hồi (502)",
    504: "Hết thời gian chờ phản hồi từ máy chủ (504)",
  };
  return statusMessages[status] || `Máy chủ trả về lỗi HTTP ${status}`;
}
window.getErrorMessage = getErrorMessage;

let _lastCheckpointStage = null;
let _lastCheckpointPage = null;
let _lastCheckpointChapter = null;
let _checkpointSeq = 0;
let _chapterNavigationSeq = 0;

async function setWorkflowCheckpoint(stage, pageIndex) {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  const canonicalIndex = Math.max(0, parseInt(pageIndex, 10) || 0);
  if (
    _lastCheckpointChapter === chapterId
    && _lastCheckpointStage === stage
    && _lastCheckpointPage === canonicalIndex
  ) {
    return;
  }
  _lastCheckpointChapter = chapterId;
  _lastCheckpointStage = stage;
  _lastCheckpointPage = canonicalIndex;
  const seq = ++_checkpointSeq;

  try {
    const resp = await fetch("/api/workflow_checkpoint", {
      method: "POST",
      
      keepalive: true,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: chapterId,
        stage,
        page_index: canonicalIndex,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (
      seq === _checkpointSeq
      && chapterId === currentChapterId
      && currentManifest
    ) {
      currentManifest.workflow = { stage, page_index: canonicalIndex };
    }
  } catch (err) {
    if (
      seq === _checkpointSeq
      && _lastCheckpointChapter === chapterId
      && _lastCheckpointStage === stage
      && _lastCheckpointPage === canonicalIndex
    ) {
      _lastCheckpointChapter = null;
      _lastCheckpointStage = null;
      _lastCheckpointPage = null;
    }
    if (!_pageLeaving) console.error("Workflow checkpoint save failed:", err);
  }
}

let _pageLeaving = false;
window.addEventListener("pagehide", () => { _pageLeaving = true; });
window.addEventListener("pageshow", () => { _pageLeaving = false; });
window.setWorkflowCheckpoint = setWorkflowCheckpoint;

function appendText(parent, tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

async function loadRecentChapters() {
  const container = document.getElementById("recent-chapters");
  if (!container) return;
  const panel = container.closest(".recent-panel");
  const setPanelVisible = (visible) => {
    if (panel) panel.hidden = !visible;
    const empty = document.getElementById("home-empty-state");
    if (empty) empty.hidden = visible;
  };

  try {
    const resp = await fetch("/api/chapters");
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      showToast(
        "Không tải được danh sách chương: " + getErrorMessage(resp.status, data),
        "error",
      );
      container.replaceChildren();
      setPanelVisible(false);
      return;
    }

    const chapters = Array.isArray(data) ? data : [];
    if (chapters.length === 0) {
      container.replaceChildren();
      setPanelVisible(false);
      return;
    }

    setPanelVisible(true);
    container.replaceChildren();
    appendText(container, "div", "recent-title", "Dự án gần đây");
    const list = document.createElement("div");
    list.className = "recent-list";
    const stageLabels = {
      preview: "Xử lý ảnh",
      review: "Kiểm tra chất lượng",
      editor: "Biên tập bản dịch",
    };

    chapters.forEach((ch) => {
      const card = document.createElement("button");
      card.className = "recent-card";
      card.type = "button";
      const chapterId = String(ch?.chapter_id || "");
      card.dataset.chapterId = chapterId;
      card.setAttribute("aria-label", `Mở lại chương ${chapterId || "không rõ"}`);
      card.addEventListener("click", () => resumeChapter(chapterId));

      const coverWrap = document.createElement("div");
      coverWrap.className = "recent-cover-wrap";
      if (ch?.cover) {
        const cover = document.createElement("img");
        cover.className = "recent-cover";
        cover.src = String(ch.cover);
        cover.alt = "";
        cover.loading = "lazy";
        cover.decoding = "async";
        cover.addEventListener("error", () => {
          cover.remove();
          appendText(coverWrap, "span", "recent-cover-fallback", "Không có ảnh bìa");
        }, { once: true });
        coverWrap.appendChild(cover);
      } else {
        appendText(coverWrap, "span", "recent-cover-fallback", "Không có ảnh bìa");
      }
      card.appendChild(coverWrap);

      const info = document.createElement("div");
      info.className = "recent-info";
      let displayName = String(ch?.chapter_name || "").trim() || chapterId || "Chương chưa đặt tên";
      const sourceUrl = String(ch?.source_url || "");
      if (sourceUrl && !ch?.chapter_name) {
        try {
          const parsed = new URL(sourceUrl);
          const pathName = decodeURIComponent(parsed.pathname).split("/").filter(Boolean).pop();
          if (pathName) displayName = pathName.replace(/[-_]+/g, " ");
        } catch (_) {}
      }
      appendText(info, "strong", "recent-name", displayName);
      appendText(
        info,
        "span",
        "recent-url",
        sourceUrl || chapterId,
      );

      const meta = document.createElement("span");
      meta.className = "recent-meta";
      meta.appendChild(
        document.createTextNode(`${Number(ch?.total_pages) || 0} trang · `),
      );
      const rawStage = String(ch?.workflow?.stage || "");
      appendText(
        meta,
        "span",
        "recent-stage-badge",
        stageLabels[rawStage] || rawStage || "Đang xử lý",
      );
      info.appendChild(meta);
      card.appendChild(info);
      list.appendChild(card);
    });

    container.appendChild(list);
  } catch (err) {
    showToast("Không tải được danh sách chương: " + err.message, "error");
    container.replaceChildren();
    setPanelVisible(false);
  }
}
window.loadRecentChapters = loadRecentChapters;

async function refreshChapterManifest(chapterId) {
  if (!chapterId) return null;
  const resp = await fetch(`/api/chapter/${encodeURIComponent(chapterId)}`);
  const data = await parseApiResponse(resp);
  if (!resp.ok) {
    throw new Error(getErrorMessage(resp.status, data));
  }
  if (chapterId !== currentChapterId) return null;
  currentManifest = data;
  return data;
}
window.refreshChapterManifest = refreshChapterManifest;


async function flushEditsBeforeChapterChange() {
  const chapterId = currentChapterId;
  if (!chapterId) return;
  try {
    if (typeof window.flushAllPendingPersists === "function") {
      await window.flushAllPendingPersists(undefined, chapterId);
    }
  } catch (_) {
    if (typeof window.cancelPendingPersist === "function") window.cancelPendingPersist();
  }
}
window.flushEditsBeforeChapterChange = flushEditsBeforeChapterChange;

async function resumeChapter(chapterId) {
  if (!chapterId) return;
  const navigationSeq = ++_chapterNavigationSeq;
  try {
    const resp = await fetch(`/api/chapter/${encodeURIComponent(chapterId)}`);
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (navigationSeq !== _chapterNavigationSeq) return;
    await flushEditsBeforeChapterChange();
    if (navigationSeq !== _chapterNavigationSeq) return;
    currentManifest = data;
    currentChapterId = currentManifest.chapter_id;
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      if (window.location.hash !== `#${currentChapterId}`) {
        window.history.replaceState(null, "", `#${currentChapterId}`);
      }
    } catch (_) {}

    const pages = currentManifest.pages || [];
    let workflow = currentManifest.workflow;
    if (!workflow || !workflow.stage) {
      workflow = { stage: pages.some((p) => p.clean || p.rendered) ? "review" : "preview", page_index: 0 };
    }

    const rawIndex = parseInt(workflow.page_index, 10) || 0;
    const pageIndex = Math.max(0, Math.min(rawIndex, Math.max(0, pages.length - 1)));

    
    if (workflow.stage === "preview") {
      window.initialPreviewCanonicalPageIndex = pageIndex;
      renderPreview();
    } else {
      window.initialReviewCanonicalPageIndex = pageIndex;
      renderReview();
    }
  } catch (err) {
    if (navigationSeq === _chapterNavigationSeq) {
      showToast("Không thể tiếp tục chương: " + err.message, "error");
    }
  }
}

async function loadChapter() {
  const urlEl = document.getElementById("chapter-url");
  if (!urlEl) return;
  const url = urlEl.value.trim();
  if (!url) return;
  const navigationSeq = ++_chapterNavigationSeq;

  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) {
    loadBtn.disabled = true;
    loadBtn.textContent = "Đang tải chương…";
  }

  try {
    const resp = await fetch("/api/chapter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, workers: getWorkersSetting() }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (navigationSeq !== _chapterNavigationSeq) return;
    await flushEditsBeforeChapterChange();
    if (navigationSeq !== _chapterNavigationSeq) return;
    currentManifest = data;
    currentChapterId = currentManifest.chapter_id;
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      window.history.replaceState(null, "", `#${currentChapterId}`);
    } catch (_) {}
    window.previewActivePageIndex = 0;
    renderPreview();
  } catch (err) {
    if (navigationSeq === _chapterNavigationSeq) {
      showToast("Không tải được chương: " + err.message, "error");
      window.suggestExtension?.();
    }
  } finally {
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.textContent = "Tải chương";
    }
  }
}

async function toggleSkip(pageIndex, card, btn) {
  const chapterId = currentChapterId;
  const page = currentManifest.pages[pageIndex];
  const newSkipped = !page.skipped;

  try {
    if (typeof window.flushPreserveRegionSaves === "function") {
      await window.flushPreserveRegionSaves(chapterId, pageIndex);
    }
    if (chapterId !== currentChapterId) return;
    const resp = await fetch("/api/skip_pages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: chapterId,
        page_indices: [pageIndex],
        skipped: newSkipped,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    if (chapterId !== currentChapterId) return;
    currentManifest = data;
    window.previewActivePageIndex = pageIndex;
    renderPreview();
  } catch (err) {
    showToast("Không đổi được trạng thái bỏ qua: " + err.message, "error");
  }
}


function getWorkersSetting() {
  const cores = Number(navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(8, cores));
}
window.getWorkersSetting = getWorkersSetting;

let _activePageProcessing = null;
let _activePageProcessingChapter = null;

function processSelectedPages() {
  if (_activePageProcessing) {
    const label = _activePageProcessingChapter === currentChapterId
      ? "Chương này đang được xử lý."
      : "Một chương khác đang được xử lý.";
    showToast(label + " Vui lòng chờ tiến trình hiện tại hoàn tất.", "info");
    return _activePageProcessing;
  }

  const chapterId = currentChapterId;
  const run = window._processSelectedPagesOnce();
  _activePageProcessingChapter = chapterId;
  _activePageProcessing = run.finally(() => {
    if (_activePageProcessing === wrappedRun) {
      _activePageProcessing = null;
      _activePageProcessingChapter = null;
    }
  });
  const wrappedRun = _activePageProcessing;
  return wrappedRun;
}

async function loadFonts() {
  try {
    const resp = await fetch("/api/fonts");
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    availableFonts = Array.isArray(data) ? data : [{ id: "default", name: "Mặc định (Truyện tranh)" }];
  } catch (e) {
    availableFonts = [{ id: "default", name: "Mặc định (Truyện tranh)" }];
    showToast("Không thể tải danh sách phông chữ, dùng phông mặc định.", "info");
  } finally {
    window.availableFonts = availableFonts;
  }
}

const _preserveRegionSaveStates = new Map();

function _clonePreserveRegions(preserveRegions) {
  if (!Array.isArray(preserveRegions)) return [];
  return preserveRegions.map((region) => ({
    x1: Number(region.x1),
    y1: Number(region.y1),
    x2: Number(region.x2),
    y2: Number(region.y2),
  }));
}

function _preserveRegionSaveKey(chapterId, pageIndex) {
  return `${chapterId}:${pageIndex}`;
}

async function _drainPreserveRegionSaves(state) {
  while (state.persistedVersion < state.version) {
    const requestVersion = state.version;
    const regions = _clonePreserveRegions(state.regions);
    const resp = await fetch("/api/save_preserve_regions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapter_id: state.chapterId,
        page_index: state.pageIndex,
        preserve_regions: regions,
      }),
    });
    const data = await parseApiResponse(resp);
    if (!resp.ok) {
      throw new Error(getErrorMessage(resp.status, data));
    }
    state.persistedVersion = requestVersion;

    if (
      requestVersion === state.version
      && state.chapterId === currentChapterId
      && currentManifest?.pages?.[state.pageIndex]
    ) {
      const serverPage = data?.pages?.[state.pageIndex];
      if (serverPage && typeof serverPage === "object") {
        Object.assign(currentManifest.pages[state.pageIndex], serverPage);
      }
      const serverRegions = serverPage?.preserve_regions;
      currentManifest.pages[state.pageIndex].preserve_regions =
        _clonePreserveRegions(Array.isArray(serverRegions) ? serverRegions : regions);
    }
  }
}

function _ensurePreserveRegionSave(state) {
  if (state.promise) return state.promise;
  const key = _preserveRegionSaveKey(state.chapterId, state.pageIndex);
  state.promise = _drainPreserveRegionSaves(state).finally(() => {
    state.promise = null;
    if (state.persistedVersion >= state.version) {
      _preserveRegionSaveStates.delete(key);
    }
  });
  return state.promise;
}

function savePreserveRegions(pageIndex, preserveRegions) {
  const chapterId = currentChapterId;
  const key = _preserveRegionSaveKey(chapterId, pageIndex);
  let state = _preserveRegionSaveStates.get(key);
  if (!state) {
    state = {
      chapterId,
      pageIndex,
      regions: [],
      version: 0,
      persistedVersion: 0,
      notifiedVersion: 0,
      promise: null,
    };
    _preserveRegionSaveStates.set(key, state);
  }
  state.regions = _clonePreserveRegions(preserveRegions);
  state.version += 1;
  const requestedVersion = state.version;
  return _ensurePreserveRegionSave(state).catch((err) => {
    if (state.notifiedVersion < requestedVersion) {
      state.notifiedVersion = state.version;
      showToast("Không lưu được vùng giữ nguyên: " + err.message, "error");
    }
  });
}

async function flushPreserveRegionSaves(chapterId = currentChapterId, pageIndex) {
  const jobs = [];
  for (const state of _preserveRegionSaveStates.values()) {
    if (state.chapterId !== chapterId) continue;
    if (pageIndex !== undefined && state.pageIndex !== pageIndex) continue;
    if (state.persistedVersion < state.version || state.promise) {
      jobs.push(_ensurePreserveRegionSave(state));
    }
  }
  await Promise.all(jobs);
}
window.flushPreserveRegionSaves = flushPreserveRegionSaves;

(() => {
  const CACHE_PREFIX = "manga_models_cache_v1_";
  const CACHE_TTL_MS = 5 * 60 * 1000;
  const inflight = new Map();

  function readEntry(provider) {
    if (!provider) return null;
    try {
      const raw = localStorage.getItem(`${CACHE_PREFIX}${provider}`);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.models)) return null;
      const age = Date.now() - Number(parsed.ts || 0);
      return { models: parsed.models, fresh: age >= 0 && age < CACHE_TTL_MS };
    } catch (_) {
      return null;
    }
  }

  function writeEntry(provider, models) {
    if (!provider || !Array.isArray(models)) return;
    try {
      localStorage.setItem(`${CACHE_PREFIX}${provider}`, JSON.stringify({
        models,
        ts: Date.now(),
      }));
    } catch (_) {}
  }

  function sameList(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  const generations = new Map();

  function getGeneration(provider) {
    return generations.get(provider) || 0;
  }

  async function requestFromServer(provider, force) {
    const key = `${provider}:${force ? "1" : "0"}`;
    if (inflight.has(key)) return inflight.get(key);
    const reqGen = getGeneration(provider);
    const job = (async () => {
      const url = `/api/visual_qc/providers/${encodeURIComponent(provider)}/models${force ? "?refresh=1" : ""}`;
      const resp = await fetch(url);
      const data = await parseApiResponse(resp);
      if (!resp.ok) throw new Error(getErrorMessage(resp.status, data));
      const models = Array.isArray(data.models) ? data.models : [];
      if (getGeneration(provider) === reqGen) writeEntry(provider, models);
      return models;
    })().finally(() => {
      if (inflight.get(key) === job) inflight.delete(key);
    });
    inflight.set(key, job);
    return job;
  }

  async function fetchModels(provider, { force = false, onFresh } = {}) {
    if (!provider) return [];
    const startGen = getGeneration(provider);
    const cached = force ? null : readEntry(provider);
    if (cached) {
      if (!cached.fresh) {
        requestFromServer(provider, false)
          .then((freshModels) => {
            if (getGeneration(provider) === startGen && typeof onFresh === "function" && !sameList(cached.models, freshModels)) {
              onFresh(freshModels);
            }
          })
          .catch(() => {});
      }
      return cached.models;
    }
    return requestFromServer(provider, force);
  }

  function invalidate(provider) {
    if (!provider) return;
    generations.set(provider, getGeneration(provider) + 1);
    inflight.delete(`${provider}:0`);
    inflight.delete(`${provider}:1`);
    try {
      localStorage.removeItem(`${CACHE_PREFIX}${provider}`);
    } catch (_) {}
  }

  window.aiModelCache = {
    getCached: readEntry,
    fetchModels,
    invalidate,
  };
})();


