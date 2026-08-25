(() => {
  const legacyRenderEditor = window.renderEditor;
  if (typeof legacyRenderEditor !== "function") return;

  const pending = new Map();

  function sourceBoxSet(obj) {
    return new Set(Array.isArray(obj?.source_boxes) ? obj.source_boxes.map(String) : []);
  }

  function needsSync(page) {
    if (!page || page.skipped) return false;
    const activeBoxes = (page.boxes || []).filter((box) => box && !box.removed && box.id);
    if (!activeBoxes.length) return false;
    const objects = page.text_objects || [];
    return activeBoxes.some((box) => !objects.some((obj) => sourceBoxSet(obj).has(String(box.id))));
  }

  async function ensurePage(pageIndex) {
    const chapterId = window.currentChapterId;
    if (!chapterId) return null;
    const key = `${chapterId}:${pageIndex}`;
    if (pending.has(key)) return pending.get(key);

    const job = (async () => {
      const response = await fetch("/api/text_objects/ensure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chapter_id: chapterId, page_indices: [pageIndex] }),
      });
      const parse = typeof window.parseApiResponse === "function"
        ? window.parseApiResponse
        : async (r) => r.json().catch(() => ({}));
      const data = await parse(response);
      if (!response.ok) {
        const getError = typeof window.getErrorMessage === "function"
          ? window.getErrorMessage
          : (status, payload) => payload?.detail || `HTTP ${status}`;
        throw new Error(getError(response.status, data));
      }
      if (chapterId !== window.currentChapterId) return null;
      window.currentManifest = data;
      return data;
    })();

    pending.set(key, job);
    try {
      return await job;
    } finally {
      pending.delete(key);
    }
  }

  window.ensureAutoTextObjects = ensurePage;

  window.renderEditor = function renderEditorWithDetectedRegions() {
    legacyRenderEditor();
    const pageIndex = Number(window.editorState?.activePageIndex || 0);
    const page = window.currentManifest?.pages?.[pageIndex];
    if (!needsSync(page)) return;

    ensurePage(pageIndex)
      .then((manifest) => {
        if (!manifest || Number(window.editorState?.activePageIndex || 0) !== pageIndex) return;
        legacyRenderEditor();
      })
      .catch((err) => {
        if (typeof window.showToast === "function") {
          window.showToast("Không thể tự tạo vùng chữ từ nhận diện: " + err.message, "error");
        }
      });
  };
})();
