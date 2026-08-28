(() => {
  const PROCESS_BATCH_SIZE = 16;
  const pageCountCache = new WeakMap();

  function pageCounts(pages) {
    if (!Array.isArray(pages)) return new Map();
    let counts = pageCountCache.get(pages);
    if (counts) return counts;
    counts = new Map();
    pages.forEach((page) => {
      const source = Number.isInteger(page?.source_page) ? page.source_page : -1;
      counts.set(source, (counts.get(source) || 0) + 1);
    });
    pageCountCache.set(pages, counts);
    return counts;
  }

  // Replace the O(n) filter inside pageLabel with a single O(n) precompute per
  // manifest page array. Large webtoon chapters can expose 100-300 slices, and
  // the navigator calls pageLabel once for every item on each render.
  window.pageLabel = function optimizedPageLabel(pages, pageIndex) {
    const page = Array.isArray(pages) ? pages[pageIndex] : null;
    if (!page) return `Trang ${Number(pageIndex) + 1}`;
    const source = Number.isInteger(page.source_page) ? page.source_page : pageIndex;
    const total = pageCounts(pages).get(source) || 1;
    if (total <= 1) return `Trang ${source + 1}`;
    const slice = Number.isInteger(page.slice_index) ? page.slice_index : 0;
    return `Trang ${source + 1} · Lát ${slice + 1}/${total}`;
  };

  // The backend now publishes each completed page immediately, so a large HTTP
  // batch no longer strands clean images behind its slowest worker. Use a larger
  // scheduling window to let shared-seam detection deduplicate almost every seam,
  // while still returning UI progress periodically on very long chapters.
  window.processSelectedPages = async function optimizedProcessSelectedPages() {
    const pages = currentManifest?.pages || [];
    const indices = pages
      .map((page, index) => (page?.skipped ? null : index))
      .filter((index) => index !== null);

    if (indices.length === 0) {
      showToast("Không có trang nào để xử lý.", "error");
      return;
    }

    const chapterId = currentChapterId;
    const total = indices.length;
    const btn = document.querySelector("#preview-toolbar .preview-primary-action")
      || document.querySelector("#preview-toolbar button");
    if (btn) btn.disabled = true;

    let completed = 0;
    try {
      for (let start = 0; start < total; start += PROCESS_BATCH_SIZE) {
        const batch = indices.slice(start, start + PROCESS_BATCH_SIZE);
        if (btn) btn.textContent = `Đang xử lý ${completed}/${total}…`;

        const resp = await fetch("/api/process_pages", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter_id: chapterId,
            page_indices: batch,
            workers: getWorkersSetting(),
          }),
        });
        const data = await parseApiResponse(resp);
        if (!resp.ok) {
          throw new Error(getErrorMessage(resp.status, data));
        }
        if (chapterId !== currentChapterId) return;

        currentManifest = data;
        completed += batch.length;
        if (btn) btn.textContent = `Đã xử lý ${completed}/${total}…`;
      }

      renderReview();
    } catch (err) {
      const prefix = completed > 0
        ? `Đã xử lý ít nhất ${completed}/${total} trang. Phần tiếp theo thất bại: `
        : "Xử lý trang thất bại: ";
      showToast(prefix + err.message, "error");
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Tiếp tục xử lý";
      }
    }
  };
})();
