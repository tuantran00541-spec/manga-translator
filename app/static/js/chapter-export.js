(() => {
  const legacyShowRenderResult = window.showRenderResult;
  if (typeof legacyShowRenderResult === "function") {
    window.showRenderResult = function showStrictRenderResult(pageIndex, outputPath) {
      legacyShowRenderResult(pageIndex, outputPath);
      const link = document.querySelector(".translation-panel-host .render-result .download-link");
      if (link && window.currentChapterId) {
        link.href = `/api/download/${encodeURIComponent(window.currentChapterId)}/${pageIndex}`;
        link.textContent = "Tải ảnh đã kết xuất";
      }
    };
  }

  function buildExportButton(toolbar) {
    if (!toolbar || toolbar.querySelector(".chapter-export-run")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ui-btn ui-btn-primary chapter-export-run";
    button.textContent = "Xuất chương (.zip)";

    button.addEventListener("click", async () => {
      const chapterId = window.currentChapterId;
      if (!chapterId) return;
      button.disabled = true;
      button.textContent = "Đang kết xuất chương…";
      try {
        if (typeof window.flushAllPendingPersists === "function") {
          await window.flushAllPendingPersists();
        }
        const response = await fetch(`/api/render/chapter?chapter_id=${encodeURIComponent(chapterId)}`, {
          method: "POST",
        });
        const data = typeof window.parseApiResponse === "function"
          ? await window.parseApiResponse(response)
          : await response.json().catch(() => ({}));
        if (!response.ok) {
          const message = typeof window.getErrorMessage === "function"
            ? window.getErrorMessage(response.status, data)
            : data?.detail || `HTTP ${response.status}`;
          throw new Error(message);
        }
        if (chapterId !== window.currentChapterId) return;
        window.currentManifest = data;

        const href = data.chapter_render?.download_url || `/api/export/${encodeURIComponent(chapterId)}.zip`;
        const anchor = document.createElement("a");
        anchor.href = href;
        anchor.download = `manga-translator-${chapterId}.zip`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        if (typeof window.showToast === "function") {
          window.showToast(`Đã kết xuất ${data.chapter_render?.rendered || 0} trang.`, "info");
        }
      } catch (err) {
        if (typeof window.showToast === "function") {
          window.showToast("Xuất chương thất bại: " + err.message, "error");
        }
      } finally {
        button.disabled = false;
        button.textContent = "Xuất chương (.zip)";
      }
    });

    toolbar.appendChild(button);
  }

  const observer = new MutationObserver(() => {
    document.querySelectorAll(".translation-sticky-toolbar").forEach(buildExportButton);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".translation-sticky-toolbar").forEach(buildExportButton);
  });
})();
