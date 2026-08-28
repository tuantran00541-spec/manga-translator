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

  async function parseResponse(response) {
    return typeof window.parseApiResponse === "function"
      ? window.parseApiResponse(response)
      : response.json().catch(() => ({}));
  }

  function errorMessage(status, data) {
    return typeof window.getErrorMessage === "function"
      ? window.getErrorMessage(status, data)
      : data?.detail || `HTTP ${status}`;
  }

  function buildExportButton(toolbar) {
    if (!toolbar || toolbar.querySelector(".chapter-export-run")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ui-btn ui-btn-primary chapter-export-run";
    button.textContent = "Final QC & xuất";
    button.title = "Kết xuất phần thay đổi rồi mở Final QC trước khi đóng gói chapter";

    button.addEventListener("click", async () => {
      const chapterId = window.currentChapterId;
      if (!chapterId) return;
      button.disabled = true;
      button.textContent = "Đang chuẩn bị Final QC…";
      try {
        if (typeof window.flushAllPendingPersists === "function") {
          await window.flushAllPendingPersists();
        }
        const response = await fetch(`/api/render/chapter?chapter_id=${encodeURIComponent(chapterId)}`, {
          method: "POST",
        });
        const data = await parseResponse(response);
        if (!response.ok) throw new Error(errorMessage(response.status, data));
        if (chapterId !== window.currentChapterId) return;
        window.currentManifest = data;
        const rendered = Number(data.chapter_render?.rendered || 0);
        const reused = Number(data.chapter_render?.reused || 0);
        if (typeof window.showToast === "function") {
          window.showToast(`Final QC: kết xuất ${rendered} trang thay đổi · tái sử dụng ${reused} trang hiện hành.`, "info");
        }
        window.initialFinalQCCanonicalPageIndex = Number(window.editorState?.activePageIndex || 0);
        if (typeof window.renderFinalQC === "function") window.renderFinalQC();
      } catch (err) {
        if (typeof window.showToast === "function") {
          window.showToast("Không thể chuẩn bị Final QC: " + err.message, "error");
        }
      } finally {
        button.disabled = false;
        button.textContent = "Final QC & xuất";
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
