(() => {
  async function renderTranslations45(pageIndex) {
    try {
      if (typeof window.flushAllPendingPersists === "function") {
        await window.flushAllPendingPersists(pageIndex);
      } else if (typeof window.flushTextObjectPersist === "function") {
        await window.flushTextObjectPersist(pageIndex);
      }
    } catch (_) {
      showToast("Không thể chèn chữ vì lưu dữ liệu không thành công.", "error");
      return;
    }

    const page = currentManifest?.pages?.[pageIndex];
    if (!page) return;

    const translations = {};
    const colors = {};
    const fonts = {};
    const font_sizes = {};
    const bolds = {};
    const stroke_widths = {};
    const stroke_colors = {};
    const bg_colors = {};
    const corner_radii = {};
    const horizontal_aligns = {};
    const vertical_aligns = {};

    (page.text_objects || []).forEach((obj) => {
      if (!obj?.translation?.trim()) return;
      translations[obj.id] = obj.translation.trim();
      const style = obj.style || {};
      colors[obj.id] = style.color || "auto";
      fonts[obj.id] = style.font || "default";
      font_sizes[obj.id] = style.fontSize || "auto";
      bolds[obj.id] = style.bold === true;
      stroke_widths[obj.id] = style.strokeWidth || "auto";
      stroke_colors[obj.id] = style.strokeColor || "auto";
      bg_colors[obj.id] = style.bgColor || "transparent";
      corner_radii[obj.id] = parseInt(style.cornerRadius || "0", 10);
      horizontal_aligns[obj.id] = ["left", "center", "right"].includes(style.horizontalAlign)
        ? style.horizontalAlign
        : "center";
      vertical_aligns[obj.id] = ["top", "middle", "bottom"].includes(style.verticalAlign)
        ? style.verticalAlign
        : "middle";
    });

    const btn = document.querySelector(".editor-render-btn");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Đang kết xuất…";
    }

    try {
      const resp = await fetch("/api/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chapter_id: currentChapterId,
          page_index: pageIndex,
          translations,
          colors,
          fonts,
          font_sizes,
          bolds,
          stroke_widths,
          stroke_colors,
          bg_colors,
          corner_radii,
          horizontal_aligns,
          vertical_aligns,
        }),
      });
      const data = await parseApiResponse(resp);
      if (!resp.ok) {
        showToast("Kết xuất ảnh thất bại: " + getErrorMessage(resp.status, data), "error");
        return;
      }

      const currentPage = currentManifest?.pages?.[pageIndex];
      if (data.committed !== true) {
        if (currentPage) currentPage.rendered = false;
        showToast(
          data.warning || "Kết quả kết xuất đã bị hủy vì dữ liệu trang thay đổi.",
          "info",
        );
        return;
      }

      if (currentPage) {
        currentPage.rendered = true;
        if (Number.isInteger(data.render_revision)) {
          currentPage.render_revision = data.render_revision;
        }
      }
      showRenderResult(pageIndex, data.output);
    } catch (err) {
      showToast("Kết xuất ảnh thất bại: " + err.message, "error");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Kết xuất bản dịch";
      }
    }
  }

  window.renderTranslations = renderTranslations45;
})();
