(() => {
  const DUPLICATE_OFFSET = 12;

  async function apiTextObject(action, payload) {
    const resp = await fetch(`/api/text_object/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await (window.parseApiResponse
      ? window.parseApiResponse(resp)
      : resp.json().catch(() => ({})));
    if (!resp.ok) {
      const message = window.getErrorMessage
        ? window.getErrorMessage(resp.status, data)
        : ((data && data.detail) || `HTTP ${resp.status}`);
      throw new Error(message);
    }
    return data;
  }

  function findPageImage(pageIndex) {
    return document.querySelector(
      `.translation-canvas-host .page-block-wrapper[data-page-index="${pageIndex}"] .page-image-wrap img`
    );
  }

  async function duplicateTextObject(pageIndex, id) {
    const source = window.findTextObject?.(pageIndex, id);
    if (!source?.region) throw new Error("Không tìm thấy text object để nhân bản");

    await Promise.all([
      window.flushTextObjectPersist?.(pageIndex) || Promise.resolve(),
      window.flushGeomPersist?.(pageIndex) || Promise.resolve(),
    ]);

    const page = window.currentManifest?.pages?.[pageIndex];
    const image = findPageImage(pageIndex);
    const width = image?.naturalWidth || page?.width || 0;
    const height = image?.naturalHeight || page?.height || 0;
    if (!width || !height) throw new Error("Không xác định được kích thước ảnh");

    const r = source.region;
    const w = r.x2 - r.x1;
    const h = r.y2 - r.y1;
    let x1 = Math.min(r.x1 + DUPLICATE_OFFSET, Math.max(0, width - w));
    let y1 = Math.min(r.y1 + DUPLICATE_OFFSET, Math.max(0, height - h));
    if (x1 === r.x1 && width > w) x1 = Math.max(0, r.x1 - DUPLICATE_OFFSET);
    if (y1 === r.y1 && height > h) y1 = Math.max(0, r.y1 - DUPLICATE_OFFSET);

    const existingIds = new Set((page?.text_objects || []).map((obj) => obj?.id).filter(Boolean));
    const created = await apiTextObject("create", {
      chapter_id: window.currentChapterId,
      page_index: pageIndex,
      shape: source.shape || "rectangle",
      region: { x1, y1, x2: x1 + w, y2: y1 + h },
    });
    const clone = (created.pages?.[pageIndex]?.text_objects || []).find(
      (obj) => obj?.id && !existingIds.has(obj.id),
    );
    if (!clone) throw new Error("Không xác định được text object mới");

    try {
      const updated = await apiTextObject("update", {
        chapter_id: window.currentChapterId,
        page_index: pageIndex,
        id: clone.id,
        ocr_text: source.ocr_text || "",
        translation: source.translation || "",
        style: JSON.parse(JSON.stringify(source.style || window.DEFAULT_TEXT_OBJECT_STYLE)),
      });
      window.editorState.selectedTextObjectId = clone.id;
      window.applyManifestResponse(updated, pageIndex, { id: clone.id });
    } catch (err) {
      try {
        await apiTextObject("delete", {
          chapter_id: window.currentChapterId,
          page_index: pageIndex,
          id: clone.id,
        });
      } catch (_) {
        // Best-effort rollback; preserve the original failure.
      }
      throw err;
    }
  }

  function installPanelActions() {
    const panel = document.querySelector(".translation-panel-host .text-editor-panel");
    if (!panel || panel.dataset.ui09Ready === "1") return;
    const pageIndex = Number(panel.dataset.pageIndex);
    const id = panel.dataset.objectId;
    if (!id) return;
    const actions = panel.querySelector(".text-object-actions");
    if (!actions) return;
    panel.dataset.ui09Ready = "1";

    const duplicate = document.createElement("button");
    duplicate.type = "button";
    duplicate.className = "text-object-action-btn";
    duplicate.textContent = "Nhân bản";
    duplicate.title = "Tạo bản sao lệch nhẹ, giữ nguyên nội dung và kiểu chữ";
    duplicate.addEventListener("click", () => {
      duplicateTextObject(pageIndex, id).catch((err) => {
        window.showToast?.("Nhân bản text object thất bại: " + err.message, "error");
      });
    });

    const deleteButton = actions.querySelector(".text-object-action-btn.danger");
    if (deleteButton) {
      const replacement = deleteButton.cloneNode(true);
      deleteButton.replaceWith(replacement);
      replacement.addEventListener("click", () => {
        if (!window.confirm("Xóa text object này? Thao tác này không thể hoàn tác.")) return;
        window.deleteTextObject(pageIndex, id).catch((err) => {
          window.showToast?.("Xóa text object thất bại: " + err.message, "error");
        });
      });
    }
    actions.insertBefore(duplicate, actions.firstChild);
  }

  const originalRenderEditorPanel = window.renderEditorPanel;
  if (typeof originalRenderEditorPanel === "function") {
    window.renderEditorPanel = function ui09RenderEditorPanel(pageIndex) {
      originalRenderEditorPanel(pageIndex);
      installPanelActions();
    };
  }

  document.addEventListener("dblclick", (event) => {
    const overlay = event.target.closest?.(".text-object-overlay:not(.drawing)");
    if (!overlay) return;
    const pageIndex = Number(overlay.dataset.pageIndex);
    const id = overlay.dataset.objectId;
    if (!Number.isInteger(pageIndex) || !id) return;
    event.preventDefault();
    event.stopPropagation();
    window.setSelectedTextObject?.(pageIndex, id);
    requestAnimationFrame(() => {
      const textarea = document.querySelector(
        `textarea.translation-textarea[data-text-object-id="${CSS.escape(id)}"]`
      );
      textarea?.focus();
      textarea?.select();
    });
  }, true);

  window.duplicateTextObject = duplicateTextObject;
  window.ui09 = { duplicateTextObject };
})();
