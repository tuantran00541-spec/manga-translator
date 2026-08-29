(() => {
  const PROCESS_BATCH_SIZE = 16;

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

  function appendText(parent, tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    parent.appendChild(node);
    return node;
  }

  // Recent-chapter values originate in persisted manifests, including source_url.
  // Build the card with textContent instead of interpolating them into innerHTML.
  window.loadRecentChapters = async function safeLoadRecentChapters() {
    const container = document.getElementById("recent-chapters");
    if (!container) return;
    const panel = container.closest(".recent-panel");
    const setPanelVisible = (visible) => {
      if (panel) panel.hidden = !visible;
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
      appendText(container, "div", "recent-title", "Chương đang xử lý");
      const list = document.createElement("div");
      list.className = "recent-list";
      const stageLabels = {
        preview: "Xử lý ảnh",
        review: "Kiểm tra chất lượng",
        editor: "Biên tập bản dịch",
      };

      chapters.forEach((ch) => {
        const card = document.createElement("div");
        card.className = "recent-card";
        const info = document.createElement("div");
        info.className = "recent-info";

        appendText(info, "strong", "", String(ch?.chapter_id || "(không rõ chương)"));
        info.appendChild(document.createElement("br"));
        appendText(
          info,
          "span",
          "recent-url",
          String(ch?.source_url || "(không có liên kết nguồn)"),
        );
        info.appendChild(document.createElement("br"));

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

        const btn = document.createElement("button");
        btn.className = "recent-resume-btn";
        btn.type = "button";
        btn.textContent = "Tiếp tục xử lý";
        btn.addEventListener("click", () => resumeChapter(String(ch?.chapter_id || "")));
        card.appendChild(btn);
        list.appendChild(card);
      });

      container.appendChild(list);
    } catch (err) {
      showToast("Không tải được danh sách chương: " + err.message, "error");
      container.replaceChildren();
      setPanelVisible(false);
    }
  };
})();
