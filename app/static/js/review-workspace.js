(() => {
  const legacyRenderReview = window.renderReview;
  if (typeof legacyRenderReview !== "function") return;

  let activeReviewIndex = 0;
  let reviewLastChapterId = null;

  function stopActiveBrushes(cards) {
    cards.forEach((card) => {
      const canvas = card.querySelector("canvas.brush-canvas");
      if (canvas && typeof canvas._stopBrush === "function") {
        canvas._stopBrush();
      } else {
        const btn = card.querySelector(".brush-toggle-btn");
        if (btn && btn.textContent.startsWith("Đang tô")) btn.click();
      }
    });
  }

  function setupReviewWorkspace() {
    const container = document.getElementById("page-view");
    if (!container) return;

    if (window.currentChapterId && reviewLastChapterId !== window.currentChapterId) {
      reviewLastChapterId = window.currentChapterId;
      activeReviewIndex = 0;
    }

    const cards = [...container.querySelectorAll(".review-card")];
    if (!cards.length) return;

    const targetCanonical = window.initialReviewCanonicalPageIndex !== undefined && window.initialReviewCanonicalPageIndex !== null
      ? window.initialReviewCanonicalPageIndex
      : (window.currentManifest?.workflow?.stage === "review" && window.currentManifest?.workflow?.page_index !== undefined
          ? window.currentManifest.workflow.page_index
          : null);

    if (targetCanonical !== null) {
      const foundIdx = cards.findIndex((c) => c.dataset.pageIndex === String(targetCanonical));
      if (foundIdx !== -1) {
        activeReviewIndex = foundIdx;
      }
      window.initialReviewCanonicalPageIndex = null;
    }

    activeReviewIndex = Math.max(0, Math.min(activeReviewIndex, cards.length - 1));

    let workspace = container.querySelector(".review-workspace-shell");
    if (workspace) workspace.remove();

    workspace = document.createElement("div");
    workspace.className = "review-workspace-shell";

    const toolbar = document.createElement("div");
    toolbar.className = "review-sticky-toolbar";
    toolbar.innerHTML = '<div class="review-toolbar-title"><strong>Sửa lỗi ảnh</strong><span>Chỉ hiển thị một lát để tập trung xử lý.</span></div>';

    const controlsSlot = document.createElement("div");
    controlsSlot.className = "review-controls-slot";
    toolbar.appendChild(controlsSlot);

    const nav = document.createElement("nav");
    nav.className = "review-page-nav workspace-nav-bar";
    nav.setAttribute("aria-label", "Điều hướng trang sửa lỗi ảnh");

    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "review-nav-btn workspace-nav-btn";
    prev.textContent = "← Trước";
    prev.setAttribute("aria-label", "Trang trước");

    const position = document.createElement("div");
    position.className = "review-position workspace-nav-position";
    position.setAttribute("aria-live", "polite");

    const next = document.createElement("button");
    next.type = "button";
    next.className = "review-nav-btn workspace-nav-btn";
    next.textContent = "Sau →";
    next.setAttribute("aria-label", "Trang sau");

    nav.append(prev, position, next);

    const canvasHost = document.createElement("div");
    canvasHost.className = "review-canvas-host";

    workspace.append(toolbar, nav, canvasHost);
    container.appendChild(workspace);

    const renderActive = () => {
      stopActiveBrushes(cards);
      cards.forEach((card) => {
        if (card.parentElement === canvasHost) card.remove();
        card.style.display = "none";
      });

      const card = cards[activeReviewIndex];
      card.style.display = "flex";
      canvasHost.appendChild(card);

      controlsSlot.innerHTML = "";
      const controls = card.querySelector(".review-controls");
      if (controls) controlsSlot.appendChild(controls);

      const label = card.querySelector(".page-block-label");
      position.innerHTML = "";

      const jumpWrap = document.createElement("label");
      jumpWrap.className = "workspace-nav-jump-wrap";
      jumpWrap.textContent = "Trang ";

      const jumpInput = document.createElement("input");
      jumpInput.type = "number";
      jumpInput.min = "1";
      jumpInput.max = String(cards.length);
      jumpInput.value = String(activeReviewIndex + 1);
      jumpInput.className = "workspace-nav-jump-input";
      jumpInput.setAttribute("aria-label", "Nhảy tới số trang");

      const doJump = () => {
        const rawVal = jumpInput.value.trim();
        if (!rawVal) {
          jumpInput.value = String(activeReviewIndex + 1);
          return;
        }
        const val = parseInt(rawVal, 10);
        if (!Number.isFinite(val) || val < 1 || val > cards.length) {
          jumpInput.value = String(activeReviewIndex + 1);
          return;
        }
        if (val - 1 !== activeReviewIndex) {
          activeReviewIndex = val - 1;
          renderActive();
        }
      };

      jumpInput.addEventListener("change", doJump);
      jumpInput.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Enter") {
          e.preventDefault();
          doJump();
        }
      });

      jumpWrap.appendChild(jumpInput);

      const totalText = document.createElement("span");
      totalText.textContent = ` / ${cards.length}`;

      const labelSpan = document.createElement("span");
      labelSpan.textContent = label ? ` · ${label.textContent}` : "";

      position.append(jumpWrap, totalText, labelSpan);

      prev.disabled = activeReviewIndex === 0;
      next.disabled = activeReviewIndex === cards.length - 1;

      const canonicalIndex = parseInt(card.dataset.pageIndex, 10) || 0;
      if (typeof window.setWorkflowCheckpoint === "function") {
        window.setWorkflowCheckpoint("review", canonicalIndex);
      }

      const canvas = card.querySelector("canvas.brush-canvas");
      const img = card.querySelector("img");
      if (canvas && img) {
        const nw = img.naturalWidth || img.width;
        const nh = img.naturalHeight || img.height;
        if (nw && nh) {
          canvas.width = nw;
          canvas.height = nh;
        }
        canvas.style.width = "100%";
        canvas.style.height = "100%";
      }
    };

    prev.addEventListener("click", () => {
      if (activeReviewIndex > 0) {
        activeReviewIndex -= 1;
        renderActive();
      }
    });
    next.addEventListener("click", () => {
      if (activeReviewIndex < cards.length - 1) {
        activeReviewIndex += 1;
        renderActive();
      }
    });

    renderActive();
  }

  window.renderReview = function reviewWorkspaceRender() {
    legacyRenderReview();
    setupReviewWorkspace();
  };
})();
