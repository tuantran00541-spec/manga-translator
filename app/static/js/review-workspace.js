// review-workspace.js - UI-03 review navigation and brush safety layer
(() => {
  const legacyRenderReview = window.renderReview;
  if (typeof legacyRenderReview !== "function") return;

  let activeReviewIndex = 0;

  function stopActiveBrushes(cards) {
    cards.forEach((card) => {
      const btn = card.querySelector(".brush-toggle-btn");
      if (btn && btn.textContent.startsWith("Đang tô")) btn.click();
      const canvas = card.querySelector("canvas.brush-canvas");
      if (canvas && canvas._brushAbort) canvas._brushAbort.abort();
    });
  }

  function setupReviewWorkspace() {
    const container = document.getElementById("page-view");
    if (!container) return;
    const cards = [...container.querySelectorAll(".review-card")];
    if (!cards.length) return;

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

    const nav = document.createElement("div");
    nav.className = "review-page-nav";
    const prev = document.createElement("button");
    prev.className = "review-nav-btn";
    prev.textContent = "← Trước";
    const position = document.createElement("span");
    position.className = "review-position";
    const next = document.createElement("button");
    next.className = "review-nav-btn";
    next.textContent = "Sau →";
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
      position.textContent = `${activeReviewIndex + 1} / ${cards.length}${label ? " · " + label.textContent : ""}`;
      prev.disabled = activeReviewIndex === 0;
      next.disabled = activeReviewIndex === cards.length - 1;
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
