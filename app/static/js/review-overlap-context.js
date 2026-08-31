(() => {
  const host = document.getElementById("page-view");
  if (!host) return;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  let scanQueued = false;

  function addContextBand(wrap, start, end, imageHeight, edge) {
    if (end <= start || imageHeight <= 0) return;
    const band = document.createElement("div");
    band.className = `review-context-band review-context-band-${edge}`;
    band.setAttribute("aria-hidden", "true");
    band.style.top = `${(start / imageHeight) * 100}%`;
    band.style.height = `${((end - start) / imageHeight) * 100}%`;

    const label = document.createElement("span");
    label.className = "review-context-band-label";
    label.textContent = "Ngữ cảnh detector · không xuất";
    band.appendChild(label);
    wrap.appendChild(band);
  }

  function decorateCard(card) {
    if (!(card instanceof HTMLElement)) return;
    const pageIndex = Number.parseInt(card.dataset.pageIndex || "", 10);
    if (!Number.isFinite(pageIndex)) return;

    const page = window.currentManifest?.pages?.[pageIndex];
    const core = page?.stitch_core;
    if (!core || typeof core !== "object") return;

    const wrap = card.querySelector(".review-image-wrap");
    const img = wrap?.querySelector("img");
    if (!wrap || !img) return;

    const apply = () => {
      const imageHeight = Number(img.naturalHeight || 0);
      if (!(imageHeight > 0)) return;

      const rawY1 = Number(core.core_y1);
      const rawY2 = Number(core.core_y2);
      if (!Number.isFinite(rawY1) || !Number.isFinite(rawY2)) return;

      const coreY1 = clamp(rawY1, 0, imageHeight);
      const coreY2 = clamp(rawY2, coreY1, imageHeight);
      wrap.querySelectorAll(".review-context-band").forEach((node) => node.remove());

      const hasTopContext = coreY1 > 0;
      const hasBottomContext = coreY2 < imageHeight;
      if (!hasTopContext && !hasBottomContext) return;

      if (hasTopContext) addContextBand(wrap, 0, coreY1, imageHeight, "top");
      if (hasBottomContext) addContextBand(wrap, coreY2, imageHeight, imageHeight, "bottom");

      const pageLabelEl = card.querySelector(".page-block-label");
      if (pageLabelEl && !pageLabelEl.querySelector(".review-context-legend")) {
        const legend = document.createElement("span");
        legend.className = "review-context-legend";
        legend.textContent = "Dải gạch là overlap dùng làm ngữ cảnh; pixel đó thuộc lát kế bên khi xuất.";
        pageLabelEl.appendChild(legend);
      }
    };

    if (img.complete && img.naturalHeight > 0) apply();
    else img.addEventListener("load", apply, { once: true });
  }

  function scan() {
    scanQueued = false;
    host.querySelectorAll(".review-card").forEach(decorateCard);
  }

  const observer = new MutationObserver(() => {
    if (scanQueued) return;
    scanQueued = true;
    window.requestAnimationFrame(scan);
  });
  observer.observe(host, { childList: true, subtree: true });
  scan();
})();
