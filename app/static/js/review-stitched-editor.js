(() => {
  const SOURCE_MASKS = new Map();
  const PAINT_THRESHOLD = 0.75;
  let brushOn = false;
  let painting = false;
  let brushRadius = 20;
  let lastPoint = null;
  let lastClickAt = 0;
  let lastClick = { x: 0, y: 0 };

  function reviewHost(workspace) {
    return workspace?.closest("#page-view.review-mode") || null;
  }

  function isStitched(workspace) {
    return Boolean(reviewHost(workspace)?.classList.contains("review-show-stitched"));
  }

  function sourcePage(workspace) {
    const select = workspace.querySelector(".review-stitched-select");
    const selected = Number.parseInt(select?.value || "", 10);
    if (Number.isFinite(selected)) return selected;
    const cardIndex = Number.parseInt(workspace.querySelector(".review-card")?.dataset.pageIndex || "", 10);
    const page = Number.isFinite(cardIndex) ? window.currentManifest?.pages?.[cardIndex] : null;
    return Number.isInteger(page?.source_page) ? page.source_page : cardIndex;
  }

  function sourceItems(workspace) {
    const source = sourcePage(workspace);
    return (window.currentManifest?.pages || [])
      .map((page, canonicalIndex) => ({ page, canonicalIndex }))
      .filter(({ page, canonicalIndex }) => {
        const value = Number.isInteger(page?.source_page) ? page.source_page : canonicalIndex;
        return value === source;
      })
      .sort((a, b) => (Number(a.page?.slice_index) || 0) - (Number(b.page?.slice_index) || 0));
  }

  function controls(workspace) {
    return {
      brushBtn: workspace.querySelector(".brush-toggle-btn"),
      clearBtn: workspace.querySelector(".clear-brush-btn"),
      repaintBtn: workspace.querySelector(".repaint-btn"),
      resetBtn: workspace.querySelector(".reset-manual-btn"),
      brushSize: workspace.querySelector(".brush-size-slider"),
      brushSizeValue: workspace.querySelector(".brush-size-value"),
      aiQcBtn: workspace.querySelector(".ai-qc-btn"),
    };
  }

  function layers(workspace) {
    return [...workspace.querySelectorAll(".review-stitched-mask-editor")]
      .map((mask) => ({
        mask,
        ctx: mask.getContext("2d"),
        base: mask.parentElement?.querySelector(".review-stitched-base-editor"),
        y1: Number(mask.dataset.sourceY) || 0,
        y2: (Number(mask.dataset.sourceY) || 0) + mask.height,
      }))
      .sort((a, b) => a.y1 - b.y1);
  }

  function sourceDimensions(workspace) {
    const chunks = layers(workspace);
    if (!chunks.length) return null;
    return {
      width: chunks[0].mask.width,
      height: Math.max(...chunks.map((chunk) => chunk.y2)),
      chunks,
    };
  }

  function validCore(page) {
    const core = page?.stitch_core;
    if (!core || typeof core !== "object") return null;
    const sourceY1 = Number(core.core_source_y1);
    const sourceY2 = Number(core.core_source_y2);
    const localY1 = Number(core.core_y1);
    const localY2 = Number(core.core_y2);
    const sliceSourceY1 = Number(core.source_y1);
    const sliceSourceY2 = Number(core.source_y2);
    if (![sourceY1, sourceY2, localY1, localY2, sliceSourceY1, sliceSourceY2].every(Number.isFinite)) return null;
    if (sourceY2 <= sourceY1 || localY2 <= localY1 || sliceSourceY2 <= sliceSourceY1) return null;
    if ((sourceY2 - sourceY1) !== (localY2 - localY1)) return null;
    return { sourceY1, sourceY2, localY1, localY2, sliceSourceY1, sliceSourceY2 };
  }

  function sourceMapping(workspace) {
    const dims = sourceDimensions(workspace);
    const items = sourceItems(workspace);
    if (!dims || !items.length) return null;
    if (items.length === 1 && !validCore(items[0].page)) {
      return {
        ...dims,
        items: [{
          ...items[0],
          sourceY1: 0,
          sourceY2: dims.height,
          localY1: 0,
          localY2: dims.height,
          sliceHeight: dims.height,
          physicalSourceY1: 0,
          core: null,
        }],
      };
    }

    let expected = 0;
    const mapped = [];
    for (const item of items) {
      const core = validCore(item.page);
      if (!core || core.sourceY1 !== expected) return null;
      mapped.push({
        ...item,
        sourceY1: core.sourceY1,
        sourceY2: core.sourceY2,
        localY1: core.localY1,
        localY2: core.localY2,
        sliceHeight: core.sliceSourceY2 - core.sliceSourceY1,
        physicalSourceY1: core.sliceSourceY1,
        core,
      });
      expected = core.sourceY2;
    }
    if (expected !== dims.height) return null;
    return { ...dims, items: mapped };
  }

  function updateBrushUi(workspace) {
    const { brushBtn, brushSize, brushSizeValue } = controls(workspace);
    workspace.querySelector(".review-stitched-shell")?.classList.toggle("review-stitched-brush-mode", brushOn);
    for (const { mask } of layers(workspace)) {
      mask.style.pointerEvents = brushOn && isStitched(workspace) ? "auto" : "none";
      mask.style.cursor = brushOn && isStitched(workspace) ? "crosshair" : "default";
    }
    if (brushBtn) {
      brushBtn.classList.toggle("ui-btn-primary", brushOn);
      brushBtn.classList.toggle("ui-btn-ghost", !brushOn);
      brushBtn.textContent = brushOn ? "Đang đánh dấu · Chọn để kết thúc" : "Đánh dấu vùng lỗi";
      brushBtn.setAttribute("aria-pressed", String(brushOn));
    }
    if (brushSize) {
      brushSize.value = String(brushRadius);
      const wrap = brushSize.closest(".brush-size-control");
      if (wrap) wrap.hidden = !brushOn;
    }
    if (brushSizeValue) brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;
  }

  function stopPainting() {
    painting = false;
    lastPoint = null;
  }

  function setBrush(workspace, value) {
    brushOn = Boolean(value);
    if (!brushOn) stopPainting();
    updateBrushUi(workspace);
  }

  function markDirty(mask) {
    mask._reviewDirty = true;
  }

  function paintDot(workspace, x, y) {
    const dims = sourceDimensions(workspace);
    if (!dims) return;
    const px = Math.max(0, Math.min(dims.width, x));
    const py = Math.max(0, Math.min(dims.height, y));
    for (const chunk of dims.chunks) {
      if (py + brushRadius < chunk.y1 || py - brushRadius > chunk.y2) continue;
      chunk.ctx.fillStyle = "rgba(220, 38, 38, 0.7)";
      chunk.ctx.beginPath();
      chunk.ctx.arc(px, py - chunk.y1, brushRadius, 0, Math.PI * 2);
      chunk.ctx.fill();
      markDirty(chunk.mask);
    }
  }

  function paintStroke(workspace, from, to) {
    const distance = Math.hypot(to.x - from.x, to.y - from.y);
    const steps = Math.max(1, Math.ceil(distance / Math.max(2, brushRadius * 0.4)));
    for (let i = 1; i <= steps; i += 1) {
      const t = i / steps;
      paintDot(workspace, from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t);
    }
  }

  function eventPoint(event, mask) {
    const rect = mask.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const scaleX = mask.width / rect.width;
    const scaleY = mask.height / rect.height;
    const y1 = Number(mask.dataset.sourceY) || 0;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: y1 + (event.clientY - rect.top) * scaleY,
      localY: (event.clientY - rect.top) * scaleY,
    };
  }

  function floodFill(workspace, layer, startX, startY) {
    const base = layer.base;
    if (!base) return false;
    const width = base.width;
    const height = base.height;
    const x0 = Math.max(0, Math.min(width - 1, Math.round(startX)));
    const y0 = Math.max(0, Math.min(height - 1, Math.round(startY)));
    let source;
    try {
      source = base.getContext("2d").getImageData(0, 0, width, height);
    } catch (err) {
      console.warn("Cannot read stitched pixels for flood fill:", err);
      return false;
    }
    const data = source.data;
    const root = (y0 * width + x0) * 4;
    const target = [data[root], data[root + 1], data[root + 2]];
    const tolerance = 22;
    const cap = Math.min(Math.round(width * height * 0.15), 260000);
    const visited = new Uint8Array(width * height);
    const stack = [[x0, y0]];
    const points = [];
    visited[y0 * width + x0] = 1;
    const matches = (index) => (
      Math.abs(data[index] - target[0]) <= tolerance
      && Math.abs(data[index + 1] - target[1]) <= tolerance
      && Math.abs(data[index + 2] - target[2]) <= tolerance
    );
    while (stack.length) {
      const [x, y] = stack.pop();
      if (!matches((y * width + x) * 4)) continue;
      points.push([x, y]);
      if (points.length > cap) {
        showToast("Vùng màu lan quá rộng. Hãy kéo cọ để đánh dấu thủ công.", "error");
        return false;
      }
      for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const flat = ny * width + nx;
        if (visited[flat]) continue;
        visited[flat] = 1;
        if (matches(flat * 4)) stack.push([nx, ny]);
      }
    }
    if (!points.length) return false;
    layer.ctx.fillStyle = "rgba(220, 38, 38, 0.7)";
    for (const [x, y] of points) layer.ctx.fillRect(x, y, 1, 1);
    markDirty(layer.mask);
    return true;
  }

  function bindMaskEvents(workspace, mask) {
    if (mask.dataset.editorBound === "1") return;
    mask.dataset.editorBound = "1";
    mask.addEventListener("pointerdown", (event) => {
      if (!isStitched(workspace) || !brushOn || event.button !== 0) return;
      event.preventDefault();
      const point = eventPoint(event, mask);
      if (!point) return;
      const now = Date.now();
      const doubleClick = now - lastClickAt < 350 && Math.hypot(point.x - lastClick.x, point.y - lastClick.y) < 20;
      lastClickAt = doubleClick ? 0 : now;
      lastClick = { x: point.x, y: point.y };
      const layer = layers(workspace).find((item) => item.mask === mask);
      if (doubleClick && layer) {
        stopPainting();
        floodFill(workspace, layer, point.x, point.localY);
        return;
      }
      painting = true;
      lastPoint = { x: point.x, y: point.y };
      try { mask.setPointerCapture(event.pointerId); } catch (_) {}
      paintDot(workspace, point.x, point.y);
    });
    mask.addEventListener("pointermove", (event) => {
      if (!brushOn || !painting || !lastPoint) return;
      const point = eventPoint(event, mask);
      if (!point) return;
      const next = { x: point.x, y: point.y };
      paintStroke(workspace, lastPoint, next);
      lastPoint = next;
    });
    mask.addEventListener("pointerup", stopPainting);
    mask.addEventListener("pointercancel", stopPainting);
    mask.addEventListener("wheel", (event) => {
      if (!brushOn || !isStitched(workspace)) return;
      event.preventDefault();
      brushRadius = Math.max(8, Math.min(80, brushRadius + (event.deltaY < 0 ? 2 : -2)));
      updateBrushUi(workspace);
    }, { passive: false });
  }

  function captureMask(workspace) {
    const source = sourcePage(workspace);
    if (!Number.isFinite(source)) return;
    const chunks = layers(workspace);
    const dirty = chunks.filter((chunk) => chunk.mask._reviewDirty);
    if (!dirty.length) {
      SOURCE_MASKS.delete(source);
      return;
    }
    const snapshot = dirty.map((chunk) => ({ y1: chunk.y1, data: chunk.mask.toDataURL("image/png") }));
    SOURCE_MASKS.set(source, snapshot);
  }

  function restoreMask(workspace) {
    const snapshot = SOURCE_MASKS.get(sourcePage(workspace));
    if (!snapshot?.length) return;
    const targets = new Map(layers(workspace).map((chunk) => [chunk.y1, chunk]));
    for (const item of snapshot) {
      const target = targets.get(item.y1);
      if (!target) continue;
      const image = new Image();
      image.onload = () => {
        if (!target.mask.isConnected) return;
        target.ctx.drawImage(image, 0, 0, target.mask.width, target.mask.height);
        markDirty(target.mask);
      };
      image.src = item.data;
    }
  }

  function clearMask(workspace) {
    for (const chunk of layers(workspace)) {
      chunk.ctx.clearRect(0, 0, chunk.mask.width, chunk.mask.height);
      chunk.mask._reviewDirty = false;
    }
    SOURCE_MASKS.delete(sourcePage(workspace));
  }

  function decorate(workspace) {
    const imageHost = workspace.querySelector(".review-stitched-image");
    if (!imageHost) return;
    const directCanvases = [...imageHost.children].filter((node) => node instanceof HTMLCanvasElement);
    if (!directCanvases.length) return;
    for (const base of directCanvases) {
      const y1 = Number(base.dataset.sourceY) || 0;
      const wrapper = document.createElement("div");
      wrapper.className = "review-stitched-edit-chunk";
      wrapper.style.position = "relative";
      wrapper.style.lineHeight = "0";
      wrapper.style.width = "100%";
      base.classList.add("review-stitched-base-editor");
      base.style.position = "relative";
      base.style.zIndex = "1";
      const mask = document.createElement("canvas");
      mask.className = "review-stitched-mask-editor";
      mask.width = base.width;
      mask.height = base.height;
      mask.dataset.sourceY = String(y1);
      mask.style.position = "absolute";
      mask.style.inset = "0";
      mask.style.zIndex = "2";
      mask.style.width = "100%";
      mask.style.height = "100%";
      mask.style.touchAction = "none";
      mask.style.pointerEvents = "none";
      wrapper.append(base, mask);
      imageHost.appendChild(wrapper);
      bindMaskEvents(workspace, mask);
    }
    restoreMask(workspace);
    updateBrushUi(workspace);
  }

  function hasPaint(workspace) {
    return layers(workspace).some((chunk) => chunk.mask._reviewDirty);
  }

  function canvasHasPaint(canvas) {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 20) return true;
    return false;
  }

  function sliceMask(workspace, item) {
    const mapping = sourceMapping(workspace);
    if (!mapping) return null;
    const entry = mapping.items.find((candidate) => candidate.canonicalIndex === item.canonicalIndex);
    if (!entry) return null;
    const canvas = document.createElement("canvas");
    canvas.width = mapping.width;
    canvas.height = entry.sliceHeight;
    const ctx = canvas.getContext("2d");
    for (const chunk of mapping.chunks) {
      const y1 = Math.max(entry.sourceY1, chunk.y1);
      const y2 = Math.min(entry.sourceY2, chunk.y2);
      if (y2 <= y1) continue;
      const h = y2 - y1;
      ctx.drawImage(
        chunk.mask,
        0,
        y1 - chunk.y1,
        mapping.width,
        h,
        0,
        entry.localY1 + (y1 - entry.sourceY1),
        mapping.width,
        h,
      );
    }
    return { canvas, entry, mapping };
  }

  function toBlob(canvas) {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Không thể tạo mask PNG")), "image/png");
    });
  }

  function setBusy(workspace, value) {
    const card = workspace.querySelector(".review-card");
    if (card) {
      card._reviewBusy = Boolean(value);
      card._syncReviewBusy?.();
    }
    workspace.querySelector(".review-stitched-shell")?.classList.toggle("is-busy", Boolean(value));
  }

  async function parseResponse(response) {
    const parse = typeof window.parseApiResponse === "function"
      ? window.parseApiResponse
      : async (r) => (await r.json().catch(() => ({})));
    const data = await parse(response);
    if (!response.ok) {
      const getErr = typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, payload) => payload?.detail || `Server trả về ${status}`;
      throw new Error(getErr(response.status, data));
    }
    return data;
  }

  async function repaint(workspace) {
    if (!hasPaint(workspace)) {
      showToast("Chưa có vùng nào được đánh dấu để xử lý.", "error");
      return;
    }
    const mapping = sourceMapping(workspace);
    if (!mapping) {
      showToast("Không thể ánh xạ ảnh ghép về các lát vì stitch_core không hợp lệ.", "error");
      return;
    }
    const choose = typeof chooseRepaintMode === "function" ? chooseRepaintMode : null;
    if (!choose || !window.currentChapterId) return;
    const repaintMode = await choose();
    if (!repaintMode || !isStitched(workspace)) return;

    setBrush(workspace, false);
    setBusy(workspace, true);
    const { repaintBtn } = controls(workspace);
    if (repaintBtn) repaintBtn.textContent = repaintMode === "lama" ? "LaMa đang xử lý…" : "Đang xử lý…";
    let processed = 0;
    let skippedTouched = false;
    try {
      for (const item of mapping.items) {
        const local = sliceMask(workspace, item);
        if (!local || !canvasHasPaint(local.canvas)) continue;
        if (item.page.skipped) {
          skippedTouched = true;
          continue;
        }
        const formData = new FormData();
        formData.append("chapter_id", window.currentChapterId);
        formData.append("page_index", String(item.canonicalIndex));
        formData.append("mode", repaintMode);
        formData.append("mask", await toBlob(local.canvas), "mask.png");
        const manifest = await parseResponse(await fetch("/api/repaint_mask", { method: "POST", body: formData }));
        currentManifest = manifest;
        window.currentManifest = manifest;
        processed += 1;
      }
      if (!processed) {
        showToast(skippedTouched ? "Vùng đánh dấu chỉ nằm trên lát đang bỏ qua." : "Mask không chạm lát nào có thể xử lý.", "error");
        return;
      }
      clearMask(workspace);
      showToast(`Đã xử lý ${processed} lát từ một thao tác trên ảnh ghép.`, "success");
      window.renderReview?.();
    } catch (err) {
      showToast("Không thể xử lý vùng đánh dấu: " + err.message, "error");
    } finally {
      if (repaintBtn) repaintBtn.textContent = "Xử lý vùng đánh dấu";
      setBusy(workspace, false);
    }
  }

  async function resetManual(workspace) {
    const items = sourceItems(workspace).filter(({ page }) => !page.skipped);
    if (!items.length || !window.currentChapterId) return;
    setBrush(workspace, false);
    setBusy(workspace, true);
    const { resetBtn } = controls(workspace);
    const previous = resetBtn?.textContent;
    if (resetBtn) resetBtn.textContent = "Đang xóa vùng chỉnh sửa…";
    try {
      for (const item of items) {
        const manifest = await parseResponse(await fetch("/api/reset_manual_mask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chapter_id: window.currentChapterId, page_index: item.canonicalIndex }),
        }));
        currentManifest = manifest;
        window.currentManifest = manifest;
      }
      clearMask(workspace);
      showToast("Đã xóa vùng chỉnh sửa thủ công trên toàn trang ảnh gốc.", "success");
      window.renderReview?.();
    } catch (err) {
      showToast("Không thể xóa vùng chỉnh sửa: " + err.message, "error");
    } finally {
      if (resetBtn) resetBtn.textContent = previous || "Xóa vùng chỉnh sửa";
      setBusy(workspace, false);
    }
  }

  function paintPolygon(workspace, points) {
    if (points.length < 3) return false;
    const minY = Math.min(...points.map((point) => point[1]));
    const maxY = Math.max(...points.map((point) => point[1]));
    let painted = false;
    for (const chunk of layers(workspace)) {
      if (maxY < chunk.y1 || minY > chunk.y2) continue;
      chunk.ctx.save();
      chunk.ctx.fillStyle = "rgba(220, 38, 38, 0.52)";
      chunk.ctx.strokeStyle = "rgba(248, 113, 113, 0.95)";
      chunk.ctx.lineWidth = 2;
      chunk.ctx.beginPath();
      chunk.ctx.moveTo(points[0][0], points[0][1] - chunk.y1);
      for (let i = 1; i < points.length; i += 1) chunk.ctx.lineTo(points[i][0], points[i][1] - chunk.y1);
      chunk.ctx.closePath();
      chunk.ctx.fill();
      chunk.ctx.stroke();
      chunk.ctx.restore();
      markDirty(chunk.mask);
      painted = true;
    }
    return painted;
  }

  async function aiQc(workspace) {
    const mapping = sourceMapping(workspace);
    if (!mapping || !window.currentChapterId) {
      showToast("Không thể ánh xạ AI QC trên ảnh ghép vì stitch_core không hợp lệ.", "error");
      return;
    }
    setBrush(workspace, false);
    setBusy(workspace, true);
    const { aiQcBtn } = controls(workspace);
    const previous = aiQcBtn?.textContent;
    if (aiQcBtn) aiQcBtn.textContent = "AI đang kiểm tra…";
    let painted = 0;
    let uncertain = 0;
    let artDamage = 0;
    try {
      for (const item of mapping.items) {
        if (item.page.skipped) continue;
        const data = await parseResponse(await fetch("/api/visual_qc/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chapter_id: window.currentChapterId, page_index: item.canonicalIndex }),
        }));
        for (const issue of Array.isArray(data.issues) ? data.issues : []) {
          if (issue.issue_type === "over_erased_art") {
            artDamage += 1;
            continue;
          }
          if ((Number(issue.confidence) || 0) < PAINT_THRESHOLD) {
            uncertain += 1;
            continue;
          }
          const polygon = Array.isArray(issue.polygon) ? issue.polygon : [];
          if (polygon.length < 3) continue;
          const points = polygon.map((point) => [
            Number(point?.[0]) || 0,
            item.physicalSourceY1 + (Number(point?.[1]) || 0),
          ]);
          const centerY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
          if (item.core && (centerY < item.sourceY1 || centerY >= item.sourceY2)) continue;
          if (paintPolygon(workspace, points)) painted += 1;
        }
      }
      if (painted) {
        const extra = [];
        if (uncertain) extra.push(`${uncertain} vùng độ tin cậy thấp`);
        if (artDamage) extra.push(`${artDamage} vùng nghi mất chi tiết`);
        showToast(`AI đã đánh dấu ${painted} vùng trên ảnh ghép${extra.length ? ` (${extra.join(", ")})` : ""}.`, "info");
      } else if (uncertain || artDamage) {
        showToast(`AI phát hiện ${uncertain} vùng độ tin cậy thấp và ${artDamage} vùng nghi mất chi tiết; chưa tự tô.`, "info");
      } else {
        showToast("Không phát hiện vùng cần xử lý lại trên trang ảnh gốc.", "success");
      }
    } catch (err) {
      showToast("Không thể hoàn tất kiểm tra bằng AI: " + err.message, "error");
    } finally {
      if (aiQcBtn) aiQcBtn.textContent = previous || "Kiểm tra trang bằng AI";
      setBusy(workspace, false);
    }
  }

  function mount(workspace) {
    if (!(workspace instanceof HTMLElement) || workspace.dataset.stitchedEditorMounted === "1") return;
    workspace.dataset.stitchedEditorMounted = "1";

    workspace.addEventListener("click", (event) => {
      const modeButton = event.target.closest("button[data-review-mode]");
      if (modeButton) {
        captureMask(workspace);
        setBrush(workspace, false);
        if (modeButton.dataset.reviewMode === "stitched" && !isStitched(workspace)) {
          setTimeout(() => window.renderReview?.(), 0);
        }
        return;
      }
      if (!isStitched(workspace)) return;
      const control = event.target.closest(
        ".brush-toggle-btn, .clear-brush-btn, .repaint-btn, .reset-manual-btn, .ai-qc-btn",
      );
      if (!control || !workspace.contains(control)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (control.matches(".brush-toggle-btn")) setBrush(workspace, !brushOn);
      else if (control.matches(".clear-brush-btn")) clearMask(workspace);
      else if (control.matches(".repaint-btn")) void repaint(workspace);
      else if (control.matches(".reset-manual-btn")) void resetManual(workspace);
      else if (control.matches(".ai-qc-btn")) void aiQc(workspace);
    }, true);

    workspace.addEventListener("input", (event) => {
      if (!isStitched(workspace) || !event.target.matches(".brush-size-slider")) return;
      event.stopImmediatePropagation();
      brushRadius = Math.max(8, Math.min(80, Number(event.target.value) || 20));
      updateBrushUi(workspace);
    }, true);

    const shell = workspace.querySelector(".review-stitched-shell");
    if (shell) {
      shell.addEventListener("click", (event) => {
        if (event.target.closest(".review-stitched-toolbar")) captureMask(workspace);
      }, true);
    }

    const observer = new MutationObserver(() => decorate(workspace));
    const imageHost = workspace.querySelector(".review-stitched-image");
    if (imageHost) observer.observe(imageHost, { childList: true });
    decorate(workspace);
  }

  function scan() {
    document.querySelectorAll("#page-view.review-mode .review-workspace-shell").forEach(mount);
  }

  window.addEventListener("blur", stopPainting);
  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scan, { once: true });
  else scan();
})();
