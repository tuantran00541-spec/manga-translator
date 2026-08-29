function createReviewCard(pageIndex, maskSnapshot = null) {
  const page = currentManifest?.pages?.[pageIndex];
  if (!page || page.skipped) return null;

  const card = document.createElement("div");
  card.className = "review-card";
  card.dataset.pageIndex = pageIndex;

  const label = document.createElement("div");
  label.className = "page-block-label";
  label.textContent = pageLabel(currentManifest.pages, pageIndex);
  card.appendChild(label);

  const controls = document.createElement("div");
  controls.className = "review-controls review-controls-top";

  const brushBtn = document.createElement("button");
  brushBtn.className = "brush-toggle-btn";
  brushBtn.textContent = "Đánh dấu vùng lỗi";
  controls.appendChild(brushBtn);

  const clearBtn = document.createElement("button");
  clearBtn.className = "clear-brush-btn";
  clearBtn.textContent = "Xóa nét đánh dấu";
  controls.appendChild(clearBtn);

  const submitBtn = document.createElement("button");
  submitBtn.className = "repaint-btn";
  submitBtn.textContent = "Xử lý vùng đánh dấu";
  controls.appendChild(submitBtn);

  const resetManualBtn = document.createElement("button");
  resetManualBtn.className = "reset-manual-btn";
  resetManualBtn.textContent = "Xóa vùng chỉnh sửa";
  controls.appendChild(resetManualBtn);

  const aiQcBtn = document.createElement("button");
  aiQcBtn.type = "button";
  aiQcBtn.className = "ai-qc-btn";
  aiQcBtn.textContent = "Kiểm tra bằng AI";
  aiQcBtn.title = "So sánh ảnh nguồn và ảnh đã xử lý để phát hiện vùng cần kiểm tra lại";
  controls.appendChild(aiQcBtn);

  const brushSizeWrap = document.createElement("label");
  brushSizeWrap.className = "brush-size-control";
  brushSizeWrap.textContent = "Kích thước cọ ";
  const brushSizeValue = document.createElement("output");
  brushSizeValue.className = "brush-size-value";
  brushSizeValue.textContent = "—";
  const brushSize = document.createElement("input");
  brushSize.type = "range";
  brushSize.min = "8";
  brushSize.max = "80";
  brushSize.step = "1";
  brushSize.className = "brush-size-slider";
  brushSize.title = "Điều chỉnh kích thước cọ";
  brushSizeWrap.append(brushSize, brushSizeValue);
  controls.appendChild(brushSizeWrap);
  card.appendChild(controls);

  const wrap = document.createElement("div");
  wrap.className = "review-image-wrap";
  const img = document.createElement("img");
  const canvas = document.createElement("canvas");
  canvas.className = "brush-canvas";
  wrap.append(img, canvas);
  card.appendChild(wrap);

  let initialized = false;
  const restoreSnapshot = () => {
    if (!maskSnapshot || !canvas.width || !canvas.height) return;
    const overlay = new Image();
    overlay.onload = () => {
      if (!canvas.isConnected) return;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
      canvas._reviewDirty = true;
    };
    overlay.src = maskSnapshot;
  };
  const initBrush = () => {
    if (initialized || !canvas.isConnected || !img.naturalWidth) return;
    initialized = true;
    setupBrush(pageIndex, img, canvas, wrap, brushBtn, clearBtn, submitBtn, resetManualBtn, aiQcBtn, brushSize, brushSizeValue);
    restoreSnapshot();
  };
  img.addEventListener("load", initBrush, { once: true });
  img.src = page.clean + "?t=" + Date.now();
  card._mountReview = initBrush;
  return card;
}
window.createReviewCard = createReviewCard;

function setupBrush(pageIndex, img, canvas, wrap, brushBtn, clearBtn, submitBtn, resetManualBtn, aiQcBtn, brushSize, brushSizeValue) {
  if (typeof canvas._cleanupBrush === "function") {
    canvas._cleanupBrush();
  } else if (canvas._brushAbort) {
    canvas._brushAbort.abort();
  }

  const abortCtrl = new AbortController();
  const { signal } = abortCtrl;
  canvas._brushAbort = abortCtrl;
  if (typeof canvas._reviewDirty !== "boolean") canvas._reviewDirty = false;

  const syncCanvasSize = () => {
    const nw = img.naturalWidth || img.width;
    const nh = img.naturalHeight || img.height;
    if (nw && nh) {
      canvas.width = nw;
      canvas.height = nh;
    }
    canvas.style.width = "100%";
    canvas.style.height = "100%";
  };
  syncCanvasSize();

  const ctx = canvas.getContext("2d");

  let srcData = null;
  const refreshSrcData = () => {
    const nw = canvas.width;
    const nh = canvas.height;
    if (!nw || !nh) return;
    try {
      const srcCanvas = document.createElement("canvas");
      srcCanvas.width = nw;
      srcCanvas.height = nh;
      const srcCtx = srcCanvas.getContext("2d");
      srcCtx.drawImage(img, 0, 0, nw, nh);
      srcData = srcCtx.getImageData(0, 0, nw, nh);
    } catch (e) {
      console.warn("Failed to refresh srcData for flood fill:", e);
    }
  };
  refreshSrcData();

  img.addEventListener("load", () => {
    syncCanvasSize();
    refreshSrcData();
  }, { signal });

  let brushRadius = Math.min(30, Math.max(10, Math.round(canvas.width * 0.018)));
  brushRadius = Math.max(8, Math.min(80, brushRadius));
  brushSize.value = String(brushRadius);
  brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;

  brushSize.addEventListener("input", () => {
    brushRadius = Number(brushSize.value);
    brushSizeValue.textContent = `${Math.round(brushRadius * 2)}px`;
  }, { signal });

  let brushOn = false;
  let painting = false;
  let lastClickTime = 0;
  let lastClickPos = { x: 0, y: 0 };

  const stopPainting = (e) => {
    if (!painting) return;
    painting = false;
    ctx.closePath();
    if (e && e.pointerId !== undefined && canvas.releasePointerCapture) {
      try {
        if (canvas.hasPointerCapture && canvas.hasPointerCapture(e.pointerId)) {
          canvas.releasePointerCapture(e.pointerId);
        }
      } catch (_) {}
    }
  };

  const stopBrush = () => {
    stopPainting();
    brushOn = false;
    wrap.classList.remove("brush-mode");
    brushBtn.textContent = "Đánh dấu vùng lỗi";
  };

  const cleanupBrush = () => {
    stopBrush();
    abortCtrl.abort();
  };

  canvas._stopBrush = stopBrush;
  canvas._cleanupBrush = cleanupBrush;

  brushBtn.addEventListener("click", () => {
    brushOn = !brushOn;
    if (!brushOn) stopPainting();
    wrap.classList.toggle("brush-mode", brushOn);
    brushBtn.textContent = brushOn ? "Đang đánh dấu · Chọn để kết thúc" : "Đánh dấu vùng lỗi";
  }, { signal });

  clearBtn.addEventListener("click", () => {
    stopPainting();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas._reviewDirty = false;
  }, { signal });

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return { x: 0, y: 0 };
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY),
    };
  }

  function paintDot(x, y) {
    ctx.fillStyle = "rgba(220, 38, 38, 0.7)";
    ctx.beginPath();
    ctx.arc(x, y, brushRadius, 0, Math.PI * 2);
    ctx.fill();
    canvas._reviewDirty = true;
  }

  function paintStrokeTo(x, y) {
    ctx.strokeStyle = "rgba(220, 38, 38, 0.7)";
    ctx.lineWidth = brushRadius * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineTo(x, y);
    ctx.stroke();
    canvas._reviewDirty = true;
  }

  const handleStart = (e) => {
    if (!brushOn) return;
    if (e.button !== undefined && e.button !== 0) return;

    const { x, y } = getCanvasCoords(e);
    const now = Date.now();
    const isDbl = (now - lastClickTime < 350) && (Math.hypot(x - lastClickPos.x, y - lastClickPos.y) < 20);
    lastClickTime = isDbl ? 0 : now;
    lastClickPos = { x, y };

    if (isDbl) {
      stopPainting(e);
      if (!srcData) refreshSrcData();
      if (srcData) {
        if (floodFillSelect(ctx, srcData, x, y, canvas.width, canvas.height)) canvas._reviewDirty = true;
      } else {
        showToast("Không thể chọn vùng tự động do thiếu dữ liệu ảnh nguồn.", "error");
      }
      return;
    }

    painting = true;
    if (e.pointerId !== undefined && canvas.setPointerCapture) {
      try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    }
    paintDot(x, y);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handleMove = (e) => {
    if (!brushOn || !painting) return;
    if (e.buttons !== undefined && e.buttons !== 1 && e.buttons !== 0) {
      stopPainting(e);
      return;
    }
    const { x, y } = getCanvasCoords(e);
    paintStrokeTo(x, y);
  };

  const handleEnd = (e) => {
    stopPainting(e);
  };

  if (window.PointerEvent) {
    canvas.addEventListener("pointerdown", handleStart, { signal });
    canvas.addEventListener("pointermove", handleMove, { signal });
    canvas.addEventListener("pointerup", handleEnd, { signal });
    canvas.addEventListener("pointercancel", handleEnd, { signal });
  } else {
    canvas.addEventListener("mousedown", handleStart, { signal });
    canvas.addEventListener("mousemove", handleMove, { signal });
    canvas.addEventListener("mouseup", handleEnd, { signal });
    canvas.addEventListener("mouseleave", handleEnd, { signal });
  }

  canvas.addEventListener("wheel", (e) => {
    if (!brushOn) return;
    e.preventDefault();
    brushRadius = Math.round(Math.max(8, Math.min(80, brushRadius + (e.deltaY < 0 ? 2 : -2))));
    brushSize.value = String(brushRadius);
    brushSizeValue.textContent = `${Math.round(brushRadius * 2)} px`;
    showToast(`Kích thước cọ: ${Math.round(brushRadius * 2)} px`, "info");
  }, { passive: false, signal });

  window.addEventListener("mouseup", handleEnd, { signal });
  window.addEventListener("pointerup", handleEnd, { signal });
  window.addEventListener("blur", () => stopPainting(), { signal });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPainting();
  }, { signal });

  submitBtn.addEventListener("click", () => {
    submitRepaint(pageIndex, canvas, img, ctx, submitBtn);
  }, { signal });

  if (aiQcBtn) {
    aiQcBtn.addEventListener("click", () => {
      inspectVisualQC(
        pageIndex, canvas, ctx, aiQcBtn, submitBtn, resetManualBtn,
        brushBtn, clearBtn, brushSize, wrap
      );
    }, { signal });
  }

  if (resetManualBtn) {
    resetManualBtn.addEventListener("click", () => {
      resetManualMask(pageIndex, img, canvas, ctx, resetManualBtn);
    }, { signal });
  }
}

async function setupGeminiQCSettings(statusEl, keyInput, saveBtn, clearBtn) {
  const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => (await r.json().catch(() => ({})));
  const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d.detail || `Máy chủ trả về ${s}`;

  const refresh = async () => {
    try {
      const resp = await fetch("/api/visual_qc/settings");
      const data = await parse(resp);
      if (!resp.ok) throw new Error(getErr(resp.status, data));
      if (data.configured) {
        statusEl.textContent = `Kiểm tra AI: Sẵn sàng · ${data.model || "Flash"}`;
        statusEl.classList.add("configured");
      } else {
        statusEl.textContent = data.source === "unavailable"
          ? "Kiểm tra AI: Kho bí mật chưa sẵn sàng"
          : "Kiểm tra AI: Chưa cấu hình";
        statusEl.classList.remove("configured");
      }
      clearBtn.disabled = !data.configured || data.source === "environment";
    } catch (err) {
      statusEl.textContent = "Kiểm tra AI: Lỗi cấu hình";
      statusEl.classList.remove("configured");
      console.warn("Gemini QC settings check failed:", err);
    }
  };

  saveBtn.addEventListener("click", async () => {
    const apiKey = keyInput.value.trim();
    if (!apiKey) {
      showToast("Nhập Gemini API key trước khi lưu.", "error");
      return;
    }
    saveBtn.disabled = true;
    saveBtn.textContent = "Đang lưu…";
    try {
      const resp = await fetch("/api/visual_qc/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey }),
      });
      const data = await parse(resp);
      if (!resp.ok) throw new Error(getErr(resp.status, data));
      keyInput.value = "";
      showToast("Đã lưu khóa API trong kho bí mật của hệ điều hành.", "success");
      await refresh();
    } catch (err) {
      showToast("Không thể lưu khóa API: " + err.message, "error");
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = "Lưu khóa API";
    }
  });

  clearBtn.addEventListener("click", async () => {
    clearBtn.disabled = true;
    try {
      const resp = await fetch("/api/visual_qc/key", { method: "DELETE" });
      const data = await parse(resp);
      if (!resp.ok) throw new Error(getErr(resp.status, data));
      showToast("Đã xóa khóa API khỏi kho bí mật của hệ điều hành.", "success");
    } catch (err) {
      showToast("Không thể xóa khóa API: " + err.message, "error");
    } finally {
      await refresh();
    }
  });

  await refresh();
}
window.setupGeminiQCSettings = setupGeminiQCSettings;

async function inspectVisualQC(
  pageIndex, canvas, ctx, aiQcBtn, submitBtn, resetManualBtn,
  brushBtn, clearBtn, brushSize, wrap
) {
  const mutableControls = [brushBtn, clearBtn, submitBtn, resetManualBtn, aiQcBtn, brushSize].filter(Boolean);
  const previousDisabled = new Map(mutableControls.map((control) => [control, control.disabled]));
  if (typeof canvas._stopBrush === "function") canvas._stopBrush();
  mutableControls.forEach((control) => { control.disabled = true; });
  if (wrap) wrap.classList.add("ai-qc-running");

  const oldText = aiQcBtn.textContent;
  aiQcBtn.textContent = "AI đang kiểm tra…";
  try {
    const resp = await fetch("/api/visual_qc/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter_id: currentChapterId, page_index: pageIndex }),
    });
    const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => (await r.json().catch(() => ({})));
    const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d.detail || `Máy chủ trả về ${s}`;
    const data = await parse(resp);
    if (!resp.ok) throw new Error(getErr(resp.status, data));

    const issues = Array.isArray(data.issues) ? data.issues : [];
    if (!issues.length) {
      showToast("Không phát hiện vùng cần xử lý lại trên trang này.", "success");
      return;
    }

    const PAINT_CONFIDENCE_THRESHOLD = 0.75;
    let painted = 0;
    let uncertain = 0;
    let artDamage = 0;
    ctx.save();
    ctx.fillStyle = "rgba(220, 38, 38, 0.52)";
    ctx.strokeStyle = "rgba(248, 113, 113, 0.95)";
    ctx.lineWidth = Math.max(2, Math.round(canvas.width * 0.0015));
    for (const issue of issues) {
      if (issue.issue_type === "over_erased_art") {
        artDamage++;
        continue;
      }
      if ((Number(issue.confidence) || 0) < PAINT_CONFIDENCE_THRESHOLD) {
        uncertain++;
        continue;
      }
      const polygon = Array.isArray(issue.polygon) ? issue.polygon : [];
      if (polygon.length < 3) continue;
      ctx.beginPath();
      ctx.moveTo(Number(polygon[0][0]) || 0, Number(polygon[0][1]) || 0);
      for (let i = 1; i < polygon.length; i++) {
        ctx.lineTo(Number(polygon[i][0]) || 0, Number(polygon[i][1]) || 0);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      painted++;
    }
    ctx.restore();

    if (painted) {
      canvas._reviewDirty = true;
      const extra = [];
      if (uncertain) extra.push(`${uncertain} vùng có độ tin cậy thấp chưa được đánh dấu`);
      if (artDamage) extra.push(`${artDamage} vùng nghi mất chi tiết cần kiểm tra thủ công`);
      const suffix = extra.length ? ` (${extra.join(", ")})` : "";
      showToast(`AI đã đánh dấu ${painted} vùng cần kiểm tra${suffix}. Xác nhận vùng đánh dấu trước khi xử lý.`, "info");
    } else if (artDamage || uncertain) {
      showToast(`AI phát hiện ${uncertain} vùng có độ tin cậy thấp và ${artDamage} vùng nghi mất chi tiết; chưa tự động đánh dấu để tránh làm hỏng ảnh.`, "info");
    } else {
      showToast("AI trả về kết quả nhưng không có vùng đánh dấu hợp lệ.", "error");
    }
  } catch (err) {
    showToast("Không thể hoàn tất kiểm tra bằng AI: " + err.message, "error");
  } finally {
    mutableControls.forEach((control) => { control.disabled = previousDisabled.get(control) || false; });
    if (wrap) wrap.classList.remove("ai-qc-running");
    aiQcBtn.textContent = oldText;
  }
}

function floodFillSelect(ctx, srcData, startX, startY, width, height) {
  const data = srcData.data;
  const startIndex = (startY * width + startX) * 4;
  const sr = data[startIndex];
  const sg = data[startIndex + 1];
  const sb = data[startIndex + 2];

  const COLOR_TOLERANCE = 22;
  const safetyCap = Math.min(Math.round(width * height * 0.15), 260000);

  function colorMatch(idx) {
    const dr = Math.abs(data[idx] - sr);
    const dg = Math.abs(data[idx + 1] - sg);
    const db = Math.abs(data[idx + 2] - sb);
    return dr <= COLOR_TOLERANCE && dg <= COLOR_TOLERANCE && db <= COLOR_TOLERANCE;
  }

  const visited = new Uint8Array(width * height);
  const stack = [[startX, startY]];
  visited[startY * width + startX] = 1;
  let visitedCount = 1;
  let capExceeded = false;

  while (stack.length > 0) {
    const [x, y] = stack.pop();
    const pxIdx = (y * width + x) * 4;

    if (!colorMatch(pxIdx)) continue;

    let xl = x;
    while (xl > 0 && !visited[y * width + xl - 1] && colorMatch((y * width + xl - 1) * 4)) {
      visited[y * width + xl - 1] = 1;
      visitedCount++;
      xl--;
    }

    let xr = x;
    while (xr < width - 1 && !visited[y * width + xr + 1] && colorMatch((y * width + xr + 1) * 4)) {
      visited[y * width + xr + 1] = 1;
      visitedCount++;
      xr++;
    }

    if (visitedCount > safetyCap) {
      capExceeded = true;
      break;
    }

    for (let nx = xl; nx <= xr; nx++) {
      if (y > 0) {
        const up = (y - 1) * width + nx;
        if (!visited[up] && colorMatch(up * 4)) {
          visited[up] = 1;
          visitedCount++;
          stack.push([nx, y - 1]);
        }
      }
      if (y < height - 1) {
        const dn = (y + 1) * width + nx;
        if (!visited[dn] && colorMatch(dn * 4)) {
          visited[dn] = 1;
          visitedCount++;
          stack.push([nx, y + 1]);
        }
      }
    }

    if (visitedCount > safetyCap) {
      capExceeded = true;
      break;
    }
  }

  if (capExceeded) {
    showToast("Không thể chọn tự động vì vùng màu lan quá rộng. Hãy đánh dấu thủ công bằng cách kéo cọ.", "error");
    return false;
  }

  const imgData = ctx.createImageData(width, height);
  const out = imgData.data;
  for (let idx = 0; idx < visited.length; idx++) {
    if (visited[idx]) {
      const o = idx * 4;
      out[o] = 220;
      out[o + 1] = 38;
      out[o + 2] = 38;
      out[o + 3] = 178;
    }
  }

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = width;
  tmpCanvas.height = height;
  tmpCanvas.getContext("2d").putImageData(imgData, 0, 0);
  ctx.drawImage(tmpCanvas, 0, 0);
  return true;
}

async function submitRepaint(pageIndex, canvas, img, ctx, submitBtn) {
  const pixelCheckCtx = document.createElement("canvas").getContext("2d");
  pixelCheckCtx.canvas.width = canvas.width;
  pixelCheckCtx.canvas.height = canvas.height;
  pixelCheckCtx.drawImage(canvas, 0, 0);
  const imgData = pixelCheckCtx.getImageData(0, 0, canvas.width, canvas.height);
  let hasPaint = false;
  for (let i = 3; i < imgData.data.length; i += 4) {
    if (imgData.data[i] > 20) {
      hasPaint = true;
      break;
    }
  }

  if (!hasPaint) {
    showToast("Chưa có vùng nào được đánh dấu để xử lý.", "error");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Đang xử lý…";

  try {
    const maskBlob = await new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Không thể tạo dữ liệu vùng đánh dấu"));
      }, "image/png");
    });

    const formData = new FormData();
    formData.append("chapter_id", currentChapterId);
    formData.append("page_index", pageIndex);
    formData.append("mask", maskBlob, "mask.png");

    const resp = await fetch("/api/repaint_mask", { method: "POST", body: formData });
    const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => (await r.json().catch(() => ({})));
    const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d.detail || `Server trả về ${s}`;
    const manifest = await parse(resp);
    if (!resp.ok) {
      throw new Error(getErr(resp.status, manifest));
    }
    currentManifest.pages[pageIndex] = manifest.pages[pageIndex];

    img.src = manifest.pages[pageIndex].clean + "?t=" + Date.now();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas._reviewDirty = false;
    showToast("Đã xử lý vùng đánh dấu.", "success");
  } catch (err) {
    showToast("Không thể xử lý vùng đánh dấu: " + err.message, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Xử lý vùng đánh dấu";
  }
}
