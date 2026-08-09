function renderReview() {
  const container = document.getElementById("page-view");
  if (!container) return;
  container.innerHTML = "";
  container.className = "";

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";

  const hint = document.createElement("span");
  hint.className = "review-hint";
  hint.innerHTML = 'Bấm "Tô lỗi" rồi <b>double-click vào giữa vùng lỗi</b> (bong bóng/nền) để tự động chọn trọn vùng đồng màu đó — không cần tô tay chính xác. Nếu vùng lỗi không đồng màu (dính nhiều chi tiết), tô tay bằng cách kéo chuột, nhưng nhớ <b>phủ kín toàn bộ</b> phần lỗi trong 1 lần, tô sót thì phần còn lại vẫn hiện nguyên.';
  toolbar.appendChild(hint);

  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Ổn rồi, vào dịch";
  nextBtn.addEventListener("click", renderEditor);
  toolbar.appendChild(nextBtn);

  container.appendChild(toolbar);

  currentManifest.pages.forEach((page, pageIndex) => {
    if (page.skipped) return;

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
    brushBtn.textContent = "Tô lỗi";
    controls.appendChild(brushBtn);

    const clearBtn = document.createElement("button");
    clearBtn.className = "clear-brush-btn";
    clearBtn.textContent = "Xóa nét vẽ";
    controls.appendChild(clearBtn);

    const submitBtn = document.createElement("button");
    submitBtn.className = "repaint-btn";
    submitBtn.textContent = "Xử lý lại vùng đã tô";
    controls.appendChild(submitBtn);

    const resetManualBtn = document.createElement("button");
    resetManualBtn.className = "reset-manual-btn";
    resetManualBtn.textContent = "Xóa vùng tô tay";
    controls.appendChild(resetManualBtn);

    const brushSizeWrap = document.createElement("label");
    brushSizeWrap.className = "brush-size-control";
    brushSizeWrap.textContent = "Cỡ cọ ";
    const brushSizeValue = document.createElement("output");
    brushSizeValue.className = "brush-size-value";
    brushSizeValue.textContent = "—";
    const brushSize = document.createElement("input");
    brushSize.type = "range";
    brushSize.min = "8";
    brushSize.max = "80";
    brushSize.step = "1";
    brushSize.className = "brush-size-slider";
    brushSize.title = "Điều chỉnh bán kính cọ";
    brushSizeWrap.appendChild(brushSize);
    brushSizeWrap.appendChild(brushSizeValue);
    controls.appendChild(brushSizeWrap);

    card.appendChild(controls);

    const wrap = document.createElement("div");
    wrap.className = "review-image-wrap";

    const img = document.createElement("img");
    img.src = page.clean;
    wrap.appendChild(img);

    const canvas = document.createElement("canvas");
    canvas.className = "brush-canvas";
    wrap.appendChild(canvas);

    card.appendChild(wrap);
    container.appendChild(card);

    const initBrush = () => setupBrush(pageIndex, img, canvas, wrap, brushBtn, clearBtn, submitBtn, resetManualBtn, brushSize, brushSizeValue);
    if (img.complete && img.naturalWidth > 0) {
      initBrush();
    } else {
      img.onload = initBrush;
    }
  });
}

function setupBrush(pageIndex, img, canvas, wrap, brushBtn, clearBtn, submitBtn, resetManualBtn, brushSize, brushSizeValue) {
  if (canvas._brushAbort) canvas._brushAbort.abort();
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";
  const ctx = canvas.getContext("2d");
  let brushRadius = Math.min(30, Math.max(10, Math.round(img.naturalWidth * 0.018)));
  brushRadius = Math.max(8, Math.min(80, brushRadius));
  brushSize.value = String(brushRadius);
  brushSizeValue.textContent = `${brushRadius}px`;

  brushSize.addEventListener("input", () => {
    brushRadius = Number(brushSize.value);
    brushSizeValue.textContent = `${brushRadius}px`;
  });

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = img.naturalWidth;
  srcCanvas.height = img.naturalHeight;
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.drawImage(img, 0, 0);
  const srcData = srcCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);

  let brushOn = false;
  let painting = false;
  let lastDblClick = 0;

  brushBtn.addEventListener("click", () => {
    brushOn = !brushOn;
    wrap.classList.toggle("brush-mode", brushOn);
    brushBtn.textContent = brushOn ? "Đang tô (bấm để tắt)" : "Tô lỗi";
  });

  clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  });

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
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
  }

  function paintStrokeTo(x, y) {
    ctx.strokeStyle = "rgba(220, 38, 38, 0.7)";
    ctx.lineWidth = brushRadius * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  canvas.addEventListener("mousedown", (e) => {
    if (!brushOn) return;
    const now = Date.now();
    const isDbl = now - lastDblClick < 350;
    lastDblClick = now;

    const { x, y } = getCanvasCoords(e);

    if (isDbl) {
      painting = false;
      floodFillSelect(ctx, srcData, x, y, canvas.width, canvas.height);
      return;
    }

    painting = true;
    paintDot(x, y);
    ctx.beginPath();
    ctx.moveTo(x, y);
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!brushOn || !painting) return;
    const { x, y } = getCanvasCoords(e);
    paintStrokeTo(x, y);
  });

  canvas.addEventListener("wheel", (e) => {
    if (!brushOn) return;
    e.preventDefault();
    brushRadius = Math.round(Math.max(8, Math.min(40, brushRadius + (e.deltaY < 0 ? 2 : -2))));
    showToast(`Cỡ cọ: ${Math.round(brushRadius * 2)} px`, "info");
  }, { passive: false });

  const abortCtrl = new AbortController();
  window.addEventListener("mouseup", () => {
    painting = false;
    ctx.closePath();
  }, { signal: abortCtrl.signal });

  canvas._brushAbort = abortCtrl;

  submitBtn.addEventListener("click", () => submitRepaint(pageIndex, canvas, img, ctx, submitBtn));
  if (resetManualBtn) {
    resetManualBtn.addEventListener("click", () => resetManualMask(pageIndex, img, canvas, ctx, resetManualBtn));
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
    showToast("Vùng này không đủ đồng màu để tự động chọn (lan quá rộng). Hãy tô tay bằng cách kéo chuột thay vì double-click.", "error");
    return;
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
    showToast("Chưa tô vùng lỗi nào trên ảnh.", "error");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Đang xử lý lại...";

  try {
    const maskBlob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    const formData = new FormData();
    formData.append("chapter_id", currentChapterId);
    formData.append("page_index", pageIndex);
    formData.append("mask", maskBlob, "mask.png");

    const resp = await fetch("/api/repaint_mask", { method: "POST", body: formData });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`Server trả về ${resp.status}: ${txt}`);
    }
    const manifest = await resp.json();
    currentManifest.pages[pageIndex] = manifest.pages[pageIndex];

    img.src = manifest.pages[pageIndex].clean + "?t=" + Date.now();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  } catch (err) {
    showToast("Xử lý lại thất bại: " + err.message, "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Xử lý lại vùng đã tô";
  }
}
