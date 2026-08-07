// review.js - Quản lý Giao diện Kiểm tra Tẩy chữ & Cọ tô lỗi

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
    label.className = "preview-label";
    label.textContent = pageLabel(currentManifest.pages, pageIndex);
    card.appendChild(label);

    const imgWrap = document.createElement("div");
    imgWrap.className = "review-image-wrap";

    const img = document.createElement("img");
    img.src = page.clean + "?t=" + Date.now();

    const canvas = document.createElement("canvas");
    canvas.className = "brush-canvas";

    imgWrap.appendChild(img);
    imgWrap.appendChild(canvas);
    card.appendChild(imgWrap);

    const controls = document.createElement("div");
    controls.className = "review-controls";

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "brush-toggle-btn";
    toggleBtn.textContent = "Tô lỗi (vẽ đỏ)";

    const clearBtn = document.createElement("button");
    clearBtn.className = "clear-brush-btn";
    clearBtn.textContent = "Xóa nét vẽ";

    const repaintBtn = document.createElement("button");
    repaintBtn.className = "repaint-btn";
    repaintBtn.textContent = "Tẩy lại vùng chọn";
    repaintBtn.disabled = true;

    controls.appendChild(toggleBtn);
    controls.appendChild(clearBtn);
    controls.appendChild(repaintBtn);
    card.appendChild(controls);

    container.appendChild(card);

    setupBrushCanvas(img, canvas, toggleBtn, clearBtn, repaintBtn, pageIndex);
  });
}

function setupBrushCanvas(img, canvas, toggleBtn, clearBtn, repaintBtn, pageIndex) {
  const ctx = canvas.getContext("2d");
  let drawing = false;
  let hasStrokes = false;
  const BRUSH_RADIUS = 16;

  function initCanvasSize() {
    if (img.clientWidth > 0 && img.clientHeight > 0) {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
    }
  }

  if (img.complete && img.clientWidth > 0) {
    initCanvasSize();
  } else {
    img.onload = initCanvasSize;
  }

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  function drawPoint(x, y) {
    ctx.fillStyle = "rgba(232, 67, 44, 0.75)";
    ctx.beginPath();
    ctx.arc(x, y, BRUSH_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    hasStrokes = true;
    repaintBtn.disabled = false;
  }

  function drawLine(p1, p2) {
    ctx.strokeStyle = "rgba(232, 67, 44, 0.75)";
    ctx.lineWidth = BRUSH_RADIUS * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    hasStrokes = true;
    repaintBtn.disabled = false;
  }

  let lastPos = null;

  canvas.addEventListener("mousedown", (e) => {
    const imgWrap = canvas.parentElement;
    if (!imgWrap.classList.contains("brush-mode")) return;
    drawing = true;
    lastPos = getPos(e);
    drawPoint(lastPos.x, lastPos.y);
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const pos = getPos(e);
    drawLine(lastPos, pos);
    lastPos = pos;
  });

  const stopDrawing = () => { drawing = false; lastPos = null; };
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseleave", stopDrawing);

  // Auto flood fill
  canvas.addEventListener("dblclick", (e) => {
    const imgWrap = canvas.parentElement;
    if (!imgWrap.classList.contains("brush-mode")) return;
    e.preventDefault();

    const pos = getPos(e);
    const scaleX = img.naturalWidth / canvas.width;
    const scaleY = img.naturalHeight / canvas.height;
    const origX = Math.round(pos.x * scaleX);
    const origY = Math.round(pos.y * scaleY);

    floodFillColor(img, origX, origY, (filledMask) => {
      if (!filledMask) return;
      drawMaskToCanvas(ctx, canvas, filledMask, img.naturalWidth, img.naturalHeight);
      hasStrokes = true;
      repaintBtn.disabled = false;
    });
  });

  toggleBtn.addEventListener("click", () => {
    const imgWrap = canvas.parentElement;
    const active = imgWrap.classList.toggle("brush-mode");
    toggleBtn.classList.toggle("active", active);
    toggleBtn.textContent = active ? "Đang tô lỗi..." : "Tô lỗi (vẽ đỏ)";
  });

  clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hasStrokes = false;
    repaintBtn.disabled = true;
  });

  repaintBtn.addEventListener("click", async () => {
    if (!hasStrokes) return;
    repaintBtn.disabled = true;
    repaintBtn.textContent = "Đang tẩy...";

    const maskBase64 = getFullSizeMaskBase64(canvas, img.naturalWidth, img.naturalHeight);

    try {
      const res = await submitRepaintApi(pageIndex, maskBase64);
      if (res.ok) {
        showToast("Tẩy lại thành công!", "success");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasStrokes = false;
        img.src = res.clean_url + "?t=" + Date.now();
        currentManifest.pages[pageIndex].clean = res.clean_url;
      } else {
        showToast("Lỗi tẩy lại: " + (res.error || "Không xác định"), "error");
      }
    } catch (err) {
      showToast("Lỗi kết nối: " + err.message, "error");
    } finally {
      repaintBtn.textContent = "Tẩy lại vùng chọn";
      repaintBtn.disabled = !hasStrokes;
    }
  });
}

function floodFillColor(img, seedX, seedY, callback) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = img.naturalWidth;
  tempCanvas.height = img.naturalHeight;
  const tCtx = tempCanvas.getContext("2d");
  tCtx.drawImage(img, 0, 0);

  const imgData = tCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
  const data = imgData.data;
  const w = tempCanvas.width;
  const h = tempCanvas.height;

  if (seedX < 0 || seedX >= w || seedY < 0 || seedY >= h) return callback(null);

  const targetIdx = (seedY * w + seedX) * 4;
  const tr = data[targetIdx], tg = data[targetIdx + 1], tb = data[targetIdx + 2];

  const tolerance = 22;
  function colorMatch(idx) {
    return Math.abs(data[idx] - tr) <= tolerance &&
           Math.abs(data[idx + 1] - tg) <= tolerance &&
           Math.abs(data[idx + 2] - tb) <= tolerance;
  }

  const mask = new Uint8Array(w * h);
  const stack = [[seedX, seedY]];
  let count = 0;
  const maxPixels = w * h * 0.45;

  while (stack.length > 0) {
    const [cx, cy] = stack.pop();
    const idx = cy * w + cx;
    if (mask[idx]) continue;

    const pIdx = idx * 4;
    if (!colorMatch(pIdx)) continue;

    mask[idx] = 1;
    count++;
    if (count > maxPixels) {
      showToast("Vùng tràn quá lớn (vượt 45% trang), hủy tự chọn để tránh hỏng nền.", "error");
      return callback(null);
    }

    if (cx > 0) stack.push([cx - 1, cy]);
    if (cx < w - 1) stack.push([cx + 1, cy]);
    if (cy > 0) stack.push([cx, cy - 1]);
    if (cy < h - 1) stack.push([cx, cy + 1]);
  }

  callback(mask);
}

function drawMaskToCanvas(ctx, canvas, mask, origW, origH) {
  const scaleX = canvas.width / origW;
  const scaleY = canvas.height / origH;
  ctx.fillStyle = "rgba(232, 67, 44, 0.75)";

  for (let y = 0; y < origH; y += 2) {
    for (let x = 0; x < origW; x += 2) {
      if (mask[y * origW + x]) {
        ctx.fillRect(x * scaleX, y * scaleY, 2 * scaleX + 0.5, 2 * scaleY + 0.5);
      }
    }
  }
}

function getFullSizeMaskBase64(canvas, origW, origH) {
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = origW;
  maskCanvas.height = origH;
  const mCtx = maskCanvas.getContext("2d");

  mCtx.fillStyle = "#000000";
  mCtx.fillRect(0, 0, origW, origH);
  mCtx.drawImage(canvas, 0, 0, origW, origH);

  const imgData = mCtx.getImageData(0, 0, origW, origH);
  const data = imgData.data;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    if (r > 150 && g < 100 && b < 100) {
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
    } else {
      data[i] = 0;
      data[i + 1] = 0;
      data[i + 2] = 0;
    }
  }

  mCtx.putImageData(imgData, 0, 0);
  return maskCanvas.toDataURL("image/png");
}
