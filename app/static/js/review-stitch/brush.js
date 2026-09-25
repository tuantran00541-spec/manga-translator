import { BRUSH_CHUNK_H, deleteSnapshot, readSnapshot, snapshotKey, state, storeSnapshot } from "./state.js";

export function makeBrushChunks(host, width, height) {
  const chunks = [];
  for (let y = 0; y < height; y += BRUSH_CHUNK_H) {
    chunks.push({
      host,
      width,
      y1: y,
      y2: Math.min(height, y + BRUSH_CHUNK_H),
      canvas: null,
      ctx: null,
      dirty: false,
    });
  }
  return chunks;
}

export function ensureBrushCanvas(chunk) {
  if (chunk.canvas && chunk.ctx) return chunk;
  const canvas = document.createElement("canvas");
  canvas.className = "stitched-brush-chunk";
  canvas.width = chunk.width;
  canvas.height = chunk.y2 - chunk.y1;
  canvas.dataset.sourceY = String(chunk.y1);
  canvas.style.top = `${chunk.y1}px`;
  chunk.host.appendChild(canvas);
  chunk.canvas = canvas;
  chunk.ctx = canvas.getContext("2d", { willReadFrequently: true });
  return chunk;
}

export function chunkHasPaint(chunk, y1 = chunk.y1, y2 = chunk.y2) {
  if (!chunk?.dirty || !chunk.canvas) return false;
  const iy1 = Math.max(chunk.y1, y1), iy2 = Math.min(chunk.y2, y2);
  if (iy2 <= iy1) return false;
  const h = iy2 - iy1, width = chunk.canvas.width, scale = Math.min(1, 512 / Math.max(width, h));
  const probe = document.createElement("canvas");
  probe.width = Math.max(1, Math.ceil(width * scale));
  probe.height = Math.max(1, Math.ceil(h * scale));
  const ctx = probe.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(chunk.canvas, 0, iy1 - chunk.y1, width, h, 0, 0, probe.width, probe.height);
  const data = ctx.getImageData(0, 0, probe.width, probe.height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 20) return true;
  return false;
}

export function captureSnapshot(shell) {
  if (!shell || state.variant !== "clean") return;
  const dirty = (shell._brushChunks || []).filter((chunk) => chunk.dirty && chunkHasPaint(chunk));
  const key = snapshotKey();
  if (!dirty.length) {
    deleteSnapshot(key);
    return;
  }
  storeSnapshot(key, dirty.map((chunk) => ({ y1: chunk.y1, dataUrl: chunk.canvas.toDataURL("image/png") })));
}

export function restoreSnapshot(shell) {
  const saved = readSnapshot(snapshotKey());
  if (!Array.isArray(saved)) return;
  const byY = new Map((shell._brushChunks || []).map((chunk) => [chunk.y1, chunk]));
  for (const item of saved) {
    const chunk = byY.get(Number(item.y1));
    if (!chunk) continue;
    ensureBrushCanvas(chunk);
    const img = new Image();
    img.onload = () => {
      if (!chunk.canvas?.isConnected) return;
      chunk.ctx.drawImage(img, 0, 0);
      chunk.dirty = true;
    };
    img.src = item.dataUrl;
  }
}

export function paintPoint(shell, x, y, radius, erase) {
  for (const chunk of shell._brushChunks || []) {
    if (y + radius < chunk.y1 || y - radius >= chunk.y2) continue;
    ensureBrushCanvas(chunk);
    const ctx = chunk.ctx; ctx.save(); ctx.globalCompositeOperation = erase ? "destination-out" : "source-over"; ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--text-primary").trim() || "#000000"; ctx.beginPath(); ctx.arc(x, y - chunk.y1, radius, 0, Math.PI * 2); ctx.fill(); ctx.restore(); chunk.dirty = true;
  }
}

export function paintStroke(shell, a, b, radius, erase) {
  const dx = b.x - a.x, dy = b.y - a.y, count = Math.max(1, Math.ceil(Math.hypot(dx, dy) / Math.max(1, radius * .45)));
  for (let i = 0; i <= count; i++) { const t = i / count; paintPoint(shell, a.x + dx * t, a.y + dy * t, radius, erase); }
}

export function drawBrushMask(ctx, chunks, desc, width) {
  for (const chunk of chunks || []) {
    if (!chunk.dirty || !chunk.canvas) continue;
    const y1 = Math.max(chunk.y1, desc.sourceY1), y2 = Math.min(chunk.y2, desc.sourceY2);
    if (y2 <= y1) continue;
    const h = y2 - y1;
    ctx.drawImage(
      chunk.canvas,
      0,
      y1 - chunk.y1,
      width,
      h,
      0,
      desc.localY1 + y1 - desc.sourceY1,
      width,
      h,
    );
  }
}

export function canvasBlob(canvas) {
  if (typeof window.canvasToBlob === "function") return window.canvasToBlob(canvas);
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Không thể mã hóa mask")), "image/png"));
}
