"use strict";

self.onmessage = async (event) => {
  const { id, bitmap, width, height } = event.data || {};
  if (!id || !bitmap || !width || !height) return;
  try {
    if (typeof OffscreenCanvas === "undefined") {
      throw new Error("OffscreenCanvas unavailable in worker");
    }
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });
    if (!ctx) throw new Error("2D context unavailable");
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close?.();
    const blob = await canvas.convertToBlob({ type: "image/png" });
    self.postMessage({ id, ok: true, blob, width, height, bytes: blob.size });
  } catch (error) {
    try { bitmap.close?.(); } catch (_) {}
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
