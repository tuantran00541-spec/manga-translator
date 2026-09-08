(() => {
  "use strict";

  const WORKER_TIMEOUT_MS = 1800;
  const original = window.frontendReleaseEncodeCanvas;
  if (typeof original !== "function") return;

  let degradedToMainThread = false;

  function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
      if (!(canvas instanceof HTMLCanvasElement) || !canvas.width || !canvas.height) {
        reject(new Error("Canvas mask không hợp lệ"));
        return;
      }
      const native = HTMLCanvasElement.prototype.toBlob;
      if (typeof native !== "function") {
        reject(new Error("canvas.toBlob unavailable"));
        return;
      }
      native.call(
        canvas,
        (blob) => blob ? resolve(blob) : reject(new Error("canvas.toBlob returned null")),
        "image/png",
      );
    });
  }

  async function guardedEncode(canvas) {
    if (degradedToMainThread) return canvasToBlob(canvas);

    let timeoutId = null;
    try {
      const timeout = new Promise((_, reject) => {
        timeoutId = setTimeout(
          () => reject(new Error(`Mask worker timed out after ${WORKER_TIMEOUT_MS}ms`)),
          WORKER_TIMEOUT_MS,
        );
      });
      return await Promise.race([Promise.resolve().then(() => original(canvas)), timeout]);
    } catch (error) {
      degradedToMainThread = true;
      console.warn("Mask worker unavailable; falling back to async canvas.toBlob for this page.", error);
      return canvasToBlob(canvas);
    } finally {
      if (timeoutId !== null) clearTimeout(timeoutId);
    }
  }

  window.frontendReleaseEncodeCanvas = guardedEncode;
  window.canvasToBlob = guardedEncode;
  window.frontendReleaseEncoderGuard = {
    timeoutMs: WORKER_TIMEOUT_MS,
    get degraded() { return degradedToMainThread; },
  };
})();
