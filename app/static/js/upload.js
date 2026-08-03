const UPLOAD_ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".zip", ".cbz"];

function initUpload() {
  const dropzone = document.getElementById("upload-dropzone");
  const input = document.getElementById("upload-input");
  if (!dropzone || !input) return;

  input.addEventListener("change", () => {
    if (input.files.length > 0) {
      handleUploadFiles(Array.from(input.files));
      input.value = "";
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) handleUploadFiles(files);
  });
}

function handleUploadFiles(files) {
  function isAllowed(f) {
    const name = f.name.toLowerCase();
    return UPLOAD_ALLOWED_EXTS.some((ext) => name.endsWith(ext));
  }

  const rejected = files.filter((f) => !isAllowed(f));
  const accepted = files.filter((f) => isAllowed(f));

  if (rejected.length > 0) {
    showToast(
      `Bỏ qua ${rejected.length} file không đúng định dạng (chỉ nhận PNG, JPG, WEBP, BMP, ZIP, CBZ).`,
      "error"
    );
  }

  if (accepted.length === 0) return;

  accepted.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  uploadChapter(accepted);
}

async function uploadChapter(files) {
  const dropzone = document.getElementById("upload-dropzone");
  const hint = document.getElementById("upload-hint");
  const originalHint = hint ? hint.textContent : "";

  if (dropzone) dropzone.classList.add("uploading");
  if (hint) hint.textContent = `Đang tải lên ${files.length} file...`;

  try {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    const workers = typeof getWorkersSetting === "function" ? getWorkersSetting() : 2;
    formData.append("workers", String(workers));

    const resp = await fetch("/api/chapter/upload", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `Server trả về lỗi ${resp.status}`);
    }
    currentManifest = await resp.json();
    currentChapterId = currentManifest.chapter_id;
    renderPreview();
  } catch (err) {
    showToast("Tải file lên thất bại: " + err.message, "error");
  } finally {
    if (dropzone) dropzone.classList.remove("uploading");
    if (hint) hint.textContent = originalHint;
  }
}
