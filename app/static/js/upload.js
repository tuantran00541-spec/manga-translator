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

  dropzone.addEventListener("click", () => input.click());

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
      `Đã bỏ qua ${rejected.length} tệp không được hỗ trợ. Chỉ chấp nhận PNG, JPG, WEBP, BMP, ZIP và CBZ.`,
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
  if (hint) hint.textContent = `Đang tải ${files.length} tệp…`;

  try {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    const workers = typeof getWorkersSetting === "function" ? getWorkersSetting() : 2;
    formData.append("workers", String(workers));

    const resp = await fetch("/api/chapter/upload", {
      method: "POST",
      body: formData,
    });
    const parse = typeof window.parseApiResponse === "function" ? window.parseApiResponse : async (r) => (await r.json().catch(() => ({})));
    const getErr = typeof window.getErrorMessage === "function" ? window.getErrorMessage : (s, d) => d.detail || `Server trả về lỗi ${s}`;
    const data = await parse(resp);
    if (!resp.ok) {
      throw new Error(getErr(resp.status, data));
    }
    currentManifest = data;
    currentChapterId = currentManifest.chapter_id;
    try {
      sessionStorage.setItem("mt_active_chapter", currentChapterId);
      window.history.replaceState(null, "", `#${currentChapterId}`);
    } catch (_) {}
    window.previewActivePageIndex = 0;
    renderPreview();
  } catch (err) {
    showToast("Không thể tải tệp lên: " + err.message, "error");
  } finally {
    if (dropzone) dropzone.classList.remove("uploading");
    if (hint) hint.textContent = originalHint;
  }
}
