(() => {
  "use strict";

  const POLL_INTERVAL_MS = 650;
  const STAGE_LABELS = {
    preparing: "Đang chuẩn bị luồng tải…",
    downloading: "Đang tải xuống dữ liệu…",
    merging: "Đang ghép hình ảnh & âm thanh…",
    canceling: "Đang hủy tác vụ…",
    completed: "Hoàn tất! Tệp đã sẵn sàng lưu về máy.",
    canceled: "Đã hủy tải xuống.",
    failed: "Tải xuống thất bại.",
  };

  let inspected = null;
  let selectedPreset = null;
  let activeJobId = null;
  let pollTimer = null;
  let inspectSeq = 0;
  let triggeredDownloadJobId = null;
  let initialized = false;

  const $ = (id) => document.getElementById(id);

  function formatBytes(bytes) {
    const val = Number(bytes);
    if (!Number.isFinite(val) || val <= 0) return "";
    const units = ["B", "KB", "MB", "GB"];
    let size = val;
    let idx = 0;
    while (size >= 1024 && idx < units.length - 1) {
      size /= 1024;
      idx += 1;
    }
    return idx === 0 ? `${Math.round(size)} B` : `${size.toFixed(1)} ${units[idx]}`;
  }

  function formatEta(seconds) {
    const val = Number(seconds);
    if (!Number.isFinite(val) || val <= 0) return "";
    const mins = Math.floor(val / 60);
    const secs = Math.floor(val % 60);
    return mins > 0 ? `${mins}p ${String(secs).padStart(2, "0")}s` : `${secs}s`;
  }

  async function apiJson(url, options = undefined) {
    const resp = await fetch(url, options);
    const data = typeof window.parseApiResponse === "function"
      ? await window.parseApiResponse(resp)
      : await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const msg = typeof window.getErrorMessage === "function"
        ? window.getErrorMessage(resp.status, data)
        : (data?.detail || `HTTP ${resp.status}`);
      throw new Error(msg);
    }
    return data;
  }

  function clearPoll() {
    if (pollTimer !== null) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function triggerFileDownload(url, filename) {
    if (!url || typeof url !== "string" || !url.startsWith("/api/media/files/")) return;
    const link = document.createElement("a");
    link.href = url;
    if (filename) link.download = filename;
    link.rel = "noopener";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function renderSafety(safety) {
    const card = $("media-safety-card");
    const badge = $("media-safety-badge");
    const summary = $("media-safety-summary");
    const host = $("media-safety-host");
    const list = $("media-safety-checks");
    if (!card || !badge || !summary || !list) return;

    if (!safety) {
      card.hidden = true;
      return;
    }

    const isSafe = Boolean(safety.safe);
    card.hidden = false;
    card.dataset.safetyState = isSafe ? "safe" : "unsafe";
    badge.textContent = isSafe ? "Liên kết an toàn" : "Cảnh báo — Đã chặn";
    badge.className = `media-safety-badge ${isSafe ? "is-safe" : "is-unsafe"}`;
    summary.textContent = String(safety.summary || "");
    if (host) {
      host.textContent = safety.hostname ? String(safety.hostname) : "";
      host.hidden = !safety.hostname;
    }

    list.replaceChildren();
    const checks = Array.isArray(safety.checks) ? safety.checks : [];
    checks.forEach((chk) => {
      const li = document.createElement("li");
      li.className = `media-safety-item ${chk?.passed ? "is-pass" : "is-fail"}`;

      const icon = document.createElement("span");
      icon.className = "media-safety-mark";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = chk?.passed ? "✓" : "✕";

      const copy = document.createElement("div");
      copy.className = "media-safety-copy";

      const label = document.createElement("strong");
      label.textContent = String(chk?.label || "");
      const detail = document.createElement("span");
      detail.textContent = String(chk?.detail || "");

      copy.append(label, detail);
      li.append(icon, copy);
      list.appendChild(li);
    });
  }

  function updateSelectedOptionUi() {
    const grid = $("media-options-grid");
    const downloadBtn = $("media-download-btn");
    if (grid) {
      grid.querySelectorAll(".media-option-card").forEach((btn) => {
        const isSelected = btn.dataset.preset === selectedPreset && !btn.disabled;
        btn.classList.toggle("is-selected", isSelected);
        btn.setAttribute("aria-checked", String(isSelected));
      });
    }
    if (downloadBtn) {
      const canDownload = Boolean(
        inspected?.safe
        && inspected?.media
        && selectedPreset
        && !activeJobId,
      );
      downloadBtn.disabled = !canDownload;
    }
  }

  function renderPicker(data) {
    const card = $("media-picker-card");
    const thumb = $("media-thumb");
    const fallback = $("media-thumb-fallback");
    const platformBadge = $("media-platform-badge");
    const durationBadge = $("media-duration-badge");
    const titleEl = $("media-title");
    const uploaderEl = $("media-uploader");
    const ffmpegNote = $("media-ffmpeg-note");
    const grid = $("media-options-grid");
    if (!card || !grid) return;

    if (!data?.safe || !data?.media) {
      card.hidden = true;
      selectedPreset = null;
      updateSelectedOptionUi();
      return;
    }

    const media = data.media;
    card.hidden = false;

    if (thumb && fallback) {
      if (media.thumbnail) {
        thumb.hidden = false;
        fallback.hidden = true;
        thumb.src = String(media.thumbnail);
        thumb.alt = String(media.title || "Ảnh xem trước video");
        thumb.onerror = () => {
          thumb.hidden = true;
          fallback.hidden = false;
        };
      } else {
        thumb.removeAttribute("src");
        thumb.hidden = true;
        fallback.hidden = false;
      }
    }

    if (platformBadge) platformBadge.textContent = String(media.platform_label || "Video");
    if (durationBadge) {
      durationBadge.textContent = String(media.duration_text || "Không rõ");
      durationBadge.hidden = !media.duration_text;
    }
    if (titleEl) titleEl.textContent = String(media.title || "Video không tiêu đề");
    if (uploaderEl) {
      uploaderEl.textContent = media.uploader ? `Kênh / tác giả: ${media.uploader}` : "";
      uploaderEl.hidden = !media.uploader;
    }
    if (ffmpegNote) {
      ffmpegNote.textContent = data.ffmpeg_available
        ? "FFmpeg sẵn sàng (hỗ trợ ghép 1080p MP4 & xuất MP3)"
        : "Chưa có FFmpeg — dùng định dạng gốc (MP4 / M4A)";
    }

    const options = Array.isArray(data.options) ? data.options : [];
    const availableDefault = options.find((opt) => opt.id === data.default_preset && opt.available);
    const firstAvailable = options.find((opt) => opt.available);
    selectedPreset = availableDefault?.id || firstAvailable?.id || null;

    grid.replaceChildren();
    options.forEach((opt) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "media-option-card";
      btn.dataset.preset = String(opt.id || "");
      btn.setAttribute("role", "radio");
      btn.disabled = !opt.available;

      const top = document.createElement("div");
      top.className = "media-option-top";

      const label = document.createElement("strong");
      label.textContent = String(opt.label || "");

      const tag = document.createElement("span");
      tag.className = "media-option-tag";
      tag.textContent = String(opt.container || "");

      top.append(label, tag);

      const meta = document.createElement("div");
      meta.className = "media-option-meta";
      const metaParts = [];
      if (opt.resolution) metaParts.push(String(opt.resolution));
      if (opt.fps) metaParts.push(`${opt.fps} fps`);
      if (opt.estimated_size) metaParts.push(String(opt.estimated_size));
      meta.textContent = metaParts.join(" · ");

      const desc = document.createElement("p");
      desc.className = "media-option-desc";
      desc.textContent = String(opt.description || "");

      btn.append(top, meta, desc);
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        selectedPreset = opt.id;
        updateSelectedOptionUi();
      });
      grid.appendChild(btn);
    });

    updateSelectedOptionUi();
  }

  function renderProgress(job) {
    const card = $("media-progress-card");
    const titleEl = $("media-progress-title");
    const stageEl = $("media-progress-stage");
    const pctEl = $("media-progress-pct");
    const barWrap = $("media-progress-bar-wrap");
    const fill = $("media-progress-fill");
    const statsEl = $("media-progress-stats");
    const saveLink = $("media-save-link");
    const cancelBtn = $("media-cancel-btn");
    if (!card) return;

    if (!job) {
      card.hidden = true;
      return;
    }

    card.hidden = false;
    card.dataset.jobStatus = String(job.status || "");

    const pct = Math.max(0, Math.min(100, Number(job.percent) || 0));
    if (titleEl) titleEl.textContent = String(job.filename || job.title || "Đang tải tệp");
    if (stageEl) {
      stageEl.textContent = job.status === "failed" && job.error
        ? String(job.error)
        : (STAGE_LABELS[job.stage] || STAGE_LABELS[job.status] || "Đang xử lý…");
    }
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
    if (barWrap) barWrap.setAttribute("aria-valuenow", String(Math.round(pct)));
    if (fill) fill.style.width = `${pct}%`;

    const statParts = [];
    if (job.status === "completed") {
      if (job.file_size_text) statParts.push(`Dung lượng: ${job.file_size_text}`);
    } else if (job.status === "running" || job.status === "pending") {
      const doneText = formatBytes(job.downloaded_bytes);
      const totalText = formatBytes(job.total_bytes);
      if (doneText && totalText) statParts.push(`${doneText} / ${totalText}`);
      else if (doneText) statParts.push(doneText);
      const speedText = formatBytes(job.speed);
      if (speedText) statParts.push(`${speedText}/s`);
      const etaText = formatEta(job.eta);
      if (etaText) statParts.push(`Còn ~${etaText}`);
    }
    if (statsEl) statsEl.textContent = statParts.join(" · ");

    const isRunning = job.status === "pending" || job.status === "running";
    if (cancelBtn) {
      cancelBtn.hidden = !isRunning;
      cancelBtn.disabled = job.stage === "canceling";
    }
    if (saveLink) {
      const ready = job.status === "completed" && Boolean(job.download_url);
      saveLink.hidden = !ready;
      if (ready) {
        saveLink.href = String(job.download_url);
        if (job.filename) saveLink.download = String(job.filename);
      } else {
        saveLink.removeAttribute("href");
      }
    }
  }

  function renderHistory(jobs) {
    const card = $("media-history-card");
    const list = $("media-history-list");
    if (!card || !list) return;

    const items = Array.isArray(jobs) ? jobs : [];
    if (items.length === 0) {
      card.hidden = true;
      list.replaceChildren();
      return;
    }

    card.hidden = false;
    list.replaceChildren();

    const presetLabels = {
      video_1080: "1080p MP4",
      video_720: "720p MP4",
      audio_only: "Audio",
    };

    items.forEach((job) => {
      const li = document.createElement("li");
      li.className = "media-history-item";

      const info = document.createElement("div");
      info.className = "media-history-info";

      const name = document.createElement("strong");
      name.textContent = String(job.filename || job.title || job.job_id);

      const meta = document.createElement("span");
      const metaParts = [
        String(job.platform_label || "Video"),
        presetLabels[job.preset] || String(job.preset || ""),
      ];
      if (job.file_size_text) metaParts.push(String(job.file_size_text));
      if (job.status !== "completed") {
        metaParts.push(STAGE_LABELS[job.status] || String(job.status || ""));
      }
      meta.textContent = metaParts.filter(Boolean).join(" · ");
      info.append(name, meta);

      const actions = document.createElement("div");
      actions.className = "media-history-actions";

      if (job.status === "completed" && job.download_url) {
        const dl = document.createElement("a");
        dl.className = "ui-btn ui-btn-compact";
        dl.href = String(job.download_url);
        if (job.filename) dl.download = String(job.filename);
        dl.textContent = "Lưu tệp";
        actions.appendChild(dl);
      }

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "ui-btn ui-btn-ghost ui-btn-compact";
      delBtn.textContent = "Xóa";
      delBtn.addEventListener("click", async () => {
        delBtn.disabled = true;
        try {
          await apiJson(`/api/media/jobs/${encodeURIComponent(job.job_id)}`, {
            method: "DELETE",
          });
          if (activeJobId === job.job_id) {
            clearPoll();
            activeJobId = null;
            renderProgress(null);
            updateSelectedOptionUi();
          }
          await loadHistory();
        } catch (err) {
          delBtn.disabled = false;
          window.showToast?.(`Không xóa được tệp: ${err.message}`, "error");
        }
      });
      actions.appendChild(delBtn);

      li.append(info, actions);
      list.appendChild(li);
    });
  }

  async function loadHistory() {
    try {
      const data = await apiJson("/api/media/jobs");
      const jobs = Array.isArray(data?.jobs) ? data.jobs : [];
      renderHistory(jobs);
      const running = jobs.find((j) => j.status === "pending" || j.status === "running");
      if (running && !activeJobId) {
        activeJobId = running.job_id;
        renderProgress(running);
        updateSelectedOptionUi();
        schedulePoll(running.job_id);
      }
    } catch (_) {}
  }

  function schedulePoll(jobId) {
    clearPoll();
    pollTimer = window.setTimeout(async () => {
      if (activeJobId !== jobId) return;
      try {
        const snapshot = await apiJson(`/api/media/jobs/${encodeURIComponent(jobId)}`);
        if (activeJobId !== jobId) return;
        renderProgress(snapshot);
        if (snapshot.status === "pending" || snapshot.status === "running") {
          schedulePoll(jobId);
          return;
        }
        activeJobId = null;
        updateSelectedOptionUi();
        await loadHistory();
        if (snapshot.status === "completed") {
          window.showToast?.("Tải xong! Đang lưu tệp về máy.", "success");
          if (triggeredDownloadJobId !== snapshot.job_id && snapshot.download_url) {
            triggeredDownloadJobId = snapshot.job_id;
            triggerFileDownload(snapshot.download_url, snapshot.filename);
          }
        } else if (snapshot.status === "failed") {
          window.showToast?.(snapshot.error || "Tải xuống thất bại.", "error");
        } else if (snapshot.status === "canceled") {
          window.showToast?.("Đã hủy lượt tải.", "info");
        }
      } catch (err) {
        if (activeJobId !== jobId) return;
        activeJobId = null;
        updateSelectedOptionUi();
        window.showToast?.(`Mất kết nối theo dõi tiến trình: ${err.message}`, "error");
      }
    }, POLL_INTERVAL_MS);
  }

  async function handleInspect(event) {
    event?.preventDefault();
    const input = $("media-url");
    const btn = $("media-inspect-btn");
    const rawUrl = String(input?.value || "").trim();
    if (!rawUrl) {
      window.showToast?.("Vui lòng dán liên kết TikTok, YouTube hoặc Facebook.", "error");
      input?.focus();
      return;
    }

    const seq = ++inspectSeq;
    inspected = null;
    selectedPreset = null;
    renderPicker(null);

    if (btn) {
      btn.disabled = true;
      btn.textContent = "Đang kiểm tra…";
    }
    const safetyCard = $("media-safety-card");
    const safetyBadge = $("media-safety-badge");
    const safetySummary = $("media-safety-summary");
    const safetyList = $("media-safety-checks");
    if (safetyCard && safetyBadge && safetySummary && safetyList) {
      safetyCard.hidden = false;
      safetyCard.dataset.safetyState = "checking";
      safetyBadge.className = "media-safety-badge";
      safetyBadge.textContent = "Đang kiểm tra";
      safetySummary.textContent = "Đang xác minh an toàn liên kết và phân tích định dạng video…";
      safetyList.replaceChildren();
    }

    try {
      const data = await apiJson("/api/media/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: rawUrl }),
      });
      if (seq !== inspectSeq) return;
      inspected = data;
      renderSafety(data.safety);
      renderPicker(data);
      if (data.safe) {
        window.showToast?.("Liên kết an toàn — hãy chọn chất lượng để tải xuống.", "success");
      } else {
        window.showToast?.(data.safety?.summary || "Liên kết không an toàn hoặc chưa được hỗ trợ.", "error");
      }
    } catch (err) {
      if (seq !== inspectSeq) return;
      inspected = null;
      renderSafety({
        safe: false,
        summary: err.message || "Không thể kiểm tra liên kết.",
        hostname: "",
        checks: [],
      });
      renderPicker(null);
      window.showToast?.(err.message || "Không thể kiểm tra liên kết.", "error");
    } finally {
      if (seq === inspectSeq && btn) {
        btn.disabled = false;
        btn.textContent = "Kiểm tra liên kết";
      }
    }
  }

  async function handleStartDownload() {
    if (!inspected?.safe || !inspected?.media || !selectedPreset || activeJobId) return;
    const downloadBtn = $("media-download-btn");
    if (downloadBtn) {
      downloadBtn.disabled = true;
      downloadBtn.textContent = "Đang khởi tạo…";
    }

    try {
      const url = inspected.media.webpage_url || inspected.safety?.resolved_url || inspected.safety?.normalized_url;
      const snapshot = await apiJson("/api/media/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          preset: selectedPreset,
          title: inspected.media.title || "",
          is_vertical: Boolean(inspected.media.is_vertical),
        }),
      });
      activeJobId = snapshot.job_id;
      triggeredDownloadJobId = null;
      renderProgress(snapshot);
      schedulePoll(snapshot.job_id);
      await loadHistory();
    } catch (err) {
      window.showToast?.(`Không thể bắt đầu tải: ${err.message}`, "error");
    } finally {
      if (downloadBtn) {
        downloadBtn.textContent = "Tải xuống ngay";
      }
      updateSelectedOptionUi();
    }
  }

  async function handleCancelDownload() {
    if (!activeJobId) return;
    const cancelBtn = $("media-cancel-btn");
    if (cancelBtn) cancelBtn.disabled = true;
    try {
      const snapshot = await apiJson(`/api/media/jobs/${encodeURIComponent(activeJobId)}/cancel`, {
        method: "POST",
      });
      renderProgress(snapshot);
    } catch (err) {
      if (cancelBtn) cancelBtn.disabled = false;
      window.showToast?.(`Không hủy được tác vụ: ${err.message}`, "error");
    }
  }

  function handleReset() {
    inspectSeq += 1;
    inspected = null;
    selectedPreset = null;
    const input = $("media-url");
    if (input) {
      input.value = "";
      input.focus();
    }
    renderSafety(null);
    renderPicker(null);
  }

  async function handlePaste() {
    const input = $("media-url");
    if (!input) return;
    if (!navigator.clipboard?.readText) {
      input.focus();
      window.showToast?.("Trình duyệt không hỗ trợ đọc clipboard tự động — hãy bấm Ctrl+V.", "info");
      return;
    }
    try {
      const text = (await navigator.clipboard.readText()).trim();
      if (!text) {
        window.showToast?.("Clipboard đang trống.", "info");
        return;
      }
      input.value = text;
      input.focus();
      await handleInspect();
    } catch (_) {
      input.focus();
      window.showToast?.("Không đọc được clipboard — hãy bấm Ctrl+V vào ô liên kết.", "info");
    }
  }

  function setupMediaDownloader() {
    if (initialized) return;
    initialized = true;
    $("media-inspect-form")?.addEventListener("submit", handleInspect);
    $("media-paste-btn")?.addEventListener("click", handlePaste);
    $("media-download-btn")?.addEventListener("click", handleStartDownload);
    $("media-cancel-btn")?.addEventListener("click", handleCancelDownload);
    $("media-reset-btn")?.addEventListener("click", handleReset);
  }

  window.openMediaView = function openMediaView() {
    setupMediaDownloader();
    loadHistory();
  };

  document.addEventListener("DOMContentLoaded", setupMediaDownloader);
})();
