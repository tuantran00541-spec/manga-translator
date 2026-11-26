(() => {
  "use strict";

  const POLL_INTERVAL_MS = 650;
  const CUSTOM_MODEL_VALUE = "__custom__";
  const SAME_PROVIDER_VALUE = "";
  const PIPER_TTS_PROVIDER_ID = "piper_tts";
  const EDGE_TTS_PROVIDER_ID = "edge_tts";
  const LOCAL_TTS_IDS = new Set([PIPER_TTS_PROVIDER_ID, EDGE_TTS_PROVIDER_ID]);
  const STORAGE_PROVIDER = "manga_media_ai_provider";
  const STORAGE_TTS_PROVIDER = "manga_media_ai_tts_provider";
  const STAGE_LABELS = {
    preparing: "Đang chuẩn bị luồng tải…",
    downloading: "Đang tải xuống dữ liệu…",
    merging: "Đang ghép hình ảnh & âm thanh…",
    extracting_audio: "Đang tách âm thanh từ video…",
    transcribing: "AI đang nghe hiểu & dịch lời thoại…",
    synthesizing_tts: "AI đang tạo giọng lồng tiếng…",
    rendering_video: "FFmpeg đang trộn âm & gắn phụ đề cứng…",
    canceling: "Đang hủy tác vụ…",
    completed: "Hoàn tất! Tệp đã sẵn sàng lưu về máy.",
    canceled: "Đã hủy tải xuống.",
    failed: "Tải xuống thất bại.",
  };

  let inspected = null;
  let capabilities = null;
  let providersMap = {};
  const providerModelsCache = {};
  let selectedPreset = null;
  let selectedAiMode = "none";
  let activeJobId = null;
  let pollTimer = null;
  let inspectSeq = 0;
  let triggeredDownloadJobId = null;
  let initialized = false;

  const $ = (id) => document.getElementById(id);

  function storage(action, key, value) {
    try {
      if (action === "get") return localStorage.getItem(key);
      if (action === "set") localStorage.setItem(key, value);
    } catch (_) {}
    return null;
  }

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

  function formatTimestamp(sec) {
    const val = Math.max(0, Number(sec) || 0);
    const mins = Math.floor(val / 60);
    const secs = Math.floor(val % 60);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
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

  function getProviderList() {
    const source = (window.aiProviderSettings && Object.keys(window.aiProviderSettings).length)
      ? window.aiProviderSettings
      : providersMap;
    return Object.values(source || {}).filter((p) => p && p.id);
  }

  function getProviderInfo(providerId) {
    const id = String(providerId || "").trim();
    if (!id) return null;
    return window.aiProviderSettings?.[id] || providersMap?.[id] || null;
  }

  function effectiveTtsProviderId() {
    const explicit = String($("media-ai-tts-provider")?.value || "").trim();
    return explicit || String($("media-ai-provider")?.value || "").trim();
  }

  function filterModelsForRole(rawModels, protocol, role) {
    const list = Array.isArray(rawModels) ? rawModels.filter((m) => typeof m === "string" && m.trim()) : [];
    const excluded = ["embedding", "imagen", "nano-banana", "image", "lyria", "omni", "genie", "decision", "antigravity", "deep-research", "veo", "aqa", "robotics", "computer-use", "-live", "moderation", "dall-e"];
    const legacy = ["gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-1.0", "gemini-pro"];

    if (role === "tts") {
      const ttsMatches = list.filter((id) => {
        const low = id.toLowerCase();
        if (protocol === "gemini" && legacy.some((p) => low.startsWith(p))) return false;
        return low.includes("tts") || low.includes("speech");
      });
      if (protocol === "gemini") {
        const seeded = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"];
        return [...new Set([...seeded, ...ttsMatches])];
      }
      if (ttsMatches.length) return ttsMatches;
      return protocol === "openai" ? ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"] : list;
    }

    // Transcribe / translate role
    const cleaned = list.filter((id) => {
      const low = id.toLowerCase();
      if (low.includes("tts") || low.includes("transcribe")) return false;
      if (excluded.some((tok) => low.includes(tok))) return false;
      if (protocol === "gemini" && legacy.some((p) => low.startsWith(p))) return false;
      return true;
    });

    if (protocol === "gemini") {
      const priority = ["gemini-4-hydrogen", "gemini-4-argon", "gemini-4-neon", "gemini-4-carbon", "gemini-3.8-flash", "gemini-3.5-flash-lite"];
      const head = priority.filter((id) => cleaned.includes(id) || id.startsWith("gemini-4-"));
      return [...new Set([...head, ...cleaned])];
    }
    return cleaned.length ? cleaned : list;
  }

  function populateModelSelect(selectEl, models, savedKey) {
    if (!selectEl) return;
    const prev = selectEl.value || storage("get", savedKey) || "";
    selectEl.replaceChildren();
    models.forEach((id) => {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      selectEl.appendChild(opt);
    });
    const customOpt = document.createElement("option");
    customOpt.value = CUSTOM_MODEL_VALUE;
    customOpt.textContent = "Tùy chỉnh ID model…";
    selectEl.appendChild(customOpt);

    if (prev && [...selectEl.options].some((o) => o.value === prev)) {
      selectEl.value = prev;
    } else if (models.length > 0) {
      selectEl.selectedIndex = 0;
    } else {
      selectEl.value = CUSTOM_MODEL_VALUE;
    }
  }

  async function fetchModelsForProvider(providerId) {
    const pid = String(providerId || "").trim();
    if (!pid) return [];
    const info = getProviderInfo(pid);
    if (info && !info.configured) {
      providerModelsCache[pid] = [];
      return [];
    }
    const cacheApi = window.aiModelCache;
    const cached = cacheApi?.getCached?.(pid);
    if (cached && Array.isArray(cached.models)) {
      providerModelsCache[pid] = cached.models;
      if (cached.fresh) return cached.models;
    }
    try {
      const models = cacheApi?.fetchModels
        ? await cacheApi.fetchModels(pid)
        : ((await apiJson(`/api/visual_qc/providers/${encodeURIComponent(pid)}/models`)).models || []);
      providerModelsCache[pid] = Array.isArray(models) ? models : [];
      return providerModelsCache[pid];
    } catch (_) {
      return providerModelsCache[pid] || [];
    }
  }

  async function refreshSelectedModels() {
    const pid = String($("media-ai-provider")?.value || "").trim();
    const ttsPid = effectiveTtsProviderId();
    const pInfo = getProviderInfo(pid);
    const ttsInfo = getProviderInfo(ttsPid);

    const [trRaw, ttsRaw] = await Promise.all([
      fetchModelsForProvider(pid),
      (ttsPid === pid || LOCAL_TTS_IDS.has(ttsPid)) ? Promise.resolve(null) : fetchModelsForProvider(ttsPid),
    ]);

    const trModels = filterModelsForRole(trRaw, pInfo?.protocol || "openai", "transcribe");
    const ttsModels = filterModelsForRole(
      ttsPid === pid ? trRaw : (ttsRaw || []),
      ttsInfo?.protocol || "openai",
      "tts",
    );

    populateModelSelect($("media-ai-transcribe-model"), trModels, `manga_media_tr_model_${pid}`);
    populateModelSelect($("media-ai-tts-model"), ttsModels, `manga_media_tts_model_${ttsPid}`);
    syncAiStudioUi();
  }

  function renderProviderSelects() {
    const list = getProviderList();
    const configured = list.filter((p) => p.configured);
    const ordered = [...configured, ...list.filter((p) => !p.configured)];

    const mainSel = $("media-ai-provider");
    const ttsSel = $("media-ai-tts-provider");

    if (mainSel) {
      const prev = mainSel.value || storage("get", STORAGE_PROVIDER) || "";
      mainSel.replaceChildren();
      ordered.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.configured ? `${p.label}` : `${p.label} (chưa có khóa)`;
        mainSel.appendChild(opt);
      });
      if (prev && [...mainSel.options].some((o) => o.value === prev)) {
        mainSel.value = prev;
      } else if (configured.length > 0) {
        mainSel.value = configured[0].id;
      }
    }

    if (ttsSel) {
      const piperReady = Boolean(capabilities?.piper_tts_available ?? true);
      const edgeReady = Boolean(capabilities?.edge_tts_available ?? true);
      const defaultTts = piperReady ? PIPER_TTS_PROVIDER_ID : (edgeReady ? EDGE_TTS_PROVIDER_ID : SAME_PROVIDER_VALUE);
      const prevTts = ttsSel.value || storage("get", STORAGE_TTS_PROVIDER) || defaultTts;
      ttsSel.replaceChildren();
      if (piperReady) {
        const piperOpt = document.createElement("option");
        piperOpt.value = PIPER_TTS_PROVIDER_ID;
        piperOpt.textContent = "Piper ONNX Local (Siêu tốc ~50ms — Offline, 0 quota)";
        ttsSel.appendChild(piperOpt);
      }
      if (edgeReady) {
        const edgeOpt = document.createElement("option");
        edgeOpt.value = EDGE_TTS_PROVIDER_ID;
        edgeOpt.textContent = "Microsoft Edge TTS (Miễn phí — 0 quota)";
        ttsSel.appendChild(edgeOpt);
      }
      const sameOpt = document.createElement("option");
      sameOpt.value = SAME_PROVIDER_VALUE;
      sameOpt.textContent = "Cùng nhà cung cấp AI ở trên";
      ttsSel.appendChild(sameOpt);
      ordered.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.configured ? `${p.label}` : `${p.label} (chưa có khóa)`;
        ttsSel.appendChild(opt);
      });
      if ([...ttsSel.options].some((o) => o.value === prevTts)) {
        ttsSel.value = prevTts;
      } else {
        ttsSel.value = defaultTts;
      }
    }
  }

  async function syncProvidersFromSettings() {
    if (!window.aiProviderSettings || !Object.keys(window.aiProviderSettings).length) {
      try {
        const data = await apiJson("/api/visual_qc/settings");
        if (data?.providers && typeof data.providers === "object") {
          providersMap = data.providers;
        }
      } catch (_) {}
    } else {
      providersMap = window.aiProviderSettings;
    }
    renderProviderSelects();
    await refreshSelectedModels();
  }

  function resolvedModelValue(selectId, customInputId) {
    const sel = $(selectId);
    const val = String(sel?.value || "").trim();
    if (val === CUSTOM_MODEL_VALUE) {
      return String($(customInputId)?.value || "").trim();
    }
    return val;
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

  function syncAiStudioUi() {
    const isAudioOnly = selectedPreset === "audio_only";
    const providers = getProviderList();
    const configuredCount = providers.filter((p) => p.configured).length;
    const ffmpegReady = Boolean(inspected?.ffmpeg_available ?? capabilities?.ffmpeg_available);
    const aiAllowed = !isAudioOnly && configuredCount > 0 && ffmpegReady;

    if (!aiAllowed && selectedAiMode !== "none") {
      selectedAiMode = "none";
    }

    const note = $("media-ai-note");
    if (note) {
      if (isAudioOnly) {
        note.textContent = "Chọn video 1080p hoặc 720p để dùng Hardsub / Lồng tiếng AI.";
      } else if (configuredCount === 0) {
        note.textContent = "Chưa có nhà cung cấp AI nào được cấu hình trong Cài đặt.";
      } else if (!ffmpegReady) {
        note.textContent = "Cần FFmpeg để gắn phụ đề cứng & lồng tiếng.";
      } else {
        note.textContent = `${configuredCount} nhà cung cấp AI sẵn sàng từ Cài đặt`;
      }
    }

    const modesWrap = $("media-ai-modes");
    if (modesWrap) {
      modesWrap.querySelectorAll(".media-ai-mode-btn").forEach((btn) => {
        const mode = btn.dataset.aiMode || "none";
        const disabled = mode !== "none" && !aiAllowed;
        btn.disabled = disabled;
        const selected = mode === selectedAiMode && !disabled;
        btn.classList.toggle("is-selected", selected);
        btn.setAttribute("aria-checked", String(selected));
      });
    }

    const configBox = $("media-ai-config");
    const trSel = $("media-ai-transcribe-model");
    const trCustomWrap = $("media-ai-transcribe-custom-wrap");
    const ttsProviderField = $("media-ai-tts-provider-field");
    const ttsField = $("media-ai-tts-model-field");
    const ttsSel = $("media-ai-tts-model");
    const ttsCustomWrap = $("media-ai-tts-custom-wrap");
    const voiceField = $("media-ai-voice-field");
    const styleField = $("media-ai-style-field");
    const animField = $("media-ai-anim-field");
    const burnWrap = $("media-ai-burn-wrap");
    const burnCheck = $("media-ai-burn-sub");
    const bilingualWrap = $("media-ai-bilingual-wrap");
    const bgWrap = $("media-ai-bg-wrap");

    const showConfig = selectedAiMode !== "none" && aiAllowed;
    if (configBox) configBox.hidden = !showConfig;
    const isDub = selectedAiMode === "dub";
    const isLocalTts = isDub && LOCAL_TTS_IDS.has(effectiveTtsProviderId());
    const showSubOptions = selectedAiMode === "hardsub" || (isDub && Boolean(burnCheck?.checked));

    if (trCustomWrap) trCustomWrap.hidden = trSel?.value !== CUSTOM_MODEL_VALUE;
    if (ttsProviderField) ttsProviderField.hidden = !isDub;
    if (ttsField) ttsField.hidden = !isDub || isLocalTts;
    if (ttsCustomWrap) ttsCustomWrap.hidden = !isDub || isLocalTts || ttsSel?.value !== CUSTOM_MODEL_VALUE;
    if (voiceField) voiceField.hidden = !isDub;
    if (burnWrap) burnWrap.hidden = !isDub;
    if (bgWrap) bgWrap.hidden = !isDub;
    if (styleField) styleField.hidden = !showSubOptions;
    if (animField) animField.hidden = !showSubOptions;
    if (bilingualWrap) bilingualWrap.hidden = !showSubOptions;

    const downloadBtn = $("media-download-btn");
    if (downloadBtn && !activeJobId) {
      if (selectedAiMode === "hardsub") {
        downloadBtn.textContent = "Tải & Gắn phụ đề cứng";
      } else if (selectedAiMode === "dub") {
        downloadBtn.textContent = burnCheck?.checked
          ? "Tải, Lồng tiếng & Hardsub"
          : "Tải & Lồng tiếng AI";
      } else {
        downloadBtn.textContent = "Tải xuống ngay";
      }
    }
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
    syncAiStudioUi();
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
      selectedAiMode = "none";
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

    void syncProvidersFromSettings();

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

  function renderTranscript(job) {
    const box = $("media-transcript-box");
    const summary = $("media-transcript-summary");
    const list = $("media-transcript-list");
    if (!box || !summary || !list) return;

    const segs = Array.isArray(job?.segments) ? job.segments : [];
    if (job?.status !== "completed" || segs.length === 0) {
      box.hidden = true;
      list.replaceChildren();
      return;
    }

    box.hidden = false;
    const srcInfo = job.source_language && job.source_language !== "unknown"
      ? ` · Nguồn: ${job.source_language}`
      : "";
    summary.textContent = `Xem lời thoại & bản dịch (${job.segment_count || segs.length} câu${srcInfo})`;
    list.replaceChildren();

    segs.forEach((seg) => {
      const li = document.createElement("li");
      li.className = "media-transcript-item";

      const ts = document.createElement("span");
      ts.className = "media-transcript-ts";
      ts.textContent = `${formatTimestamp(seg.start_sec)}–${formatTimestamp(seg.end_sec)}`;

      const body = document.createElement("div");
      body.className = "media-transcript-body";

      const tr = document.createElement("strong");
      tr.textContent = String(seg.translated_text || "");
      body.appendChild(tr);

      if (seg.original_text && seg.original_text !== seg.translated_text) {
        const orig = document.createElement("span");
        orig.textContent = String(seg.original_text);
        body.appendChild(orig);
      }

      li.append(ts, body);
      list.appendChild(li);
    });
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
    const srtLink = $("media-srt-link");
    const cancelBtn = $("media-cancel-btn");
    if (!card) return;

    if (!job) {
      card.hidden = true;
      renderTranscript(null);
      return;
    }

    card.hidden = false;
    card.dataset.jobStatus = String(job.status || "");

    const pct = Math.max(0, Math.min(100, Number(job.percent) || 0));
    if (titleEl) titleEl.textContent = String(job.filename || job.title || "Đang tải tệp");
    if (stageEl) {
      if (job.status === "failed" && job.error) {
        stageEl.textContent = String(job.error);
      } else if (job.status === "completed") {
        stageEl.textContent = STAGE_LABELS.completed;
      } else if (job.stage_detail) {
        stageEl.textContent = String(job.stage_detail);
      } else {
        stageEl.textContent = STAGE_LABELS[job.stage] || STAGE_LABELS[job.status] || "Đang xử lý…";
      }
    }
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
    if (barWrap) barWrap.setAttribute("aria-valuenow", String(Math.round(pct)));
    if (fill) fill.style.width = `${pct}%`;

    const statParts = [];
    if (job.status === "completed") {
      if (job.file_size_text) statParts.push(`Dung lượng: ${job.file_size_text}`);
      if (job.segment_count > 0) statParts.push(`${job.segment_count} câu thoại`);
      if (job.transcribe_model) statParts.push(job.transcribe_model);
      if (job.tts_model) statParts.push(job.tts_model);
    } else if (job.status === "running" || job.status === "pending") {
      if (job.stage === "downloading") {
        const doneText = formatBytes(job.downloaded_bytes);
        const totalText = formatBytes(job.total_bytes);
        if (doneText && totalText) statParts.push(`${doneText} / ${totalText}`);
        else if (doneText) statParts.push(doneText);
        const speedText = formatBytes(job.speed);
        if (speedText) statParts.push(`${speedText}/s`);
        const etaText = formatEta(job.eta);
        if (etaText) statParts.push(`Còn ~${etaText}`);
      }
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
    if (srtLink) {
      const hasSrt = job.status === "completed" && Boolean(job.subtitle_url);
      srtLink.hidden = !hasSrt;
      if (hasSrt) {
        srtLink.href = String(job.subtitle_url);
        if (job.subtitle_filename) srtLink.download = String(job.subtitle_filename);
      } else {
        srtLink.removeAttribute("href");
      }
    }
    renderTranscript(job);
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
    const aiLabels = {
      hardsub: "Hardsub AI",
      dub: "Lồng tiếng AI",
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
      if (aiLabels[job.ai_mode]) metaParts.push(aiLabels[job.ai_mode]);
      if (job.file_size_text) metaParts.push(String(job.file_size_text));
      if (job.status !== "completed") {
        metaParts.push(STAGE_LABELS[job.status] || String(job.status || ""));
      }
      meta.textContent = metaParts.filter(Boolean).join(" · ");
      info.append(name, meta);

      const actions = document.createElement("div");
      actions.className = "media-history-actions";

      if (job.status === "completed" && job.subtitle_url) {
        const srt = document.createElement("a");
        srt.className = "ui-btn ui-btn-ghost ui-btn-compact";
        srt.href = String(job.subtitle_url);
        if (job.subtitle_filename) srt.download = String(job.subtitle_filename);
        srt.textContent = "SRT";
        actions.appendChild(srt);
      }

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

  async function loadCapabilities() {
    try {
      capabilities = await apiJson("/api/media/capabilities");
    } catch (_) {}
    await syncProvidersFromSettings();
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
          window.showToast?.("Hoàn tất! Đang lưu tệp về máy.", "success");
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
      window.showToast?.("Vui lòng dán liên kết video công khai trước khi kiểm tra.", "error");
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
        window.showToast?.("Liên kết an toàn — hãy chọn chất lượng và chế độ xử lý.", "success");
      } else {
        window.showToast?.(data.safety?.summary || "Liên kết không an toàn hoặc chưa có video khả dụng.", "error");
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

    const aiMode = selectedPreset === "audio_only" ? "none" : selectedAiMode;
    const providerId = String($("media-ai-provider")?.value || "gemini");
    const ttsProviderId = String($("media-ai-tts-provider")?.value || "");
    const rawAsr = String($("media-ai-asr")?.value || "auto:small");
    const [asrEngine, whisperModel] = rawAsr.includes(":") ? rawAsr.split(":", 2) : [rawAsr, "small"];
    const payload = {
      url: inspected.media.webpage_url || inspected.safety?.resolved_url || inspected.safety?.normalized_url,
      preset: selectedPreset,
      title: inspected.media.title || "",
      is_vertical: Boolean(inspected.media.is_vertical),
      ai_mode: aiMode,
      provider_id: providerId,
      tts_provider_id: ttsProviderId,
      target_lang: String($("media-ai-lang")?.value || "vi"),
      voice: String($("media-ai-voice")?.value || "auto"),
      burn_subtitles: Boolean($("media-ai-burn-sub")?.checked ?? true),
      bilingual: Boolean($("media-ai-bilingual")?.checked ?? false),
      subtitle_style: String($("media-ai-sub-style")?.value || "classic"),
      subtitle_animation: String($("media-ai-sub-anim")?.value || "reveal"),
      keep_original_audio: Boolean($("media-ai-keep-bg")?.checked ?? true),
      transcribe_model: resolvedModelValue("media-ai-transcribe-model", "media-ai-transcribe-custom"),
      tts_model: LOCAL_TTS_IDS.has(effectiveTtsProviderId())
        ? ""
        : resolvedModelValue("media-ai-tts-model", "media-ai-tts-custom"),
      context_hint: String($("media-ai-context")?.value || "").trim(),
      asr_engine: asrEngine || "auto",
      whisper_model: whisperModel || "small",
    };

    try {
      const snapshot = await apiJson("/api/media/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      activeJobId = snapshot.job_id;
      triggeredDownloadJobId = null;
      renderProgress(snapshot);
      schedulePoll(snapshot.job_id);
      await loadHistory();
    } catch (err) {
      window.showToast?.(`Không thể bắt đầu tải: ${err.message}`, "error");
    } finally {
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
    selectedAiMode = "none";
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
    $("media-ai-open-settings")?.addEventListener("click", () => {
      document.getElementById("settings-toggle")?.click();
    });
    $("media-ai-provider")?.addEventListener("change", () => {
      const pid = String($("media-ai-provider")?.value || "");
      if (pid) storage("set", STORAGE_PROVIDER, pid);
      void refreshSelectedModels();
    });
    $("media-ai-tts-provider")?.addEventListener("change", () => {
      const pid = String($("media-ai-tts-provider")?.value || "");
      storage("set", STORAGE_TTS_PROVIDER, pid);
      void refreshSelectedModels();
    });
    $("media-ai-transcribe-model")?.addEventListener("change", () => {
      const pid = String($("media-ai-provider")?.value || "");
      const val = String($("media-ai-transcribe-model")?.value || "");
      if (pid && val && val !== CUSTOM_MODEL_VALUE) storage("set", `manga_media_tr_model_${pid}`, val);
      syncAiStudioUi();
    });
    $("media-ai-tts-model")?.addEventListener("change", () => {
      const pid = effectiveTtsProviderId();
      const val = String($("media-ai-tts-model")?.value || "");
      if (pid && val && val !== CUSTOM_MODEL_VALUE) storage("set", `manga_media_tts_model_${pid}`, val);
      syncAiStudioUi();
    });
    $("media-ai-burn-sub")?.addEventListener("change", syncAiStudioUi);
    $("media-ai-modes")?.querySelectorAll(".media-ai-mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        selectedAiMode = btn.dataset.aiMode || "none";
        syncAiStudioUi();
      });
    });

    document.addEventListener("ai-providers-updated", () => {
      void syncProvidersFromSettings();
    });
  }

  window.openMediaView = function openMediaView() {
    setupMediaDownloader();
    void loadCapabilities();
    void loadHistory();
  };

  document.addEventListener("DOMContentLoaded", setupMediaDownloader);
})();
