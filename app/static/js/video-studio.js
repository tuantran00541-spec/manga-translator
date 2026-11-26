(() => {
  "use strict";

  const POLL_MS = 1000;
  const PAGE_SIZE = 100;
  const CUSTOM_MODEL_VALUE = "__custom__";
  const PIPER_TTS_PROVIDER_ID = "piper_tts";
  const EDGE_TTS_PROVIDER_ID = "edge_tts";
  const LOCAL_TTS_IDS = new Set([PIPER_TTS_PROVIDER_ID, EDGE_TTS_PROVIDER_ID]);

  const VOICE_OPTIONS = [
    { id: "Puck", label: "Quang Huy / Puck (Nam Review Phim)" },
    { id: "Kore", label: "Ngọc Lan / Kore (Nữ Review Phim)" },
    { id: "Aoede", label: "Thu Hà / Aoede (Nữ nhẹ nhàng)" },
    { id: "Sulafat", label: "Yến Nhi / Sulafat (Nữ Nam trẻ trung)" },
    { id: "Leda", label: "Minh Anh / Leda (Nữ tin tức)" },
    { id: "Charon", label: "Trầm Vang / Charon (Nam kể chuyện)" },
    { id: "Fenrir", label: "Điềm Đạm / Fenrir (Nam thuyết minh)" },
  ];

  const STATUS_META = {
    idle: { label: "Chờ xử lý", cls: "is-idle" },
    ingesting: { label: "Đang tải video", cls: "is-running" },
    running_prepare: { label: "Đang bóc băng & dịch", cls: "is-running" },
    ready_to_edit: { label: "Sẵn sàng biên tập", cls: "is-ready" },
    running_render: { label: "Đang xuất bản", cls: "is-running" },
    completed: { label: "Hoàn tất", cls: "is-done" },
    paused: { label: "Đã tạm dừng (Checkpoint)", cls: "is-paused" },
    failed: { label: "Lỗi xử lý", cls: "is-failed" },
  };

  let initialized = false;
  let providersMap = {};
  let capabilities = { piper_tts_available: true, edge_tts_available: true };
  const providerModelsCache = {};

  let projects = [];
  let queueStatus = { running: false, project_ids: [], current_project_id: null, completed_ids: [], failed_ids: [] };
  let activeProjectId = null;
  let activeProject = null;
  let activeSegments = [];
  let activeSourceVideoUrl = null;
  let isDirty = false;
  let settingsDirty = false;
  let currentPage = 1;
  let currentPlayingSegId = null;
  let pollTimer = null;
  let previewAudio = null;
  let previewBlobUrl = null;
  let previewActiveBtn = null;
  let wasSelectedActive = false;

  const $ = (id) => document.getElementById(id);

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

  function toast(msg, type = "info") {
    if (typeof window.showToast === "function") window.showToast(msg, type);
  }

  function formatTimecode(sec) {
    const v = Math.max(0, Number(sec) || 0);
    const h = Math.floor(v / 3600);
    const m = Math.floor((v % 3600) / 60);
    const s = v % 60;
    const sStr = s.toFixed(1).padStart(4, "0");
    return h > 0
      ? `${h}:${String(m).padStart(2, "0")}:${sStr}`
      : `${String(m).padStart(2, "0")}:${sStr}`;
  }

  function parseTimecode(raw, fallback = 0) {
    const text = String(raw || "").trim().replace(",", ".");
    if (!text) return fallback;
    if (/^\d+(?:\.\d+)?$/.test(text)) return Math.max(0, parseFloat(text));
    const parts = text.split(":").map((p) => parseFloat(p));
    if (parts.some((n) => !Number.isFinite(n) || n < 0)) return fallback;
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    return fallback;
  }

  function getProviderList() {
    const src = (window.aiProviderSettings && Object.keys(window.aiProviderSettings).length)
      ? window.aiProviderSettings
      : providersMap;
    return Object.values(src || {}).filter((p) => p && p.id);
  }

  function getProviderInfo(pid) {
    const id = String(pid || "").trim();
    if (!id) return null;
    return window.aiProviderSettings?.[id] || providersMap?.[id] || null;
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
        return [...new Set(["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", ...ttsMatches])];
      }
      if (ttsMatches.length) return ttsMatches;
      return protocol === "openai" ? ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"] : list;
    }

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

  async function fetchModelsForProvider(providerId) {
    const pid = String(providerId || "").trim();
    if (!pid || LOCAL_TTS_IDS.has(pid)) return [];
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

  function populateModelSelect(selectEl, models, desiredValue) {
    if (!selectEl) return;
    selectEl.replaceChildren();
    const autoOpt = document.createElement("option");
    autoOpt.value = "";
    autoOpt.textContent = "Mặc định của nhà cung cấp";
    selectEl.appendChild(autoOpt);

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

    const wanted = String(desiredValue || "").trim();
    if (!wanted) {
      selectEl.value = "";
    } else if ([...selectEl.options].some((o) => o.value === wanted)) {
      selectEl.value = wanted;
    } else {
      selectEl.value = CUSTOM_MODEL_VALUE;
    }
  }

  function renderProviderSelects(desiredProvider, desiredTtsProvider) {
    const list = getProviderList();
    const configured = list.filter((p) => p.configured);
    const ordered = [...configured, ...list.filter((p) => !p.configured)];

    const mainSel = $("studio-set-provider");
    if (mainSel) {
      const target = desiredProvider || mainSel.value || (configured[0]?.id || ordered[0]?.id || "gemini");
      mainSel.replaceChildren();
      ordered.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.configured ? p.label : `${p.label} (chưa có khóa)`;
        mainSel.appendChild(opt);
      });
      if ([...mainSel.options].some((o) => o.value === target)) {
        mainSel.value = target;
      } else if (mainSel.options.length) {
        mainSel.selectedIndex = 0;
      }
    }

    const ttsSel = $("studio-set-tts-provider");
    if (ttsSel) {
      const targetTts = desiredTtsProvider || ttsSel.value || PIPER_TTS_PROVIDER_ID;
      ttsSel.replaceChildren();

      const piperOpt = document.createElement("option");
      piperOpt.value = PIPER_TTS_PROVIDER_ID;
      piperOpt.textContent = capabilities.piper_tts_available !== false
        ? "Piper ONNX Local (Offline ~50ms/câu — Khuyên dùng)"
        : "Piper ONNX Local (chưa cài piper-tts)";
      ttsSel.appendChild(piperOpt);

      const edgeOpt = document.createElement("option");
      edgeOpt.value = EDGE_TTS_PROVIDER_ID;
      edgeOpt.textContent = capabilities.edge_tts_available !== false
        ? "Microsoft Edge TTS (Miễn phí Online)"
        : "Microsoft Edge TTS (chưa cài edge-tts)";
      ttsSel.appendChild(edgeOpt);

      ordered.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.configured ? `Cloud: ${p.label}` : `Cloud: ${p.label} (chưa có khóa)`;
        ttsSel.appendChild(opt);
      });

      if ([...ttsSel.options].some((o) => o.value === targetTts)) {
        ttsSel.value = targetTts;
      } else {
        ttsSel.value = PIPER_TTS_PROVIDER_ID;
      }
    }
  }

  async function syncSettingsModels(desiredTrModel = null, desiredTtsModel = null) {
    const pid = String($("studio-set-provider")?.value || "").trim();
    const ttsPid = String($("studio-set-tts-provider")?.value || "").trim() || PIPER_TTS_PROVIDER_ID;
    const pInfo = getProviderInfo(pid);
    const ttsInfo = getProviderInfo(ttsPid);

    const isLocalTts = LOCAL_TTS_IDS.has(ttsPid);
    const ttsWrap = $("studio-set-tts-model-wrap");
    const ttsCustomWrap = $("studio-set-tts-custom-wrap");
    if (ttsWrap) ttsWrap.hidden = isLocalTts;
    if (ttsCustomWrap && isLocalTts) ttsCustomWrap.hidden = true;

    const [trRaw, ttsRaw] = await Promise.all([
      fetchModelsForProvider(pid),
      isLocalTts ? Promise.resolve([]) : (ttsPid === pid ? Promise.resolve(null) : fetchModelsForProvider(ttsPid)),
    ]);

    const trModels = filterModelsForRole(trRaw, pInfo?.protocol || "openai", "transcribe");
    const trTarget = desiredTrModel !== null ? desiredTrModel : getSelectedModelValue("studio-set-model", "studio-set-model-custom");
    populateModelSelect($("studio-set-model"), trModels, trTarget);
    const trIsCustom = $("studio-set-model")?.value === CUSTOM_MODEL_VALUE;
    if ($("studio-set-model-custom-wrap")) $("studio-set-model-custom-wrap").hidden = !trIsCustom;
    if (trIsCustom && desiredTrModel && $("studio-set-model-custom")) {
      $("studio-set-model-custom").value = desiredTrModel;
    }

    if (!isLocalTts) {
      const ttsModels = filterModelsForRole(ttsPid === pid ? trRaw : (ttsRaw || []), ttsInfo?.protocol || "openai", "tts");
      const ttsTarget = desiredTtsModel !== null ? desiredTtsModel : getSelectedModelValue("studio-set-tts-model", "studio-set-tts-custom");
      populateModelSelect($("studio-set-tts-model"), ttsModels, ttsTarget);
      const ttsIsCustom = $("studio-set-tts-model")?.value === CUSTOM_MODEL_VALUE;
      if (ttsCustomWrap) ttsCustomWrap.hidden = !ttsIsCustom;
      if (ttsIsCustom && desiredTtsModel && $("studio-set-tts-custom")) {
        $("studio-set-tts-custom").value = desiredTtsModel;
      }
    }
  }

  function getSelectedModelValue(selectId, customInputId) {
    const sel = $(selectId);
    if (!sel) return "";
    if (sel.value === CUSTOM_MODEL_VALUE) {
      return String($(customInputId)?.value || "").trim();
    }
    return String(sel.value || "").trim();
  }

  async function loadProvidersAndCapabilities() {
    const tasks = [];
    if (!window.aiProviderSettings || !Object.keys(window.aiProviderSettings).length) {
      tasks.push(
        apiJson("/api/visual_qc/settings")
          .then((data) => {
            if (Array.isArray(data?.providers)) {
              const map = {};
              data.providers.forEach((p) => { if (p?.id) map[p.id] = p; });
              providersMap = map;
            }
          })
          .catch(() => {})
      );
    }
    tasks.push(
      apiJson("/api/media/capabilities")
        .then((cap) => { if (cap && typeof cap === "object") capabilities = cap; })
        .catch(() => {})
    );
    await Promise.all(tasks);
    renderProviderSelects(activeProject?.settings?.provider_id, activeProject?.settings?.tts_provider_id);
    await syncSettingsModels(activeProject?.settings?.transcribe_model ?? "", activeProject?.settings?.tts_model ?? "");
  }

  function markDirty(segmentsChanged = true) {
    if (segmentsChanged) isDirty = true;
    else settingsDirty = true;
    const dot = $("studio-dirty-badge");
    if (dot) dot.hidden = !(isDirty || settingsDirty);
  }

  function clearDirty() {
    isDirty = false;
    settingsDirty = false;
    const dot = $("studio-dirty-badge");
    if (dot) dot.hidden = true;
  }

  function collectSettingsFromUi() {
    const asrRaw = String($("studio-set-asr")?.value || "auto:small");
    const [asrEngine, whisperModel] = asrRaw.includes(":") ? asrRaw.split(":") : [asrRaw, "small"];
    const volPct = Math.max(0, Math.min(50, Number($("studio-set-orig-vol")?.value || 16)));
    return {
      run_mode: String($("studio-set-run-mode")?.value || "studio"),
      output_mode: String($("studio-set-output-mode")?.value || "dub_hardsub"),
      asr_engine: asrEngine || "auto",
      whisper_model: whisperModel || "small",
      target_lang: String($("studio-set-lang")?.value || "vi"),
      provider_id: String($("studio-set-provider")?.value || "gemini"),
      transcribe_model: getSelectedModelValue("studio-set-model", "studio-set-model-custom"),
      tts_provider_id: String($("studio-set-tts-provider")?.value || PIPER_TTS_PROVIDER_ID),
      tts_model: LOCAL_TTS_IDS.has(String($("studio-set-tts-provider")?.value || ""))
        ? ""
        : getSelectedModelValue("studio-set-tts-model", "studio-set-tts-custom"),
      default_voice: String($("studio-set-voice")?.value || "Puck"),
      subtitle_style: String($("studio-set-sub-style")?.value || "box"),
      subtitle_animation: String($("studio-set-sub-anim")?.value || "reveal"),
      bilingual: Boolean($("studio-set-bilingual")?.checked),
      keep_original_audio: Boolean($("studio-set-keep-bg")?.checked),
      original_volume: Number((volPct / 100).toFixed(2)),
      context_hint: String($("studio-set-context")?.value || "").trim(),
    };
  }

  function collectStoryBibleFromUi() {
    const splitLines = (id) =>
      String($(id)?.value || "")
        .split(/\r?\n/)
        .map((s) => s.trim())
        .filter(Boolean);
    return {
      source_language: String(activeProject?.story_bible?.source_language || "unknown"),
      genre: String(activeProject?.story_bible?.genre || "general"),
      synopsis: String($("studio-bible-synopsis")?.value || "").trim(),
      glossary: splitLines("studio-bible-glossary"),
      pronoun_rules: splitLines("studio-bible-pronouns"),
      style_notes: String($("studio-bible-style")?.value || "").trim(),
    };
  }

  function collectSpeakerVoicesFromUi() {
    const map = {};
    document.querySelectorAll("#studio-speaker-voices-list select[data-speaker]").forEach((sel) => {
      const spk = sel.dataset.speaker;
      if (spk && sel.value) map[spk] = sel.value;
    });
    return map;
  }

  async function applySettingsToUi(settings) {
    const s = settings || {};
    if ($("studio-set-run-mode")) $("studio-set-run-mode").value = s.run_mode || "studio";
    if ($("studio-set-output-mode")) $("studio-set-output-mode").value = s.output_mode || "dub_hardsub";
    const asrVal = `${s.asr_engine || "auto"}:${s.whisper_model || "small"}`;
    const asrSel = $("studio-set-asr");
    if (asrSel) {
      if ([...asrSel.options].some((o) => o.value === asrVal)) {
        asrSel.value = asrVal;
      } else if (s.asr_engine === "cloud") {
        asrSel.value = "cloud:small";
      } else {
        asrSel.value = "auto:small";
      }
    }
    if ($("studio-set-lang")) $("studio-set-lang").value = s.target_lang || "vi";
    if ($("studio-set-voice")) $("studio-set-voice").value = s.default_voice || "Puck";
    if ($("studio-set-sub-style")) $("studio-set-sub-style").value = s.subtitle_style || "box";
    if ($("studio-set-sub-anim")) $("studio-set-sub-anim").value = s.subtitle_animation || "reveal";
    if ($("studio-set-bilingual")) $("studio-set-bilingual").checked = Boolean(s.bilingual);
    if ($("studio-set-keep-bg")) $("studio-set-keep-bg").checked = s.keep_original_audio !== false;
    const volPct = Math.round(Math.max(0, Math.min(0.5, Number(s.original_volume ?? 0.16))) * 100);
    if ($("studio-set-orig-vol")) $("studio-set-orig-vol").value = String(volPct);
    if ($("studio-set-orig-vol-val")) $("studio-set-orig-vol-val").textContent = `${volPct}%`;
    if ($("studio-set-context")) $("studio-set-context").value = s.context_hint || "";

    renderProviderSelects(s.provider_id, s.tts_provider_id);
    await syncSettingsModels(s.transcribe_model || "", s.tts_model || "");
    updateLiveSubtitleStyleClass();
  }

  function applyStoryBibleToUi(bible) {
    const b = bible || {};
    if ($("studio-bible-synopsis")) $("studio-bible-synopsis").value = b.synopsis || "";
    if ($("studio-bible-glossary")) $("studio-bible-glossary").value = Array.isArray(b.glossary) ? b.glossary.join("\n") : "";
    if ($("studio-bible-pronouns")) $("studio-bible-pronouns").value = Array.isArray(b.pronoun_rules) ? b.pronoun_rules.join("\n") : "";
    if ($("studio-bible-style")) $("studio-bible-style").value = b.style_notes || "";
  }

  function renderSpeakerVoicesEditor() {
    const host = $("studio-speaker-voices-list");
    const filterSel = $("studio-seg-speaker-filter");
    if (!host) return;

    const speakers = [...new Set(activeSegments.map((s) => String(s.speaker || "S1")))].sort();
    if (!speakers.length) speakers.push("S1");

    const curMap = activeProject?.speaker_voices || {};
    const defVoice = $("studio-set-voice")?.value || activeProject?.settings?.default_voice || "Puck";

    host.replaceChildren();
    speakers.forEach((spk) => {
      const row = document.createElement("div");
      row.className = "studio-speaker-row";

      const badge = document.createElement("span");
      badge.className = "studio-speaker-tag";
      badge.textContent = spk;

      const sel = document.createElement("select");
      sel.className = "ui-select studio-speaker-select";
      sel.dataset.speaker = spk;
      VOICE_OPTIONS.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = v.label;
        sel.appendChild(opt);
      });
      const assigned = curMap[spk] && curMap[spk] !== "auto" ? curMap[spk] : (defVoice === "auto" ? "Puck" : defVoice);
      sel.value = [...sel.options].some((o) => o.value === assigned) ? assigned : "Puck";
      sel.addEventListener("change", () => {
        if (activeProject) {
          activeProject.speaker_voices = { ...(activeProject.speaker_voices || {}), [spk]: sel.value };
        }
        markDirty(false);
      });

      const sampleBtn = document.createElement("button");
      sampleBtn.type = "button";
      sampleBtn.className = "ui-btn ui-btn-ghost ui-btn-compact studio-preview-btn";
      sampleBtn.title = `Nghe thử giọng của ${spk}`;
      sampleBtn.textContent = "▶ Thử";
      sampleBtn.addEventListener("click", () => {
        const sampleLine = `Xin chào, đây là giọng lồng tiếng thử nghiệm cho nhân vật ${spk}.`;
        playPreviewTts(sampleBtn, sampleLine, sel.value);
      });

      row.append(badge, sel, sampleBtn);
      host.appendChild(row);
    });

    if (filterSel) {
      const prev = filterSel.value;
      filterSel.replaceChildren();
      const allOpt = document.createElement("option");
      allOpt.value = "";
      allOpt.textContent = "Tất cả nhân vật";
      filterSel.appendChild(allOpt);
      speakers.forEach((spk) => {
        const opt = document.createElement("option");
        opt.value = spk;
        opt.textContent = spk;
        filterSel.appendChild(opt);
      });
      filterSel.value = speakers.includes(prev) ? prev : "";
    }
  }

  function renderProjectList() {
    const listEl = $("studio-project-list");
    const emptyEl = $("studio-projects-empty");
    const countEl = $("studio-project-count");
    if (!listEl) return;

    if (countEl) countEl.textContent = String(projects.length);
    if (emptyEl) emptyEl.hidden = projects.length > 0;
    listEl.replaceChildren();

    projects.forEach((proj) => {
      const card = document.createElement("div");
      card.className = `studio-project-card${proj.project_id === activeProjectId ? " is-selected" : ""}${proj.active ? " is-active" : ""}`;
      card.setAttribute("role", "listitem");
      card.tabIndex = 0;

      const thumbWrap = document.createElement("div");
      thumbWrap.className = "studio-proj-thumb-wrap";
      if (proj.thumbnail_url) {
        const img = document.createElement("img");
        img.src = proj.thumbnail_url;
        img.alt = "";
        img.loading = "lazy";
        thumbWrap.appendChild(img);
      } else {
        const fb = document.createElement("span");
        fb.className = "studio-proj-thumb-fallback";
        fb.textContent = proj.source_type === "url" ? "URL" : "MP4";
        thumbWrap.appendChild(fb);
      }
      if (proj.duration_text) {
        const dur = document.createElement("span");
        dur.className = "studio-proj-dur";
        dur.textContent = proj.duration_text;
        thumbWrap.appendChild(dur);
      }

      const info = document.createElement("div");
      info.className = "studio-proj-card-body";

      const title = document.createElement("strong");
      title.className = "studio-proj-card-title";
      title.textContent = proj.title || "Dự án Video";
      title.title = proj.title || "";

      const meta = document.createElement("div");
      meta.className = "studio-proj-card-meta";
      const st = STATUS_META[proj.status] || { label: proj.status, cls: "is-idle" };
      const pill = document.createElement("span");
      pill.className = `studio-status-pill ${st.cls}`;
      pill.textContent = st.label;

      const sub = document.createElement("span");
      sub.className = "studio-muted-text";
      const parts = [];
      if (proj.segment_count) parts.push(`${proj.segment_count} câu`);
      if (proj.file_size_text) parts.push(proj.file_size_text);
      if (proj.active && proj.percent > 0) parts.push(`${Math.round(proj.percent)}%`);
      sub.textContent = parts.join(" · ") || "Mới tạo";

      meta.append(pill, sub);
      info.append(title, meta);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "studio-proj-card-del";
      delBtn.title = "Xóa dự án";
      delBtn.setAttribute("aria-label", `Xóa dự án ${proj.title}`);
      delBtn.textContent = "×";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteProject(proj.project_id, proj.title);
      });

      card.append(thumbWrap, info, delBtn);
      const openHandler = () => selectProject(proj.project_id);
      card.addEventListener("click", openHandler);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openHandler();
        }
      });
      listEl.appendChild(card);
    });
  }

  function renderQueueStatus() {
    const badge = $("studio-queue-badge");
    const startBtn = $("studio-queue-start-btn");
    const cancelBtn = $("studio-queue-cancel-btn");
    const running = Boolean(queueStatus?.running);

    if (startBtn) startBtn.hidden = running;
    if (cancelBtn) cancelBtn.hidden = !running;
    if (badge) {
      if (running) {
        const total = (queueStatus.project_ids || []).length;
        const done = (queueStatus.completed_ids || []).length;
        const fail = (queueStatus.failed_ids || []).length;
        badge.textContent = `Hàng đợi đang chạy: ${done + fail}/${total}${fail ? ` (${fail} lỗi)` : ""}`;
        badge.hidden = false;
      } else {
        badge.hidden = true;
      }
    }
  }

  function setStepState(id, state, subLabel) {
    const el = $(id);
    if (!el) return;
    el.classList.remove("is-done", "is-active", "is-pending");
    el.classList.add(state);
    const small = el.querySelector("small");
    if (small) small.textContent = subLabel;
  }

  function renderWorkbenchHeaderAndProgress(proj) {
    const wb = $("studio-workbench");
    if (!wb) return;
    if (!proj) {
      wb.hidden = true;
      return;
    }
    wb.hidden = false;

    const titleInput = $("studio-proj-title");
    if (titleInput && document.activeElement !== titleInput && !settingsDirty) {
      titleInput.value = proj.title || "";
    }

    const st = STATUS_META[proj.status] || { label: proj.status, cls: "is-idle" };
    const badge = $("studio-proj-status-badge");
    if (badge) {
      badge.textContent = st.label;
      badge.className = `studio-status-pill ${st.cls}`;
    }

    const metaParts = [];
    if (proj.duration_text) metaParts.push(proj.duration_text);
    if (proj.width && proj.height) metaParts.push(`${proj.width}×${proj.height}`);
    if (proj.file_size_text) metaParts.push(proj.file_size_text);
    metaParts.push(`${activeSegments.length || proj.segment_count || 0} câu`);
    if ($("studio-proj-meta")) $("studio-proj-meta").textContent = metaParts.join(" · ");

    const busy = Boolean(proj.active);
    const prepBtn = $("studio-prepare-btn");
    const reBtn = $("studio-retranslate-btn");
    const cancelBtn = $("studio-cancel-btn");
    const renderBtn = $("studio-render-btn");
    const delBtn = $("studio-delete-btn");

    const ck = proj.checkpoints || {};
    if (prepBtn) {
      prepBtn.disabled = busy;
      if (proj.status === "paused" || (ck.asr_done && !ck.translation_done)) {
        prepBtn.textContent = "Tiếp tục (Checkpoint)";
      } else if (ck.translation_done) {
        prepBtn.textContent = "Kiểm tra & Chuẩn bị lại";
      } else {
        prepBtn.textContent = "Bắt đầu / Tiếp tục";
      }
    }
    if (reBtn) {
      reBtn.disabled = busy || !(activeSegments.length || proj.segment_count);
    }
    if (cancelBtn) cancelBtn.hidden = !busy;
    if (renderBtn) {
      renderBtn.disabled = busy || !(activeSegments.length || proj.segment_count);
    }
    if (delBtn) delBtn.disabled = busy;

    // Checkpoint stepper
    setStepState(
      "studio-step-video",
      ck.video_ready ? "is-done" : (proj.stage === "downloading" ? "is-active" : "is-pending"),
      ck.video_ready ? "Sẵn sàng" : (proj.stage === "downloading" ? "Đang tải" : "Chờ"),
    );

    const asrDone = Boolean(ck.asr_done);
    const asrTot = Number(ck.asr_total_chunks || 0);
    const asrComp = Number(ck.asr_completed_chunks || 0);
    setStepState(
      "studio-step-asr",
      asrDone ? "is-done" : (proj.stage === "asr" ? "is-active" : "is-pending"),
      asrDone ? "Xong" : (asrTot > 1 ? `${asrComp}/${asrTot} đoạn` : (proj.stage === "asr" ? "Đang nghe" : "Chờ")),
    );

    setStepState(
      "studio-step-bible",
      ck.bible_done || ck.translation_done ? "is-done" : (proj.stage === "story_bible" ? "is-active" : "is-pending"),
      ck.bible_done ? "Đã lập" : (ck.translation_done ? "Gộp 1 lô" : (proj.stage === "story_bible" ? "Đang phân tích" : "Chờ")),
    );

    const trDone = Boolean(ck.translation_done);
    const trBatches = Number(ck.translated_batches || 0);
    const totBatches = Number(ck.total_batches || 0);
    setStepState(
      "studio-step-translate",
      trDone ? "is-done" : (proj.stage === "translating" ? "is-active" : "is-pending"),
      trDone ? "Xong" : (totBatches > 0 ? `${trBatches}/${totBatches} lô` : (proj.stage === "translating" ? "Đang dịch" : "Chờ")),
    );

    setStepState(
      "studio-step-render",
      ck.render_done ? "is-done" : (proj.status === "running_render" ? "is-active" : "is-pending"),
      ck.render_done ? "Đã xuất" : (proj.status === "running_render" ? "Đang render" : "Chờ"),
    );

    // Progress bar & detail
    const pct = Math.max(0, Math.min(100, Number(proj.percent || 0)));
    if ($("studio-progress-fill")) $("studio-progress-fill").style.width = `${pct}%`;
    if ($("studio-progress-pct")) $("studio-progress-pct").textContent = `${Math.round(pct)}%`;
    if ($("studio-progress-detail")) $("studio-progress-detail").textContent = proj.stage_detail || "";

    const errEl = $("studio-error-banner");
    if (errEl) {
      if (proj.error) {
        errEl.textContent = proj.error;
        errEl.hidden = false;
      } else {
        errEl.hidden = true;
      }
    }

    // Export downloads
    const ex = proj.exports || {};
    const bindLink = (id, url, fname) => {
      const a = $(id);
      if (!a) return false;
      if (url) {
        a.href = url;
        if (fname) a.download = fname;
        a.hidden = false;
        return true;
      }
      a.hidden = true;
      return false;
    };
    const anyExport = [
      bindLink("studio-dl-video", ex.video_url, ex.video_filename),
      bindLink("studio-dl-audio", ex.audio_url, ex.audio_filename),
      bindLink("studio-dl-srt", ex.srt_url, ex.srt_filename),
      bindLink("studio-dl-ass", ex.ass_url, ex.ass_filename),
    ].some(Boolean);
    if ($("studio-exports-bar")) $("studio-exports-bar").hidden = !anyExport;

    // Video source
    const videoEl = $("studio-video-player");
    const placeholder = $("studio-video-placeholder");
    if (videoEl && placeholder) {
      if (proj.has_source_video && proj.source_video_url) {
        placeholder.hidden = true;
        videoEl.hidden = false;
        if (activeSourceVideoUrl !== proj.source_video_url) {
          activeSourceVideoUrl = proj.source_video_url;
          videoEl.src = proj.source_video_url;
        }
      } else {
        activeSourceVideoUrl = null;
        videoEl.removeAttribute("src");
        videoEl.hidden = true;
        placeholder.hidden = false;
      }
    }
  }

  function getFilteredSegments() {
    const q = String($("studio-seg-search")?.value || "").trim().toLowerCase();
    const spk = String($("studio-seg-speaker-filter")?.value || "").trim();
    const result = [];
    for (let i = 0; i < activeSegments.length; i += 1) {
      const seg = activeSegments[i];
      if (spk && String(seg.speaker || "S1") !== spk) continue;
      if (q) {
        const hay = `${seg.original_text || ""} ${seg.translated_text || ""} ${seg.speaker || ""}`.toLowerCase();
        if (!hay.includes(q)) continue;
      }
      result.push({ seg, index: i });
    }
    return result;
  }

  function computeCps(seg) {
    const dur = Math.max(0.25, Number(seg.end_sec || 0) - Number(seg.start_sec || 0));
    const len = String(seg.translated_text || seg.original_text || "").trim().length;
    return Math.round(len / dur);
  }

  function cpsBadgeMeta(cps) {
    if (cps <= 18) return { cls: "is-ok", title: "Tốc độ đọc thoải mái" };
    if (cps <= 24) return { cls: "is-warn", title: "Hơi nhanh — có thể rút gọn nhẹ" };
    return { cls: "is-high", title: "Dài so với thời lượng — nên rút gọn để giọng đọc tự nhiên hơn" };
  }

  function resolveVoiceForSegment(seg) {
    const ov = String(seg.voice_override || "").trim();
    if (ov) return ov;
    const spkMap = collectSpeakerVoicesFromUi();
    const spk = String(seg.speaker || "S1");
    if (spkMap[spk]) return spkMap[spk];
    const def = String($("studio-set-voice")?.value || "Puck");
    return def === "auto" ? (seg.gender === "male" ? "Puck" : "Kore") : def;
  }

  function stopPreviewAudio() {
    if (previewAudio) {
      try {
        previewAudio.pause();
        previewAudio.src = "";
      } catch (_) {}
    }
    if (previewBlobUrl) {
      URL.revokeObjectURL(previewBlobUrl);
      previewBlobUrl = null;
    }
    if (previewActiveBtn) {
      previewActiveBtn.classList.remove("is-playing");
      previewActiveBtn.disabled = false;
      previewActiveBtn = null;
    }
  }

  async function playPreviewTts(btn, text, voiceId) {
    if (!activeProjectId) return;
    const clean = String(text || "").trim();
    if (!clean) {
      toast("Câu thoại đang trống.", "warning");
      return;
    }
    const wasSame = previewActiveBtn === btn;
    stopPreviewAudio();
    if (wasSame) return;

    previewActiveBtn = btn;
    btn.classList.add("is-playing");
    btn.disabled = true;

    try {
      const settings = collectSettingsFromUi();
      const resp = await fetch(`/api/studio/projects/${encodeURIComponent(activeProjectId)}/preview-tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: clean.slice(0, 500),
          voice: voiceId || "Puck",
          tts_provider_id: settings.tts_provider_id,
          tts_model: settings.tts_model,
          target_lang: settings.target_lang,
        }),
      });
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData?.detail || `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      if (previewActiveBtn !== btn) return; // Canceled by another click
      previewBlobUrl = URL.createObjectURL(blob);
      previewAudio = new Audio(previewBlobUrl);
      btn.disabled = false;
      previewAudio.addEventListener("ended", () => stopPreviewAudio(), { once: true });
      previewAudio.addEventListener("error", () => stopPreviewAudio(), { once: true });
      await previewAudio.play();
    } catch (err) {
      stopPreviewAudio();
      toast(`Không nghe thử được: ${err.message || err}`, "error");
    }
  }

  function seekVideoTo(sec, autoPlay = true) {
    const videoEl = $("studio-video-player");
    if (!videoEl || videoEl.hidden) return;
    try {
      videoEl.currentTime = Math.max(0, Number(sec) || 0);
      if (autoPlay) videoEl.play().catch(() => {});
    } catch (_) {}
  }

  function renderSegmentsPage() {
    const listEl = $("studio-segments-list");
    const emptyEl = $("studio-segments-empty");
    const infoEl = $("studio-seg-count-info");
    const pageLabel = $("studio-page-label");
    const prevBtn = $("studio-page-prev");
    const nextBtn = $("studio-page-next");
    if (!listEl) return;

    const filtered = getFilteredSegments();
    const total = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    currentPage = Math.max(1, Math.min(currentPage, totalPages));

    if (emptyEl) emptyEl.hidden = activeSegments.length > 0;
    if (prevBtn) prevBtn.disabled = currentPage <= 1;
    if (nextBtn) nextBtn.disabled = currentPage >= totalPages;
    if (pageLabel) pageLabel.textContent = `Trang ${currentPage}/${totalPages}`;

    const startIdx = (currentPage - 1) * PAGE_SIZE;
    const slice = filtered.slice(startIdx, startIdx + PAGE_SIZE);
    if (infoEl) {
      if (!activeSegments.length) {
        infoEl.textContent = "Chưa có câu phụ đề nào.";
      } else if (!total) {
        infoEl.textContent = `Không tìm thấy câu nào khớp bộ lọc (tổng ${activeSegments.length} câu).`;
      } else {
        infoEl.textContent = `Hiển thị ${startIdx + 1}–${startIdx + slice.length} / ${total} câu${total !== activeSegments.length ? ` (tổng ${activeSegments.length})` : ""}`;
      }
    }

    const frag = document.createDocumentFragment();
    slice.forEach(({ seg, index }) => {
      const row = document.createElement("div");
      row.className = `studio-seg-row${seg.id === currentPlayingSegId ? " is-playing" : ""}`;
      row.dataset.segId = String(seg.id);

      // Top metadata row: index, timestamps, speaker, voice override, CPS, actions
      const head = document.createElement("div");
      head.className = "studio-seg-head";

      const numBtn = document.createElement("button");
      numBtn.type = "button";
      numBtn.className = "studio-seg-num";
      numBtn.title = "Bấm để tua video tới câu này";
      numBtn.textContent = `#${index + 1}`;
      numBtn.addEventListener("click", () => seekVideoTo(seg.start_sec, true));

      const startIn = document.createElement("input");
      startIn.type = "text";
      startIn.className = "ui-input studio-time-input";
      startIn.value = formatTimecode(seg.start_sec);
      startIn.ariaLabel = "Thời gian bắt đầu";
      startIn.addEventListener("change", () => {
        const val = parseTimecode(startIn.value, seg.start_sec);
        seg.start_sec = Number(val.toFixed(3));
        if (seg.end_sec <= seg.start_sec + 0.15) {
          seg.end_sec = Number((seg.start_sec + 0.5).toFixed(3));
          endIn.value = formatTimecode(seg.end_sec);
        }
        startIn.value = formatTimecode(seg.start_sec);
        updateCpsBadge();
        markDirty(true);
      });

      const arrow = document.createElement("span");
      arrow.className = "studio-time-arrow";
      arrow.textContent = "→";

      const endIn = document.createElement("input");
      endIn.type = "text";
      endIn.className = "ui-input studio-time-input";
      endIn.value = formatTimecode(seg.end_sec);
      endIn.ariaLabel = "Thời gian kết thúc";
      endIn.addEventListener("change", () => {
        const val = Math.max(seg.start_sec + 0.15, parseTimecode(endIn.value, seg.end_sec));
        seg.end_sec = Number(val.toFixed(3));
        endIn.value = formatTimecode(seg.end_sec);
        updateCpsBadge();
        markDirty(true);
      });

      const spkIn = document.createElement("input");
      spkIn.type = "text";
      spkIn.className = "ui-input studio-spk-input";
      spkIn.maxLength = 16;
      spkIn.value = seg.speaker || "S1";
      spkIn.ariaLabel = "Nhân vật";
      spkIn.title = "Mã nhân vật (VD: S1, S2, NamChinh)";
      spkIn.addEventListener("change", () => {
        seg.speaker = spkIn.value.replace(/[^A-Za-z0-9_-]/g, "").slice(0, 16) || "S1";
        spkIn.value = seg.speaker;
        renderSpeakerVoicesEditor();
        markDirty(true);
      });

      const voiceSel = document.createElement("select");
      voiceSel.className = "ui-select studio-voice-override";
      voiceSel.ariaLabel = "Giọng riêng cho câu này";
      const defOpt = document.createElement("option");
      defOpt.value = "";
      defOpt.textContent = "Giọng nhân vật";
      voiceSel.appendChild(defOpt);
      VOICE_OPTIONS.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = v.label.split(" (")[0];
        voiceSel.appendChild(opt);
      });
      voiceSel.value = seg.voice_override || "";
      voiceSel.addEventListener("change", () => {
        seg.voice_override = voiceSel.value;
        markDirty(true);
      });

      const cpsBadge = document.createElement("span");
      const updateCpsBadge = () => {
        const cps = computeCps(seg);
        const m = cpsBadgeMeta(cps);
        cpsBadge.className = `studio-cps-badge ${m.cls}`;
        cpsBadge.textContent = `${cps} kt/s`;
        cpsBadge.title = m.title;
      };
      updateCpsBadge();

      const actWrap = document.createElement("div");
      actWrap.className = "studio-seg-row-actions";

      const prevTtsBtn = document.createElement("button");
      prevTtsBtn.type = "button";
      prevTtsBtn.className = "ui-btn ui-btn-compact studio-preview-btn";
      prevTtsBtn.title = "Nghe thử giọng câu này (~50ms Piper ONNX)";
      prevTtsBtn.textContent = "▶ Nghe thử";
      prevTtsBtn.addEventListener("click", () => {
        const txt = seg.translated_text || seg.original_text || "";
        playPreviewTts(prevTtsBtn, txt, resolveVoiceForSegment(seg));
      });

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "ui-btn ui-btn-ghost ui-btn-compact studio-seg-del-btn";
      delBtn.title = "Xóa câu này";
      delBtn.textContent = "×";
      delBtn.addEventListener("click", () => {
        activeSegments.splice(index, 1);
        activeSegments.forEach((s, idx) => { s.id = idx; });
        markDirty(true);
        renderSegmentsPage();
        renderSpeakerVoicesEditor();
        updateLiveSubtitleForCurrentTime();
      });

      actWrap.append(prevTtsBtn, delBtn);
      head.append(numBtn, startIn, arrow, endIn, spkIn, voiceSel, cpsBadge, actWrap);

      // Text editors (original + translated)
      const body = document.createElement("div");
      body.className = "studio-seg-texts";

      const origIn = document.createElement("input");
      origIn.type = "text";
      origIn.className = "ui-input studio-orig-input";
      origIn.value = seg.original_text || "";
      origIn.placeholder = "Lời gốc…";
      origIn.ariaLabel = `Lời gốc câu ${index + 1}`;
      origIn.addEventListener("input", () => {
        seg.original_text = origIn.value;
        markDirty(true);
        updateLiveSubtitleForCurrentTime();
      });

      const trIn = document.createElement("textarea");
      trIn.className = "ui-textarea studio-tr-input";
      trIn.rows = 2;
      trIn.value = seg.translated_text || "";
      trIn.placeholder = "Bản dịch & lời lồng tiếng…";
      trIn.ariaLabel = `Bản dịch câu ${index + 1}`;
      trIn.addEventListener("input", () => {
        seg.translated_text = trIn.value;
        updateCpsBadge();
        markDirty(true);
        updateLiveSubtitleForCurrentTime();
      });

      body.append(origIn, trIn);
      row.append(head, body);
      frag.appendChild(row);
    });

    listEl.replaceChildren(frag);
  }

  function updateLiveSubtitleStyleClass() {
    const overlay = $("studio-live-sub");
    if (!overlay) return;
    const style = String($("studio-set-sub-style")?.value || "box");
    overlay.classList.remove("style-box", "style-classic", "style-cinema");
    overlay.classList.add(`style-${style}`);
  }

  function findSegmentAtTime(t) {
    let lo = 0;
    let hi = activeSegments.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const seg = activeSegments[mid];
      if (t < seg.start_sec) {
        hi = mid - 1;
      } else if (t > seg.end_sec) {
        lo = mid + 1;
      } else {
        return seg;
      }
    }
    return null;
  }

  function updateLiveSubtitleForCurrentTime() {
    const videoEl = $("studio-video-player");
    const overlay = $("studio-live-sub");
    const timeEl = $("studio-player-time");
    if (!videoEl) return;

    const cur = Number(videoEl.currentTime || 0);
    const dur = Number(videoEl.duration || activeProject?.duration_sec || 0);
    if (timeEl) timeEl.textContent = `${formatTimecode(cur)} / ${formatTimecode(dur)}`;

    const seg = findSegmentAtTime(cur);
    const showOverlay = Boolean($("studio-toggle-overlay")?.checked);

    if (overlay) {
      if (seg && showOverlay) {
        const mainText = String(seg.translated_text || seg.original_text || "").trim();
        const origText = String(seg.original_text || "").trim();
        const bilingual = Boolean($("studio-set-bilingual")?.checked);
        overlay.replaceChildren();
        const strong = document.createElement("div");
        strong.className = "studio-live-sub-main";
        strong.textContent = mainText;
        overlay.appendChild(strong);
        if (bilingual && origText && origText.toLowerCase() !== mainText.toLowerCase()) {
          const origDiv = document.createElement("div");
          origDiv.className = "studio-live-sub-orig";
          origDiv.textContent = origText;
          overlay.appendChild(origDiv);
        }
        overlay.hidden = !mainText;
      } else {
        overlay.hidden = true;
      }
    }

    const nextId = seg ? seg.id : null;
    if (nextId !== currentPlayingSegId) {
      if (currentPlayingSegId !== null) {
        document.querySelector(`.studio-seg-row[data-seg-id="${currentPlayingSegId}"]`)?.classList.remove("is-playing");
      }
      currentPlayingSegId = nextId;
      if (nextId !== null) {
        document.querySelector(`.studio-seg-row[data-seg-id="${nextId}"]`)?.classList.add("is-playing");
      }
    }
  }

  function jumpToActiveSegment() {
    if (currentPlayingSegId === null) {
      toast("Hãy phát video tới đoạn có lời thoại để nhảy tới câu tương ứng.", "info");
      return;
    }
    const filtered = getFilteredSegments();
    const pos = filtered.findIndex((item) => item.seg.id === currentPlayingSegId);
    if (pos < 0) {
      toast("Câu đang phát đang bị ẩn bởi bộ lọc tìm kiếm.", "info");
      return;
    }
    const targetPage = Math.floor(pos / PAGE_SIZE) + 1;
    if (targetPage !== currentPage) {
      currentPage = targetPage;
      renderSegmentsPage();
    }
    const row = document.querySelector(`.studio-seg-row[data-seg-id="${currentPlayingSegId}"]`);
    row?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function selectProject(projectId, { forceReload = true } = {}) {
    if (!projectId) return;
    if (activeProjectId === projectId && !forceReload) return;
    stopPreviewAudio();
    activeProjectId = projectId;
    currentPage = 1;
    currentPlayingSegId = null;
    clearDirty();
    renderProjectList();

    try {
      const proj = await apiJson(`/api/studio/projects/${encodeURIComponent(projectId)}`);
      activeProject = proj;
      activeSegments = Array.isArray(proj.segments) ? proj.segments : [];
      wasSelectedActive = Boolean(proj.active);
      await applySettingsToUi(proj.settings);
      applyStoryBibleToUi(proj.story_bible);
      renderSpeakerVoicesEditor();
      renderWorkbenchHeaderAndProgress(proj);
      renderSegmentsPage();
      updateLiveSubtitleForCurrentTime();
      schedulePollIfNeeded();
    } catch (err) {
      toast(`Không mở được dự án: ${err.message || err}`, "error");
    }
  }

  async function saveChanges({ silent = false } = {}) {
    if (!activeProjectId || !activeProject) return true;
    if (activeProject.active) {
      toast("Dự án đang chạy nền — hãy Tạm dừng trước khi lưu chỉnh sửa.", "warning");
      return false;
    }
    const payload = {
      title: String($("studio-proj-title")?.value || activeProject.title || "").trim(),
      settings: collectSettingsFromUi(),
      speaker_voices: collectSpeakerVoicesFromUi(),
      story_bible: collectStoryBibleFromUi(),
    };
    if (isDirty) {
      payload.segments = activeSegments;
    }
    try {
      const updated = await apiJson(`/api/studio/projects/${encodeURIComponent(activeProjectId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      activeProject = updated;
      if (Array.isArray(updated.segments)) {
        activeSegments = updated.segments;
      }
      clearDirty();
      renderWorkbenchHeaderAndProgress(updated);
      renderSpeakerVoicesEditor();
      renderSegmentsPage();
      await refreshProjectList({ keepSelection: true });
      if (!silent) toast("Đã lưu dự án thành công.", "success");
      return true;
    } catch (err) {
      toast(`Lưu thất bại: ${err.message || err}`, "error");
      return false;
    }
  }

  async function uploadLocalFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const initSubInput = $("studio-init-sub-input");
    const initSubFile = files.length === 1 && initSubInput?.files?.[0] ? initSubInput.files[0] : null;
    const statusEl = $("studio-upload-status");
    const defaultSettings = {
      run_mode: String($("studio-default-run-mode")?.value || "studio"),
      provider_id: String($("studio-set-provider")?.value || ""),
      tts_provider_id: String($("studio-set-tts-provider")?.value || PIPER_TTS_PROVIDER_ID),
      default_voice: String($("studio-set-voice")?.value || "Puck"),
    };

    let lastCreated = null;
    for (let i = 0; i < files.length; i += 1) {
      const file = files[i];
      if (statusEl) statusEl.textContent = `Đang tải lên (${i + 1}/${files.length}): ${file.name}…`;
      const form = new FormData();
      form.append("video", file);
      if (initSubFile && i === 0) form.append("subtitle", initSubFile);
      form.append("settings_json", JSON.stringify(defaultSettings));
      try {
        const created = await apiJson("/api/studio/projects/upload", { method: "POST", body: form });
        lastCreated = created;
      } catch (err) {
        toast(`Tải "${file.name}" thất bại: ${err.message || err}`, "error");
      }
    }

    if (statusEl) statusEl.textContent = "";
    if ($("studio-video-input")) $("studio-video-input").value = "";
    if (initSubInput) initSubInput.value = "";
    if ($("studio-init-sub-name")) $("studio-init-sub-name").textContent = "";

    await refreshProjectList({ keepSelection: true });
    if (lastCreated?.project_id) {
      toast(`Đã nạp ${files.length} video vào Xưởng Video AI.`, "success");
      await selectProject(lastCreated.project_id, { forceReload: true });
    }
  }

  async function addProjectsFromUrls() {
    const raw = String($("studio-urls-input")?.value || "");
    const urls = raw.split(/\r?\n/).map((u) => u.trim()).filter(Boolean);
    if (!urls.length) {
      toast("Hãy dán ít nhất một liên kết video.", "warning");
      return;
    }
    const btn = $("studio-add-urls-btn");
    if (btn) btn.disabled = true;
    const preset = String($("studio-url-preset")?.value || "video_720");
    const runMode = String($("studio-default-run-mode")?.value || "studio");
    const settings = {
      run_mode: runMode,
      tts_provider_id: String($("studio-set-tts-provider")?.value || PIPER_TTS_PROVIDER_ID),
      default_voice: String($("studio-set-voice")?.value || "Puck"),
    };

    let okCount = 0;
    let lastId = null;
    const failedLines = [];
    for (const url of urls) {
      try {
        const created = await apiJson("/api/studio/projects/from-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, preset, settings }),
        });
        okCount += 1;
        lastId = created.project_id;
      } catch (err) {
        failedLines.push(`${url} — ${err.message || err}`);
      }
    }
    if (btn) btn.disabled = false;

    if (okCount > 0) {
      if ($("studio-urls-input")) $("studio-urls-input").value = "";
      toast(`Đã thêm ${okCount} video từ liên kết vào Xưởng.`, "success");
      await refreshProjectList({ keepSelection: true });
      if (lastId) await selectProject(lastId, { forceReload: true });
    }
    if (failedLines.length) {
      toast(`Không thêm được ${failedLines.length} liên kết: ${failedLines[0]}`, "error");
    }
  }

  async function attachSubtitleFile(file) {
    if (!file || !activeProjectId) return;
    const form = new FormData();
    form.append("subtitle", file);
    try {
      const updated = await apiJson(`/api/studio/projects/${encodeURIComponent(activeProjectId)}/subtitle`, {
        method: "POST",
        body: form,
      });
      activeProject = updated;
      activeSegments = Array.isArray(updated.segments) ? updated.segments : [];
      currentPage = 1;
      clearDirty();
      renderWorkbenchHeaderAndProgress(updated);
      renderSpeakerVoicesEditor();
      renderSegmentsPage();
      updateLiveSubtitleForCurrentTime();
      await refreshProjectList({ keepSelection: true });
      toast(`Đã nạp ${activeSegments.length} câu từ ${file.name}.`, "success");
    } catch (err) {
      toast(`Nạp phụ đề thất bại: ${err.message || err}`, "error");
    } finally {
      if ($("studio-attach-sub-input")) $("studio-attach-sub-input").value = "";
    }
  }

  async function triggerPrepare({ forceRetranslate = false } = {}) {
    if (!activeProjectId) return;
    if (isDirty || settingsDirty) {
      const ok = await saveChanges({ silent: true });
      if (!ok) return;
    }
    try {
      const summary = await apiJson(`/api/studio/projects/${encodeURIComponent(activeProjectId)}/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          auto_render: String($("studio-set-run-mode")?.value || "studio") === "auto",
          force_retranslate: Boolean(forceRetranslate),
        }),
      });
      activeProject = { ...activeProject, ...summary };
      wasSelectedActive = true;
      renderWorkbenchHeaderAndProgress(activeProject);
      schedulePollIfNeeded(true);
      toast(forceRetranslate ? "Đang chạy dịch lại toàn bộ câu thoại…" : "Đã khởi chạy xử lý dự án.", "info");
    } catch (err) {
      toast(`Không thể bắt đầu: ${err.message || err}`, "error");
    }
  }

  async function triggerRender() {
    if (!activeProjectId) return;
    if (isDirty || settingsDirty) {
      const ok = await saveChanges({ silent: true });
      if (!ok) return;
    }
    try {
      const summary = await apiJson(`/api/studio/projects/${encodeURIComponent(activeProjectId)}/render`, {
        method: "POST",
      });
      activeProject = { ...activeProject, ...summary };
      wasSelectedActive = true;
      renderWorkbenchHeaderAndProgress(activeProject);
      schedulePollIfNeeded(true);
      toast("Đang tổng hợp giọng đọc & xuất bản thành phẩm…", "info");
    } catch (err) {
      toast(`Không thể xuất bản: ${err.message || err}`, "error");
    }
  }

  async function triggerCancelProject() {
    if (!activeProjectId) return;
    const cancelBtn = $("studio-cancel-btn");
    if (cancelBtn) cancelBtn.disabled = true;
    try {
      const updated = await apiJson(`/api/studio/projects/${encodeURIComponent(activeProjectId)}/cancel`, {
        method: "POST",
      });
      wasSelectedActive = false;
      activeProject = updated;
      if (!isDirty && Array.isArray(updated.segments)) {
        activeSegments = updated.segments;
        renderSegmentsPage();
        renderSpeakerVoicesEditor();
      }
      renderWorkbenchHeaderAndProgress(updated);
      await refreshProjectList({ keepSelection: true });
      toast("Đã tạm dừng ngay — Checkpoint đã được lưu.", "info");
    } catch (err) {
      toast(`Không dừng được: ${err.message || err}`, "error");
    } finally {
      if (cancelBtn) cancelBtn.disabled = false;
    }
  }

  async function deleteProject(projectId, title) {
    if (!projectId) return;
    if (!window.confirm(`Xóa dự án "${title || projectId}" và toàn bộ tệp tạm của dự án này?`)) return;
    try {
      await apiJson(`/api/studio/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" });
      toast("Đã xóa dự án.", "info");
      if (activeProjectId === projectId) {
        stopPreviewAudio();
        activeProjectId = null;
        activeProject = null;
        activeSegments = [];
        activeSourceVideoUrl = null;
        clearDirty();
        renderWorkbenchHeaderAndProgress(null);
      }
      await refreshProjectList({ keepSelection: true });
    } catch (err) {
      toast(`Không xóa được: ${err.message || err}`, "error");
    }
  }

  async function startBatchQueue() {
    try {
      const st = await apiJson("/api/studio/queue/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ auto_render: true }),
      });
      queueStatus = st;
      renderQueueStatus();
      schedulePollIfNeeded(true);
      toast(`Đã khởi chạy hàng đợi ${st.project_ids?.length || 0} dự án.`, "success");
    } catch (err) {
      toast(`Không chạy được hàng đợi: ${err.message || err}`, "error");
    }
  }

  async function cancelBatchQueue() {
    try {
      const st = await apiJson("/api/studio/queue/cancel", { method: "POST" });
      queueStatus = st;
      renderQueueStatus();
      toast("Đã gửi yêu cầu dừng hàng đợi (Checkpoint được bảo toàn).", "info");
    } catch (err) {
      toast(`Không dừng được hàng đợi: ${err.message || err}`, "error");
    }
  }

  function addNewSegment() {
    if (!activeProject) return;
    const videoEl = $("studio-video-player");
    const lastEnd = activeSegments.length ? Number(activeSegments[activeSegments.length - 1].end_sec || 0) : 0;
    const startSec = videoEl && !videoEl.hidden && videoEl.currentTime > 0
      ? Number(videoEl.currentTime.toFixed(2))
      : Number((lastEnd + 0.2).toFixed(2));
    const newSeg = {
      id: activeSegments.length,
      start_sec: startSec,
      end_sec: Number((startSec + 2.5).toFixed(2)),
      speaker: "S1",
      gender: "female",
      tone: "natural and clear",
      original_text: "",
      translated_text: "",
      voice_override: "",
    };
    activeSegments.push(newSeg);
    activeSegments.sort((a, b) => a.start_sec - b.start_sec);
    activeSegments.forEach((s, i) => { s.id = i; });
    markDirty(true);

    const insertedIdx = activeSegments.indexOf(newSeg);
    currentPage = Math.max(1, Math.floor(insertedIdx / PAGE_SIZE) + 1);
    renderSegmentsPage();
    renderSpeakerVoicesEditor();
    const row = document.querySelector(`.studio-seg-row[data-seg-id="${newSeg.id}"] .studio-tr-input`);
    row?.focus();
  }

  async function refreshProjectList({ keepSelection = true } = {}) {
    try {
      const data = await apiJson("/api/studio/projects");
      projects = Array.isArray(data?.projects) ? data.projects : [];
      if (data?.queue) queueStatus = data.queue;
      renderProjectList();
      renderQueueStatus();

      if (keepSelection && activeProjectId) {
        const exists = projects.some((p) => p.project_id === activeProjectId);
        if (!exists) {
          activeProjectId = null;
          activeProject = null;
          activeSegments = [];
          renderWorkbenchHeaderAndProgress(null);
        }
      } else if (!activeProjectId && projects.length > 0) {
        await selectProject(projects[0].project_id, { forceReload: true });
      }
    } catch (_) {}
  }

  function clearPoll() {
    if (pollTimer !== null) {
      window.clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function schedulePollIfNeeded(forceImmediate = false) {
    clearPoll();
    const anyActive = Boolean(queueStatus?.running) || projects.some((p) => p.active) || Boolean(activeProject?.active);
    if (!anyActive && !forceImmediate) return;
    pollTimer = window.setTimeout(pollTick, forceImmediate ? 350 : POLL_MS);
  }

  async function pollTick() {
    pollTimer = null;
    if (document.body.dataset.landingMode !== "studio") return;

    await refreshProjectList({ keepSelection: true });

    if (activeProjectId) {
      try {
        const fresh = await apiJson(`/api/studio/projects/${encodeURIComponent(activeProjectId)}`);
        const becameIdle = wasSelectedActive && !fresh.active;
        wasSelectedActive = Boolean(fresh.active);

        if (!isDirty && (fresh.active || becameIdle || fresh.segment_count !== activeSegments.length)) {
          activeSegments = Array.isArray(fresh.segments) ? fresh.segments : [];
          renderSegmentsPage();
          renderSpeakerVoicesEditor();
        }
        if (!settingsDirty && becameIdle) {
          applyStoryBibleToUi(fresh.story_bible);
        }
        activeProject = fresh;
        renderWorkbenchHeaderAndProgress(fresh);
        updateLiveSubtitleForCurrentTime();
      } catch (_) {}
    }

    schedulePollIfNeeded(false);
  }

  function bindEvents() {
    const dropzone = $("studio-dropzone");
    const videoInput = $("studio-video-input");
    if (dropzone && videoInput) {
      dropzone.addEventListener("click", () => videoInput.click());
      videoInput.addEventListener("change", () => uploadLocalFiles(videoInput.files));
      ["dragenter", "dragover"].forEach((ev) => {
        dropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dropzone.classList.add("is-dragover");
        });
      });
      ["dragleave", "drop"].forEach((ev) => {
        dropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dropzone.classList.remove("is-dragover");
        });
      });
      dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer?.files?.length) uploadLocalFiles(e.dataTransfer.files);
      });
    }

    $("studio-init-sub-input")?.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if ($("studio-init-sub-name")) $("studio-init-sub-name").textContent = f ? f.name : "";
    });

    $("studio-add-urls-btn")?.addEventListener("click", addProjectsFromUrls);
    $("studio-refresh-btn")?.addEventListener("click", () => refreshProjectList({ keepSelection: true }));
    $("studio-queue-start-btn")?.addEventListener("click", startBatchQueue);
    $("studio-queue-cancel-btn")?.addEventListener("click", cancelBatchQueue);

    $("studio-prepare-btn")?.addEventListener("click", () => triggerPrepare({ forceRetranslate: false }));
    $("studio-retranslate-btn")?.addEventListener("click", () => {
      if (!window.confirm("Dịch lại toàn bộ lời thoại từ đầu bằng model AI hiện tại?")) return;
      triggerPrepare({ forceRetranslate: true });
    });
    $("studio-cancel-btn")?.addEventListener("click", triggerCancelProject);
    $("studio-render-btn")?.addEventListener("click", triggerRender);
    $("studio-delete-btn")?.addEventListener("click", () => deleteProject(activeProjectId, activeProject?.title));

    $("studio-open-settings-btn")?.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      document.getElementById("settings-toggle")?.click();
    });

    // Settings change tracking
    $("studio-proj-title")?.addEventListener("input", () => markDirty(false));
    const settingsIds = [
      "studio-set-run-mode",
      "studio-set-output-mode",
      "studio-set-asr",
      "studio-set-lang",
      "studio-set-model-custom",
      "studio-set-tts-custom",
      "studio-set-voice",
      "studio-set-sub-style",
      "studio-set-sub-anim",
      "studio-set-context",
      "studio-set-keep-bg",
      "studio-set-bilingual",
      "studio-bible-synopsis",
      "studio-bible-glossary",
      "studio-bible-pronouns",
      "studio-bible-style",
    ];
    settingsIds.forEach((id) => {
      const el = $(id);
      if (!el) return;
      const ev = el.tagName === "INPUT" && (el.type === "text" || el.type === "range") ? "input" : "change";
      el.addEventListener(ev, () => {
        markDirty(false);
        if (id === "studio-set-sub-style" || id === "studio-set-bilingual") {
          updateLiveSubtitleStyleClass();
          updateLiveSubtitleForCurrentTime();
        }
      });
    });

    $("studio-set-orig-vol")?.addEventListener("input", (e) => {
      if ($("studio-set-orig-vol-val")) $("studio-set-orig-vol-val").textContent = `${e.target.value}%`;
      markDirty(false);
    });

    $("studio-set-provider")?.addEventListener("change", async () => {
      markDirty(false);
      await syncSettingsModels("", null);
    });
    $("studio-set-tts-provider")?.addEventListener("change", async () => {
      markDirty(false);
      await syncSettingsModels(null, "");
    });
    $("studio-set-model")?.addEventListener("change", () => {
      markDirty(false);
      if ($("studio-set-model-custom-wrap")) {
        $("studio-set-model-custom-wrap").hidden = $("studio-set-model").value !== CUSTOM_MODEL_VALUE;
      }
    });
    $("studio-set-tts-model")?.addEventListener("change", () => {
      markDirty(false);
      if ($("studio-set-tts-custom-wrap")) {
        $("studio-set-tts-custom-wrap").hidden = $("studio-set-tts-model").value !== CUSTOM_MODEL_VALUE;
      }
    });

    $("studio-save-meta-btn")?.addEventListener("click", () => saveChanges({ silent: false }));
    $("studio-save-segs-btn")?.addEventListener("click", () => saveChanges({ silent: false }));
    $("studio-add-seg-btn")?.addEventListener("click", addNewSegment);
    $("studio-attach-sub-input")?.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f) attachSubtitleFile(f);
    });

    $("studio-seg-search")?.addEventListener("input", () => {
      currentPage = 1;
      renderSegmentsPage();
    });
    $("studio-seg-speaker-filter")?.addEventListener("change", () => {
      currentPage = 1;
      renderSegmentsPage();
    });
    $("studio-page-prev")?.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage -= 1;
        renderSegmentsPage();
      }
    });
    $("studio-page-next")?.addEventListener("click", () => {
      currentPage += 1;
      renderSegmentsPage();
    });

    const videoEl = $("studio-video-player");
    if (videoEl) {
      videoEl.addEventListener("timeupdate", updateLiveSubtitleForCurrentTime);
      videoEl.addEventListener("loadedmetadata", updateLiveSubtitleForCurrentTime);
      videoEl.addEventListener("seeked", updateLiveSubtitleForCurrentTime);
    }
    $("studio-toggle-overlay")?.addEventListener("change", updateLiveSubtitleForCurrentTime);
    $("studio-jump-active-btn")?.addEventListener("click", jumpToActiveSegment);

    document.addEventListener("ai-providers-updated", () => {
      renderProviderSelects(activeProject?.settings?.provider_id, activeProject?.settings?.tts_provider_id);
      syncSettingsModels(null, null);
    });

    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        if (document.body.dataset.landingMode === "studio" && activeProjectId && (isDirty || settingsDirty)) {
          e.preventDefault();
          saveChanges({ silent: false });
        }
      }
    });
  }

  async function openStudioView() {
    if (!initialized) {
      initialized = true;
      bindEvents();
      await loadProvidersAndCapabilities();
    } else {
      renderProviderSelects(activeProject?.settings?.provider_id, activeProject?.settings?.tts_provider_id);
    }
    await refreshProjectList({ keepSelection: true });
    schedulePollIfNeeded(false);
  }

  window.openStudioView = openStudioView;
})();
