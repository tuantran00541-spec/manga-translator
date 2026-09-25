// A.I mode: paste a chapter URL, the server runs every step and hands back a ZIP.
(() => {
  const POLL_MS = 1500;
  const JOB_KEY = "manga_ai_mode_job";
  const STATUS_TEXT = {
    pending: "Chờ", running: "Đang chạy", done: "Xong", failed: "Lỗi", cancelled: "Đã hủy",
  };
  const CLOUD = "manga-cloud";
  let pollTimer = null;
  let currentJob = null;
  let account = null;

  const $ = (id) => document.getElementById(id);

  function formatSeconds(value) {
    const seconds = Math.max(0, Math.round(Number(value) || 0));
    return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}p${String(seconds % 60).padStart(2, "0")}s`;
  }

  async function requestJson(url, options) {
    const response = await fetch(url, options);
    const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({})));
    const data = await parse(response);
    if (!response.ok) {
      const message = window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`;
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function storage(action, key, value) {
    try {
      if (action === "get") return localStorage.getItem(key);
      if (action === "set") localStorage.setItem(key, value);
      if (action === "remove") localStorage.removeItem(key);
    } catch (_) {}
    return null;
  }

  function syncProviderFields() {
    const provider = $("ai-mode-provider");
    const model = $("ai-mode-model");
    const budgetField = $("ai-mode-budget-field");
    if (!provider || !model) return;
    model.value = storage("get", "manga_translation_vision_model_" + provider.value) || "";
    const cloud = provider.value === CLOUD;
    const modelField = model.closest("label");
    if (modelField) modelField.hidden = cloud;
    // Only DeepSeek reports token cost, so only there a budget can be enforced.
    if (budgetField) budgetField.hidden = cloud || provider.value !== "deepseek";
  }

  function ensureCloudOption() {
    const provider = $("ai-mode-provider");
    if (!provider || !account?.tiers) return;
    if (![...provider.options].some((o) => o.value === CLOUD)) {
      provider.prepend(new Option("Manga Cloud", CLOUD));
    }
    const stored = storage("get", "manga_translation_provider");
    if (!stored || stored === CLOUD || !account.features?.includes("byok")) provider.value = CLOUD;
  }

  function renderAccount() {
    const bar = $("ai-mode-plan");
    if (!bar) return;
    bar.hidden = !account?.tiers;
    if (!account?.tiers) return;
    const signedIn = account.signed_in && !account.invalid_token;
    const quota = account.quota;
    let text = "Chưa đăng nhập Manga Cloud — dán token để dùng A.I mode.";
    if (account.invalid_token) text = "Token Manga Cloud không hợp lệ — đăng nhập lại.";
    else if (account.offline) text = "Không kết nối được Manga Cloud — tạm dùng tính năng gói Free.";
    else if (signedIn && quota) {
      text = `Gói ${account.plan_label || account.plan} · còn ${quota.remaining}/${quota.limit} chương A.I mode tháng ${quota.period}`;
    }
    $("ai-mode-plan-text").textContent = text;
    $("ai-mode-token-form").hidden = signedIn;
    $("ai-mode-signout").hidden = !signedIn;
  }

  async function loadAccount() {
    try {
      account = await requestJson("/api/account");
    } catch (_) {
      account = null;
    }
    ensureCloudOption();
    renderAccount();
    syncProviderFields();
  }

  async function saveToken(event) {
    event.preventDefault();
    const token = $("ai-mode-token").value.trim();
    if (!token) return;
    try {
      account = await requestJson("/api/account/token", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }),
      });
      $("ai-mode-token").value = "";
    } catch (err) {
      window.showToast?.("Không lưu được token: " + err.message, "error");
    }
    ensureCloudOption();
    renderAccount();
    syncProviderFields();
  }

  function setRunning(running) {
    const form = $("ai-mode-form");
    form?.querySelectorAll("input, select, button").forEach((el) => { el.disabled = running; });
    const cancel = $("ai-mode-cancel");
    if (cancel) cancel.hidden = !running;
  }

  function reportLines(job) {
    const r = job.report || {};
    const lines = [];
    if (r.credit_pages?.length) lines.push(`Bỏ qua ${r.credit_pages.length} lát credit: lát ${r.credit_pages.map((i) => i + 1).join(", ")}`);
    if (r.credit_rejected?.length) lines.push(`AI coi ${r.credit_rejected.length} lát là credit — quá nhiều nên không bỏ lát nào`);
    if (r.logo_regions) lines.push(`Giữ nguyên ${r.logo_regions} vùng logo`);
    if (r.repainted_regions) lines.push(`Repaint ${r.repainted_regions} vùng AI thấy còn sót ở ${r.repaint_pages?.length || 0} lát`);
    if (r.source_lang) lines.push(`Ngôn ngữ gốc: ${window.SOURCE_LANG_LABELS?.[r.source_lang] || r.source_lang}`);
    if (r.translated || r.unreadable) lines.push(`Dịch ${r.translated || 0} vùng chữ${r.unreadable ? `, ${r.unreadable} vùng AI không đọc được` : ""}`);
    if (r.editorial_blockers) {
      const pages = [...new Set((r.blocker_samples || []).map((b) => b.page))].slice(0, 8);
      lines.push(`${r.editorial_blockers} chỗ nên xem lại bằng mắt${pages.length ? ` (ví dụ lát ${pages.join(", ")})` : ""}`);
    }
    const errors = [
      ["quét credit/logo", r.scan_errors], ["kiểm tra/repaint", r.qc_errors],
      ["dịch", r.translate_errors], ["render", r.render_errors],
    ].filter(([, list]) => list?.length);
    errors.forEach(([label, list]) => {
      const first = typeof list[0] === "string" ? list[0] : list[0]?.error || "";
      lines.push(`Lỗi khi ${label} (${list.length}): ${first}`);
    });
    return lines;
  }

  function render(job) {
    currentJob = job;
    const status = $("ai-mode-status");
    if (!status || !job) return;
    status.hidden = false;
    const stages = $("ai-mode-stages");
    stages.replaceChildren(...job.stages.map((stage) => {
      const item = document.createElement("li");
      item.className = `ai-mode-stage is-${stage.status}`;
      const label = document.createElement("strong");
      label.textContent = stage.label;
      const state = document.createElement("span");
      const count = stage.status === "running" && stage.total ? ` ${stage.done}/${stage.total}` : "";
      const elapsed = stage.status === "done" && stage.elapsed_s != null ? ` · ${formatSeconds(stage.elapsed_s)}` : "";
      state.textContent = (STATUS_TEXT[stage.status] || stage.status) + count + (stage.detail ? ` · ${stage.detail}` : "") + elapsed;
      item.append(label, state);
      return item;
    }));

    const running = job.status === "pending" || job.status === "running";
    const cost = job.cost_usd == null ? "" : ` · chi phí ~$${Number(job.cost_usd).toFixed(4)}`;
    const summary = $("ai-mode-summary");
    if (running) summary.textContent = (job.cancel_requested ? "Đang hủy sau bước hiện tại…" : "AI đang làm, bạn có thể để trang này mở hoặc quay lại sau.") + cost;
    else if (job.status === "completed") summary.textContent = "Xong! File zip đã sẵn sàng." + cost;
    else if (job.status === "cancelled") summary.textContent = "Đã hủy. Những gì đã làm vẫn được lưu trong chương." + cost;
    else summary.textContent = `Dừng vì lỗi: ${job.error || "không rõ"}` + cost;

    const download = $("ai-mode-download");
    download.hidden = !job.download_url;
    if (job.download_url) download.href = job.download_url;
    const open = $("ai-mode-open");
    open.hidden = running || !job.chapter_id;
    setRunning(running);

    const report = $("ai-mode-report");
    const lines = reportLines(job);
    report.hidden = running || !lines.length;
    $("ai-mode-report-list").replaceChildren(...lines.map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    }));
  }

  async function poll(jobId) {
    window.clearTimeout(pollTimer);
    try {
      const job = await requestJson(`/api/ai_mode/jobs/${encodeURIComponent(jobId)}`);
      render(job);
      if (job.status === "pending" || job.status === "running") {
        pollTimer = window.setTimeout(() => poll(jobId), POLL_MS);
      } else if (job.status === "completed") {
        window.showToast?.("A.I mode xong, tải file zip ở khung A.I mode.", "success");
      }
    } catch (err) {
      if (err.status === 404) {
        storage("remove", JOB_KEY);
        setRunning(false);
        return;
      }
      pollTimer = window.setTimeout(() => poll(jobId), POLL_MS * 2);
    }
  }

  async function start(event) {
    event.preventDefault();
    const url = $("ai-mode-url")?.value.trim();
    if (!url) return;
    const provider = $("ai-mode-provider").value;
    const payload = {
      url,
      provider,
      model: $("ai-mode-model").value.trim() || null,
      target_lang: $("ai-mode-target").value,
      budget_usd: Number($("ai-mode-budget").value || 0.3),
      workers: typeof window.getWorkersSetting === "function" ? window.getWorkersSetting() : 2,
    };
    setRunning(true);
    try {
      const job = await requestJson("/api/ai_mode/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      storage("set", JOB_KEY, job.job_id);
      render(job);
      poll(job.job_id);
      if (provider === CLOUD) loadAccount();
    } catch (err) {
      setRunning(false);
      window.showToast?.("Không chạy được A.I mode: " + err.message, "error");
    }
  }

  async function restore() {
    try {
      const active = await requestJson("/api/ai_mode/active");
      const jobId = active.job?.job_id || storage("get", JOB_KEY);
      if (jobId) poll(jobId);
    } catch (_) {}
  }

  function init() {
    const form = $("ai-mode-form");
    if (!form) return;
    const provider = $("ai-mode-provider");
    const stored = storage("get", "manga_translation_provider");
    if (stored && [...provider.options].some((o) => o.value === stored)) provider.value = stored;
    provider.addEventListener("change", () => {
      storage("set", "manga_translation_provider", provider.value);
      syncProviderFields();
    });
    provider.addEventListener("ai-providers-updated", () => {
      ensureCloudOption();
      syncProviderFields();
    });
    $("ai-mode-token-form").addEventListener("submit", saveToken);
    $("ai-mode-signout").addEventListener("click", async () => {
      try {
        account = await requestJson("/api/account/token", { method: "DELETE" });
      } catch (err) {
        window.showToast?.("Không đăng xuất được: " + err.message, "error");
      }
      renderAccount();
    });
    $("ai-mode-model").addEventListener("change", (event) => {
      storage("set", "manga_translation_vision_model_" + provider.value, event.target.value.trim());
    });
    form.addEventListener("submit", start);
    $("ai-mode-cancel").addEventListener("click", async () => {
      if (!currentJob) return;
      try {
        render(await requestJson(`/api/ai_mode/jobs/${encodeURIComponent(currentJob.job_id)}/cancel`, { method: "POST" }));
      } catch (err) {
        window.showToast?.("Không hủy được: " + err.message, "error");
      }
    });
    $("ai-mode-open").addEventListener("click", () => {
      if (currentJob?.chapter_id) window.resumeChapter?.(currentJob.chapter_id);
    });
    syncProviderFields();
    window.syncAIProviderSelects?.();
    loadAccount();
    restore();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
