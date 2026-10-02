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
    // The Jev judge runs behind the Manga Cloud gateway.
    const polishField = $("ai-mode-polish-field");
    if (polishField) polishField.hidden = !cloud;
    // Manga Cloud picks its own reading model; the limit is for the user's own key.
    const imagesField = $("ai-mode-images-field");
    if (imagesField) imagesField.hidden = cloud;
  }

  function ensureCloudOption() {
    const provider = $("ai-mode-provider");
    if (!provider || !account?.tiers) return;
    if (![...provider.options].some((o) => o.value === CLOUD)) {
      provider.prepend(new Option("Manga Cloud", CLOUD));
    }
    const stored = storage("get", "manga_translation_provider");
    if (!stored || stored === CLOUD) provider.value = CLOUD;
  }

  const postJson = (url, body) => requestJson(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });
  let topups = null;
  let loginEmail = "";

  function formatMoney(value, currency) {
    return currency === "VND"
      ? `${Number(value).toLocaleString("vi-VN")}đ`
      : `$${Number(value).toFixed(2)}`;
  }

  function renderTopup(signedIn) {
    const box = $("ai-mode-upgrade");
    const options = $("ai-mode-upgrade-options");
    const providers = topups?.providers || {};
    const canBuy = signedIn && !account?.offline && (providers.payos || providers.lemonsqueezy);
    box.hidden = !canBuy;
    if (!canBuy) return;
    const note = document.createElement("p");
    note.className = "ai-mode-topup-note";
    note.textContent = `Trả đúng chi phí A.I + ${topups.fee_percent}% phí; một chương dài khoảng `
      + `${formatMoney(topups.chapter_estimate_usd, "USD")}. Dùng key A.I của bạn thì miễn phí.`;
    const buttons = [];
    const add = (provider, quote, via) => {
      const paid = quote.pay !== quote.credit ? ` (trả ${formatMoney(quote.pay, quote.currency)})` : "";
      buttons.push([provider, quote.credit, `Nạp ${formatMoney(quote.credit, quote.currency)}${paid} · ${via}`]);
    };
    if (providers.payos) (topups.topups?.payos || []).forEach((quote) => add("payos", quote, "QR ngân hàng"));
    if (providers.lemonsqueezy) (topups.topups?.lemonsqueezy || []).forEach((quote) => add("lemonsqueezy", quote, "Thẻ quốc tế"));
    options.replaceChildren(note, ...buttons.map(([provider, amount, text]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ui-btn ui-btn-compact";
      button.textContent = text;
      button.addEventListener("click", () => checkout(provider, amount, button));
      return button;
    }));
  }

  function renderAccount() {
    const bar = $("ai-mode-plan");
    if (!bar) return;
    bar.hidden = !account?.tiers;
    if (!account?.tiers) return;
    const signedIn = account.signed_in && !account.invalid_token;
    let text = "Đăng nhập để dùng Manga Cloud, hoặc chọn nhà cung cấp khác và dùng key A.I của bạn (miễn phí).";
    if (account.invalid_token) text = "Phiên đăng nhập đã hết — đăng nhập lại.";
    else if (account.offline) text = "Không kết nối được Manga Cloud — vẫn dùng được key A.I của bạn.";
    else if (signedIn && typeof account.balance_usd === "number") {
      text = `${account.email} · Số dư ${formatMoney(account.balance_usd, "USD")}`;
      if (typeof account.chapters_left === "number") text += ` (≈ ${account.chapters_left} chương)`;
    }
    $("ai-mode-plan-text").textContent = text;
    $("ai-mode-login-form").hidden = signedIn;
    $("ai-mode-account").hidden = !signedIn;
    renderTopup(signedIn);
  }

  async function loadAccount() {
    try {
      account = await requestJson("/api/account");
      if (account?.tiers && !topups) topups = await requestJson("/api/account/topups").catch(() => null);
    } catch (_) {
      account = null;
    }
    ensureCloudOption();
    renderAccount();
    syncProviderFields();
  }

  function resetLogin() {
    loginEmail = "";
    $("ai-mode-code-field").hidden = true;
    $("ai-mode-code").value = "";
    $("ai-mode-email").disabled = false;
    $("ai-mode-login-submit").textContent = "Gửi mã";
  }

  async function submitLogin(event) {
    event.preventDefault();
    const submit = $("ai-mode-login-submit");
    submit.disabled = true;
    try {
      if (!loginEmail) {
        const email = $("ai-mode-email").value.trim();
        if (!email) return;
        const sent = await postJson("/api/account/login/start", { email });
        loginEmail = sent.email || email;
        $("ai-mode-email").disabled = true;
        $("ai-mode-code-field").hidden = false;
        $("ai-mode-code").focus();
        submit.textContent = "Đăng nhập";
        window.showToast?.(sent.dev_code ? `Mã thử nghiệm: ${sent.dev_code}` : `Đã gửi mã tới ${loginEmail}`, "success");
        return;
      }
      account = await postJson("/api/account/login/verify", { email: loginEmail, code: $("ai-mode-code").value.trim() });
      resetLogin();
      ensureCloudOption();
      renderAccount();
      syncProviderFields();
    } catch (err) {
      window.showToast?.("Không đăng nhập được: " + err.message, "error");
    } finally {
      submit.disabled = false;
    }
  }

  async function checkout(provider, amount, button) {
    button.disabled = true;
    try {
      const result = await postJson("/api/account/checkout", { provider, amount });
      window.open(result.checkout_url, "_blank", "noopener");
      window.showToast?.("Đã mở trang thanh toán. Số dư tự cộng sau khi thanh toán xong.", "success");
    } catch (err) {
      window.showToast?.("Không mở được thanh toán: " + err.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  async function signOut() {
    try {
      account = await postJson("/api/account/logout");
    } catch (err) {
      window.showToast?.("Không đăng xuất được: " + err.message, "error");
    }
    resetLogin();
    renderAccount();
  }

  async function signOutEverywhere() {
    if (!window.confirm("Đăng xuất Manga Cloud trên mọi máy đang dùng tài khoản này?")) return;
    try {
      account = await postJson("/api/account/logout-all");
      window.showToast?.("Đã đăng xuất mọi thiết bị", "success");
    } catch (err) {
      window.showToast?.("Không đăng xuất được: " + err.message, "error");
    }
    resetLogin();
    renderAccount();
  }

  async function deleteAccount() {
    const warning = "Xoá tài khoản Manga Cloud? Email và phiên đăng nhập bị xoá. "
      + "Số dư còn lại được giữ và trở lại nếu bạn đăng nhập lại bằng đúng email này.";
    if (!window.confirm(warning)) return;
    try {
      account = await requestJson("/api/account", { method: "DELETE" });
      window.showToast?.("Đã xoá tài khoản", "success");
    } catch (err) {
      window.showToast?.("Không xoá được: " + err.message, "error");
    }
    resetLogin();
    renderAccount();
  }

  async function showPayments() {
    const list = $("ai-mode-payment-list");
    try {
      const { ledger } = await requestJson("/api/account/ledger");
      const kinds = { topup: "Nạp tiền", chapter: "Chương A.I mode", refund: "Hoàn tiền", adjust: "Điều chỉnh" };
      list.replaceChildren(...(ledger.length ? ledger : [null]).map((line) => {
        const item = document.createElement("li");
        const sign = line && line.amount_usd > 0 ? "+" : line && line.amount_usd < 0 ? "−" : "";
        item.textContent = line
          ? `${new Date(line.created_at * 1000).toLocaleDateString("vi-VN")} · ${kinds[line.kind] || line.kind} · `
            + `${sign}${formatMoney(Math.abs(line.amount_usd), "USD")} · còn ${formatMoney(line.balance_usd, "USD")}`
          : "Chưa có giao dịch nào.";
        return item;
      }));
      list.hidden = false;
    } catch (err) {
      window.showToast?.("Không tải được lịch sử: " + err.message, "error");
    }
  }

  function watchReturnFromCheckout() {
    const params = new URLSearchParams(window.location.search);
    const state = params.get("billing");
    if (!state) return;
    params.delete("billing");
    const query = params.toString();
    window.history.replaceState(null, "", window.location.pathname + (query ? `?${query}` : "") + window.location.hash);
    if (state !== "paid") return;
    window.showToast?.("Đang chờ xác nhận thanh toán…", "info");
    const before = account?.balance_usd;
    let tries = 0;
    const timer = window.setInterval(async () => {
      tries += 1;
      await loadAccount();
      const now = account?.balance_usd;
      const grew = typeof now === "number" && typeof before === "number" && now > before;
      if (grew || tries >= 20) {
        window.clearInterval(timer);
        if (grew) window.showToast?.(`Đã cộng ${formatMoney(now - before, "USD")} vào số dư`, "success");
      }
    }, 3000);
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
    if (r.textless_pages?.length) lines.push(`Bỏ qua ${r.textless_pages.length} lát không có chữ (giữ ảnh gốc)`);
    if (r.logo_regions) lines.push(`Giữ nguyên ${r.logo_regions} vùng logo`);
    if (r.kept_regions) lines.push(`Trả lại ${r.kept_regions} vùng hình vẽ bị xoá nhầm`);
    if (r.missed_added) lines.push(`Thêm ${r.missed_added} vùng chữ bị sót rồi xoá và dịch`);
    if (r.retried_pages?.length) lines.push(`Dịch lại ${r.retried_pages.length} lát theo lô nhỏ`);
    if (r.restored_regions) {
      const pages = [...new Set((r.review_list || []).map((item) => item.page))].slice(0, 8);
      lines.push(`${r.restored_regions} vùng AI không dịch được, đã giữ ảnh gốc${pages.length ? ` (lát ${pages.join(", ")})` : ""}`);
    }
    if (r.repainted_regions) lines.push(`Repaint ${r.repainted_regions} vùng AI thấy còn sót ở ${r.repaint_pages?.length || 0} lát`);
    if (r.residue_left?.length) {
      const pages = [...new Set(r.residue_left.map((item) => item.page))].slice(0, 8);
      lines.push(`${r.residue_left.length} vùng vẫn còn chữ sau 2 lần xóa, nên xem lại (lát ${pages.join(", ")})`);
    }
    if (r.source_lang) lines.push(`Ngôn ngữ gốc: ${window.SOURCE_LANG_LABELS?.[r.source_lang] || r.source_lang}`);
    if (r.translated || r.unreadable) lines.push(`Dịch ${r.translated || 0} vùng chữ${r.unreadable ? `, ${r.unreadable} vùng AI không đọc được` : ""}`);
    const polish = r.polish || {};
    if (polish.judged) lines.push(`Jev chấm ${polish.judged} câu, gắn cờ ${polish.flagged || 0}, AI viết lại ${polish.rewritten || 0}${polish.still_flagged ? `, còn ${polish.still_flagged} câu nên xem lại` : ""}`);
    if (r.editorial_blockers) {
      const pages = [...new Set((r.blocker_samples || []).map((b) => b.page))].slice(0, 8);
      lines.push(`${r.editorial_blockers} chỗ nên xem lại bằng mắt${pages.length ? ` (ví dụ lát ${pages.join(", ")})` : ""}`);
    }
    const errors = [
      ["quét credit/logo", r.scan_errors], ["so ảnh gốc và clean", r.qc_errors],
      ["dịch", r.translate_errors], ["render", r.render_errors], ["soát câu bằng Jev", polish.errors],
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
    else if (job.status === "completed") summary.textContent = "Xong! Chương đã có chữ, mở ra để xem, sửa và xuất." + cost;
    else if (job.status === "cancelled") summary.textContent = "Đã hủy. Những gì đã làm vẫn được lưu trong chương." + cost;
    else summary.textContent = `Dừng vì lỗi: ${job.error || "không rõ"}` + cost;

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
      const wasRunning = currentJob?.job_id === jobId && (currentJob.status === "pending" || currentJob.status === "running");
      const job = await requestJson(`/api/ai_mode/jobs/${encodeURIComponent(jobId)}`);
      render(job);
      if (job.status === "pending" || job.status === "running") {
        pollTimer = window.setTimeout(() => poll(jobId), POLL_MS);
      } else if (job.status === "completed" && wasRunning && job.chapter_id) {
        // A job finishing while watched opens its chapter; an old finished job only shows its summary.
        window.showToast?.("A.I mode xong, đang mở chương.", "success");
        window.openLetteredChapter = job.chapter_id;
        window.resumeChapter?.(job.chapter_id);
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
      story_notes: $("ai-mode-notes")?.value.trim() || "",
      polish: provider === CLOUD && Boolean($("ai-mode-polish")?.checked),
      images_per_request: Math.min(8, Math.max(1, Math.round(Number($("ai-mode-images")?.value) || 2))),
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
    $("ai-mode-login-form").addEventListener("submit", submitLogin);
    $("ai-mode-signout").addEventListener("click", signOut);
    $("ai-mode-signout-all").addEventListener("click", signOutEverywhere);
    $("ai-mode-delete-account").addEventListener("click", deleteAccount);
    $("ai-mode-payments").addEventListener("click", showPayments);
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
      if (!currentJob?.chapter_id) return;
      if (currentJob.status === "completed") window.openLetteredChapter = currentJob.chapter_id;
      window.resumeChapter?.(currentJob.chapter_id);
    });
    syncProviderFields();
    window.syncAIProviderSelects?.();
    loadAccount().then(watchReturnFromCheckout);
    restore();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
