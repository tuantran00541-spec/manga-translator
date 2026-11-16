
(() => {
  const POLL_MS = 1500;
  const JOB_KEY = "manga_ai_mode_job";
  const QUEUE_KEY = "manga_ai_mode_queue";
  const MAX_QUEUE_URLS = 500;
  const RANGE_STEP = 30;
  const STATUS_TEXT = {
    pending: "Chờ", running: "Đang chạy", done: "Xong",
    completed: "Xong", completed_with_blockers: "Cần rà",
    completed_with_errors: "Xong (có lỗi)",
    failed: "Lỗi", cancelled: "Đã hủy", skipped: "Bỏ qua",
  };
  const CLOUD = "manga-cloud";
  let pollTimer = null;
  let pollGeneration = 0;
  let currentJob = null;
  let currentQueue = null;
  let discoveredChapters = [];
  let scopeMode = "single";
  let account = null;
  const visionModelsCache = {};
  let visionModelsLoading = "";

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

  function splitModelId(id) {
    const raw = String(id || "").trim();
    const slash = raw.indexOf("/");
    return slash > 0 ? { name: raw.slice(slash + 1), org: raw.slice(0, slash) } : { name: raw, org: "" };
  }

  function buildCheckIcon() {
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("class", "agent-opt-check");
    icon.setAttribute("viewBox", "0 0 16 16");
    icon.setAttribute("aria-hidden", "true");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", "M3.5 8.5L6.5 11.5L12.5 4.5");
    icon.append(path);
    return icon;
  }

  function buildModelOption({ title, sub, selected, onClick }) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.role = "menuitemradio";
    btn.className = `agent-model-opt${selected ? " is-selected" : ""}`;
    btn.setAttribute("aria-checked", String(Boolean(selected)));
    const copy = document.createElement("span");
    copy.className = "agent-model-opt-copy";
    const head = document.createElement("span");
    head.className = "agent-model-opt-head";
    const t = document.createElement("span");
    t.className = "agent-model-opt-title";
    t.textContent = title;
    head.append(t);
    copy.append(head);
    if (sub) {
      const s = document.createElement("span");
      s.className = "agent-model-opt-sub";
      s.textContent = sub;
      copy.append(s);
    }
    btn.append(copy);
    if (selected) btn.append(buildCheckIcon());
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      onClick();
    });
    return btn;
  }

  function updateVisionTriggerLabel() {
    const model = $("ai-mode-model");
    const trigger = $("ai-mode-model-trigger");
    const label = $("ai-mode-model-trigger-label");
    if (!model || !label) return;
    const val = model.value.trim();
    if (!val) {
      label.textContent = "Mặc định";
      if (trigger) trigger.title = "Dùng mô hình thị giác mặc định trong Cài đặt";
      return;
    }
    const { name } = splitModelId(val);
    label.textContent = name || val;
    if (trigger) trigger.title = val;
  }

  function closeVisionModelMenu() {
    const pop = $("ai-mode-model-pop");
    const trigger = $("ai-mode-model-trigger");
    if (pop) pop.hidden = true;
    if (trigger) trigger.setAttribute("aria-expanded", "false");
  }

  function selectVisionModel(value) {
    const provider = $("ai-mode-provider")?.value || "";
    const model = $("ai-mode-model");
    if (!model) return;
    model.value = String(value || "").trim();
    if (provider) storage("set", "manga_translation_vision_model_" + provider, model.value);
    updateVisionTriggerLabel();
    closeVisionModelMenu();
  }

  function renderVisionModelMenu() {
    const pop = $("ai-mode-model-pop");
    const provider = $("ai-mode-provider")?.value || "";
    const current = ($("ai-mode-model")?.value || "").trim();
    if (!pop) return;

    const search = document.createElement("input");
    search.type = "search";
    search.className = "agent-subpop-search";
    search.placeholder = "Tìm hoặc nhập ID mô hình…";
    search.addEventListener("click", (e) => e.stopPropagation());

    const list = document.createElement("div");
    list.className = "agent-subpop-list";

    const info = window.aiProviderSettings?.[provider];
    const cached = Array.isArray(visionModelsCache[provider]) ? [...visionModelsCache[provider]] : [];
    const defaultModel = String(info?.model || "").trim();
    if (defaultModel && !cached.includes(defaultModel)) cached.unshift(defaultModel);
    if (current && !cached.includes(current)) cached.unshift(current);

    const drawList = (query = "") => {
      const q = query.trim();
      const qLower = q.toLowerCase();
      const rows = [];

      if (!qLower || "mặc định".includes(qLower) || defaultModel.toLowerCase().includes(qLower)) {
        rows.push(buildModelOption({
          title: "Mặc định",
          sub: defaultModel ? `Theo cài đặt (${defaultModel})` : "Theo thiết lập của nhà cung cấp",
          selected: !current,
          onClick: () => selectVisionModel(""),
        }));
      }

      const filtered = qLower ? cached.filter((id) => id.toLowerCase().includes(qLower)) : cached;
      if (rows.length && filtered.length) {
        const sep = document.createElement("div");
        sep.className = "agent-menu-sep";
        rows.push(sep);
      }

      filtered.forEach((id) => {
        const { name, org } = splitModelId(id);
        rows.push(buildModelOption({
          title: name || id,
          sub: org || info?.label || provider,
          selected: id === current,
          onClick: () => selectVisionModel(id),
        }));
      });

      if (q && !cached.some((id) => id.toLowerCase() === qLower)) {
        rows.push(buildModelOption({
          title: `Dùng: ${q}`,
          sub: "Nhập mô hình tùy chỉnh",
          selected: q === current,
          onClick: () => selectVisionModel(q),
        }));
      }

      if (!rows.length) {
        const msg = visionModelsLoading === provider
          ? "Đang tải danh sách mô hình…"
          : "Không tìm thấy mô hình phù hợp.";
        const empty = document.createElement("p");
        empty.className = "agent-subpop-empty";
        empty.textContent = msg;
        rows.push(empty);
      } else if (!filtered.length && visionModelsLoading === provider && !q) {
        const loadingNote = document.createElement("p");
        loadingNote.className = "agent-subpop-empty";
        loadingNote.textContent = "Đang tải danh sách mô hình…";
        rows.push(loadingNote);
      }

      list.replaceChildren(...rows);
    };

    search.addEventListener("input", () => drawList(search.value));
    search.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && search.value.trim()) {
        event.preventDefault();
        const exact = cached.find((id) => id.toLowerCase() === search.value.trim().toLowerCase());
        selectVisionModel(exact || search.value.trim());
      } else if (event.key === "Escape") {
        closeVisionModelMenu();
      }
    });

    drawList("");
    pop.replaceChildren(search, list);
  }

  async function loadVisionModels(providerId) {
    const provider = String(providerId || "").trim();
    if (!provider || provider === CLOUD) return;
    const info = window.aiProviderSettings?.[provider];
    if (info && !info.configured) {
      visionModelsCache[provider] = [];
      if (!$("ai-mode-model-pop")?.hidden) renderVisionModelMenu();
      return;
    }
    const cacheApi = window.aiModelCache;
    const cached = cacheApi?.getCached?.(provider);
    if (cached) {
      visionModelsCache[provider] = cached.models;
      if (!$("ai-mode-model-pop")?.hidden) renderVisionModelMenu();
      if (cached.fresh) return;
    } else if (Array.isArray(visionModelsCache[provider])) {
      return;
    }
    if (visionModelsLoading === provider) return;
    visionModelsLoading = cached ? "" : provider;
    if (!cached && !$("ai-mode-model-pop")?.hidden) renderVisionModelMenu();
    try {
      const models = cacheApi?.fetchModels
        ? await cacheApi.fetchModels(provider, {
            onFresh: (freshModels) => {
              visionModelsCache[provider] = freshModels;
              if (!$("ai-mode-model-pop")?.hidden && $("ai-mode-provider")?.value === provider) {
                renderVisionModelMenu();
              }
            },
          })
        : ((await requestJson(`/api/visual_qc/providers/${encodeURIComponent(provider)}/models`)).models || []);
      visionModelsCache[provider] = Array.isArray(models) ? models : [];
    } catch (_) {
      if (!cached) visionModelsCache[provider] = [];
    } finally {
      if (visionModelsLoading === provider) visionModelsLoading = "";
      if (!$("ai-mode-model-pop")?.hidden) renderVisionModelMenu();
    }
  }

  function syncProviderFields() {
    const provider = $("ai-mode-provider");
    const model = $("ai-mode-model");
    const budgetField = $("ai-mode-budget-field");
    if (!provider || !model) return;
    model.value = storage("get", "manga_translation_vision_model_" + provider.value) || "";
    updateVisionTriggerLabel();
    const cloud = provider.value === CLOUD;
    const modelField = $("ai-mode-model-field") || model.closest("label");
    if (modelField) modelField.hidden = cloud;
    if (cloud) closeVisionModelMenu();
    else void loadVisionModels(provider.value);
    
    if (budgetField) budgetField.hidden = cloud || provider.value !== "deepseek";
    
    const polishField = $("ai-mode-polish-field");
    if (polishField) polishField.hidden = !(cloud || window.aiProviderSettings?.[provider.value]?.capabilities?.judge);
    
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
    note.textContent = `Thanh toán theo đúng chi phí AI + ${topups.fee_percent}% phí; một chương dài ước tính khoảng `
      + `${formatMoney(topups.chapter_estimate_usd, "USD")}. Miễn phí hoàn toàn khi sử dụng khóa API cá nhân.`;
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
    let text = "Đăng nhập để sử dụng Manga Cloud, hoặc chọn nhà cung cấp khác với khóa API cá nhân (miễn phí).";
    if (account.invalid_token) text = "Phiên đăng nhập đã hết hạn — vui lòng đăng nhập lại.";
    else if (account.offline) text = "Không thể kết nối Manga Cloud — vẫn có thể sử dụng khóa API cá nhân.";
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
    const warning = "Xóa tài khoản Manga Cloud? Địa chỉ email và phiên đăng nhập sẽ bị xóa. "
      + "Số dư còn lại vẫn được bảo lưu và khôi phục khi bạn đăng nhập lại bằng đúng email này.";
    if (!window.confirm(warning)) return;
    try {
      account = await requestJson("/api/account", { method: "DELETE" });
      window.showToast?.("Đã xóa tài khoản", "success");
    } catch (err) {
      window.showToast?.("Không thể xóa tài khoản: " + err.message, "error");
    }
    resetLogin();
    renderAccount();
  }

  async function showPayments() {
    const list = $("ai-mode-payment-list");
    try {
      const { ledger } = await requestJson("/api/account/ledger");
      const kinds = { topup: "Nạp tiền", chapter: "Chương Chế độ AI", refund: "Hoàn tiền", adjust: "Điều chỉnh" };
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

  function parseQueueUrls() {
    const raw = $("ai-mode-queue-urls")?.value || "";
    const out = [];
    const seen = new Set();
    raw.split(/\r?\n/).forEach((line) => {
      const clean = line.trim();
      if (!clean) return;
      const key = clean.replace(/\/+$/, "");
      if (seen.has(key)) return;
      seen.add(key);
      out.push(clean);
    });
    return out;
  }

  function updateQueueCount() {
    const urls = parseQueueUrls();
    const countEl = $("ai-mode-queue-count");
    if (countEl) countEl.textContent = String(urls.length);
    const runBtn = $("ai-mode-run");
    if (runBtn && scopeMode === "queue") {
      runBtn.textContent = urls.length ? `Chạy hàng đợi (${Math.min(urls.length, MAX_QUEUE_URLS)} chương)` : "Chạy hàng đợi";
    } else if (runBtn) {
      runBtn.textContent = "Chạy Chế độ AI";
    }
  }

  function setScopeMode(nextMode) {
    scopeMode = nextMode === "queue" ? "queue" : "single";
    const isQueue = scopeMode === "queue";
    const singleBtn = $("ai-mode-scope-single");
    const queueBtn = $("ai-mode-scope-queue");
    if (singleBtn) {
      singleBtn.classList.toggle("is-active", !isQueue);
      singleBtn.setAttribute("aria-pressed", String(!isQueue));
    }
    if (queueBtn) {
      queueBtn.classList.toggle("is-active", isQueue);
      queueBtn.setAttribute("aria-pressed", String(isQueue));
    }
    const singleField = $("ai-mode-single-url-field");
    const urlInput = $("ai-mode-url");
    const queueBox = $("ai-mode-queue-box");
    if (singleField) singleField.hidden = isQueue;
    if (urlInput) urlInput.required = !isQueue;
    if (queueBox) queueBox.hidden = !isQueue;
    updateQueueCount();
  }

  function currentSeriesUrl() {
    if (scopeMode === "queue") {
      return parseQueueUrls()[0] || $("ai-mode-toc-url")?.value.trim() || "";
    }
    return $("ai-mode-url")?.value.trim() || "";
  }

  function shortChapterLabel(url) {
    try {
      const u = new URL(url);
      const parts = u.pathname.split("/").filter(Boolean);
      return parts.length ? parts.slice(-2).join("/") : u.hostname;
    } catch (_) {
      return String(url || "");
    }
  }

  function setRunning(running, { isQueue = false } = {}) {
    const form = $("ai-mode-form");
    form?.querySelectorAll("input, select, textarea, button").forEach((el) => { el.disabled = running; });
    const cancel = $("ai-mode-cancel");
    if (cancel) cancel.hidden = !running || isQueue;
    const queueCancel = $("ai-mode-queue-cancel");
    if (queueCancel) queueCancel.hidden = !running || !isQueue;
  }

  function reportLines(job) {
    const r = job.report || {};
    const lines = [];
    if (r.credit_pages?.length) lines.push(`Đã loại bỏ ${r.credit_pages.length} lát giới thiệu nhóm dịch: lát ${r.credit_pages.map((i) => i + 1).join(", ")}`);
    if (r.credit_rejected?.length) lines.push(`AI xác định ${r.credit_rejected.length} lát là trang giới thiệu — vượt ngưỡng an toàn nên giữ nguyên toàn bộ`);
    if (r.textless_pages?.length) lines.push(`Bỏ qua ${r.textless_pages.length} lát không chứa chữ (giữ nguyên ảnh gốc)`);
    if (r.logo_regions) lines.push(`Giữ nguyên ${r.logo_regions} vùng biểu trưng (logo)`);
    if (r.series?.chapters) lines.push(`Bộ nhớ bộ truyện: ${r.series.names} tên riêng, ${r.series.terms} thuật ngữ, ${r.series.address} cặp xưng hô từ ${r.series.chapters} chương trước`);
    if (r.skip_kept?.length) lines.push(`Không bỏ qua ${r.skip_kept.length} lát vì vẫn phát hiện lời thoại: lát ${r.skip_kept.map((i) => i + 1).join(", ")}`);
    if (r.art_kept) lines.push(`Giữ nguyên ${r.art_kept} vùng hình vẽ bị nhận diện nhầm là chữ`);
    if (r.kept_regions) lines.push(`Khôi phục ${r.kept_regions} vùng hình vẽ bị xóa nhầm`);
    if (r.missed_added) lines.push(`Bổ sung ${r.missed_added} vùng chữ bị bỏ sót rồi tiến hành tẩy nền và dịch`);
    if (r.retried_pages?.length) lines.push(`Dịch lại ${r.retried_pages.length} lát theo lô nhỏ`);
    if (r.restored_regions) {
      const pages = [...new Set((r.review_list || []).map((item) => item.page))].slice(0, 8);
      lines.push(`${r.restored_regions} vùng AI không thể dịch, đã khôi phục ảnh gốc${pages.length ? ` (lát ${pages.join(", ")})` : ""}`);
    }
    if (r.repainted_regions) lines.push(`Đã vá nền bổ sung ${r.repainted_regions} vùng còn sót tại ${r.repaint_pages?.length || 0} lát`);
    if (r.residue_left?.length) {
      const pages = [...new Set(r.residue_left.map((item) => item.page))].slice(0, 8);
      lines.push(`${r.residue_left.length} vùng vẫn còn dấu vết chữ sau 2 lần xóa, cần kiểm tra lại (lát ${pages.join(", ")})`);
    }
    if (r.source_lang) lines.push(`Ngôn ngữ gốc: ${window.SOURCE_LANG_LABELS?.[r.source_lang] || r.source_lang}`);
    if (r.translated || r.unreadable) lines.push(`Đã dịch ${r.translated || 0} vùng chữ${r.unreadable ? `, ${r.unreadable} vùng AI không đọc được` : ""}`);
    const polish = r.polish || {};
    if (polish.judged) lines.push(`Bộ hiệu đính đã rà soát ${polish.judged} câu, gắn cờ ${polish.flagged || 0}, AI biên tập lại ${polish.rewritten || 0}${polish.still_flagged ? `, còn ${polish.still_flagged} câu cần kiểm tra lại` : ""}`);
    if (r.editorial_blockers) {
      const pages = [...new Set((r.blocker_samples || []).map((b) => b.page))].slice(0, 8);
      lines.push(`${r.editorial_blockers} vị trí cần kiểm tra thủ công${pages.length ? ` (ví dụ lát ${pages.join(", ")})` : ""}`);
    }
    const errors = [
      ["quét trang giới thiệu/biểu trưng", r.scan_errors], ["đối chiếu ảnh gốc và nền sạch", r.qc_errors],
      ["dịch thuật", r.translate_errors], ["kết xuất", r.render_errors], ["hiệu đính câu", polish.errors],
    ].filter(([, list]) => list?.length);
    errors.forEach(([label, list]) => {
      const first = typeof list[0] === "string" ? list[0] : list[0]?.error || "";
      lines.push(`Lỗi khi ${label} (${list.length}): ${first}`);
    });
    return lines;
  }

  function renderStageList(job) {
    const stages = $("ai-mode-stages");
    if (!stages) return;
    if (!job?.stages?.length) {
      stages.replaceChildren();
      return;
    }
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
  }

  function render(job) {
    currentQueue = null;
    currentJob = job;
    const status = $("ai-mode-status");
    if (!status || !job) return;
    status.hidden = false;
    const queuePanel = $("ai-mode-queue-panel");
    if (queuePanel) queuePanel.hidden = true;
    renderStageList(job);

    const running = job.status === "pending" || job.status === "running";
    const cost = job.cost_usd == null ? "" : ` · chi phí ~$${Number(job.cost_usd).toFixed(4)}`;
    const summary = $("ai-mode-summary");
    if (running) summary.textContent = (job.cancel_requested ? "Đang hủy sau bước hiện tại…" : "AI đang xử lý, bạn có thể giữ trang này mở hoặc quay lại sau.") + cost;
    else if (job.status === "completed") summary.textContent = "Hoàn tất! Chương đã được chèn chữ và sẵn sàng để kiểm tra, chỉnh sửa và xuất bản." + cost;
    else if (job.status === "cancelled") summary.textContent = "Đã hủy. Kết quả đã thực hiện vẫn được lưu trong chương." + cost;
    else summary.textContent = `Dừng do lỗi: ${job.error || "không xác định"}` + cost;

    const open = $("ai-mode-open");
    open.hidden = running || !job.chapter_id;
    setRunning(running, { isQueue: false });

    const report = $("ai-mode-report");
    const lines = reportLines(job);
    report.hidden = running || !lines.length;
    $("ai-mode-report-list").replaceChildren(...lines.map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    }));
  }

  function formatQueueError(errText) {
    return String(errText || "").replace(/^Download chapter failed:\s*/i, "Tải chương thất bại: ");
  }

  function queueStatusIndicator(status) {
    const wrap = document.createElement("span");
    wrap.className = "ai-mode-queue-indicator";
    wrap.setAttribute("aria-hidden", "true");
    if (status === "completed" || status === "completed_with_blockers" || status === "failed") {
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 16 16");
      const done = status !== "failed";
      svg.setAttribute("class", `ai-mode-queue-icon ${done ? "is-done" : "is-failed"}`);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", done ? "M3.5 8.5 6.5 11.5 12.5 4.5" : "M4 4l8 8M12 4l-8 8");
      svg.append(path);
      wrap.append(svg);
      return wrap;
    }
    const dot = document.createElement("span");
    dot.className = status === "running" ? "ai-mode-queue-dot is-running" : "ai-mode-queue-dot";
    wrap.append(dot);
    return wrap;
  }

  function renderQueue(queue) {
    currentJob = null;
    currentQueue = queue;
    const status = $("ai-mode-status");
    const queuePanel = $("ai-mode-queue-panel");
    if (!status || !queuePanel || !queue) return;
    status.hidden = false;
    queuePanel.hidden = false;

    const running = queue.status === "pending" || queue.status === "running";
    const cost = queue.total_cost_usd == null ? "" : ` · chi phí ~$${Number(queue.total_cost_usd).toFixed(4)}`;
    const currentPos = queue.current_index >= 0 ? `chương ${queue.current_index + 1}/${queue.total}` : `${queue.completed_count}/${queue.total} hoàn tất`;
    const summaryEl = $("ai-mode-queue-summary");
    if (running) {
      summaryEl.textContent = (queue.cancel_requested ? "Đang hủy hàng đợi sau chương hiện tại…" : `Đang chạy ${currentPos}`) + cost;
    } else if (queue.status === "completed") {
      summaryEl.textContent = `Hàng đợi hoàn tất (${queue.completed_count}/${queue.total} chương)` + cost;
    } else if (queue.status === "completed_with_errors") {
      summaryEl.textContent = `Hoàn tất ${queue.completed_count}/${queue.total} chương · ${queue.failed_count} lỗi` + cost;
    } else if (queue.status === "cancelled") {
      summaryEl.textContent = `Đã hủy hàng đợi (${queue.completed_count}/${queue.total} chương đã xong)` + cost;
    } else {
      summaryEl.textContent = `Hàng đợi dừng do lỗi (${queue.completed_count}/${queue.total})` + cost;
    }

    const itemsEl = $("ai-mode-queue-items");
    itemsEl.replaceChildren(...(queue.items || []).map((item) => {
      const li = document.createElement("li");
      li.className = `ai-mode-queue-item is-${item.status}`;
      li.append(queueStatusIndicator(item.status));
      const info = document.createElement("div");
      info.className = "ai-mode-queue-meta";
      const title = document.createElement("strong");
      title.textContent = `#${item.index + 1} · ${shortChapterLabel(item.url)}`;
      title.title = item.url;
      const detailParts = [STATUS_TEXT[item.status] || item.status];
      if (item.cost_usd != null) detailParts.push(`~$${Number(item.cost_usd).toFixed(4)}`);
      if (item.error) detailParts.push(formatQueueError(item.error));
      else if (item.status === "skipped") detailParts.push("Dừng do chương trước gặp lỗi");
      const sub = document.createElement("span");
      sub.textContent = detailParts.join(" · ");
      info.append(title, sub);
      li.append(info);

      if (item.chapter_id) {
        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.className = "ui-btn ui-btn-compact";
        openBtn.textContent = "Mở chương";
        openBtn.addEventListener("click", () => {
          if (item.status === "completed") window.openLetteredChapter = item.chapter_id;
          window.resumeChapter?.(item.chapter_id);
        });
        li.append(openBtn);
      }
      return li;
    }));

    const retryBtn = $("ai-mode-queue-retry");
    const retryable = !running && (queue.items || []).some((i) => ["failed", "cancelled", "skipped"].includes(i.status));
    if (retryBtn) retryBtn.hidden = !retryable;

    if (queue.active_job) {
      renderStageList(queue.active_job);
      $("ai-mode-summary").textContent = `Đang xử lý chương #${queue.current_index + 1}: ${shortChapterLabel(queue.active_job.url)}`;
    } else {
      renderStageList(null);
      $("ai-mode-summary").textContent = "";
    }
    const open = $("ai-mode-open");
    if (open) open.hidden = true;
    const report = $("ai-mode-report");
    if (report) report.hidden = true;
    setRunning(running, { isQueue: true });
  }

  async function pollTarget(config, id, generation, schedule) {
    if (generation === undefined) {
      generation = ++pollGeneration;
      window.clearTimeout(pollTimer);
    }
    try {
      const wasRunning = config.wasRunning(id);
      const data = await requestJson(config.url(id));
      if (generation !== pollGeneration) return;
      config.render(data);
      if (data.status === "pending" || data.status === "running") {
        pollTimer = window.setTimeout(() => schedule(generation), POLL_MS);
      } else if (wasRunning) {
        config.onFinish(data);
      }
    } catch (err) {
      if (generation !== pollGeneration) return;
      if (err.status === 404) {
        storage("remove", config.storageKey);
        setRunning(false, { isQueue: config.isQueue });
        return;
      }
      pollTimer = window.setTimeout(() => schedule(generation), POLL_MS * 2);
    }
  }

  function poll(jobId, generation) {
    return pollTarget({
      url: (id) => `/api/ai_mode/jobs/${encodeURIComponent(id)}`,
      wasRunning: (id) => currentJob?.job_id === id && (currentJob.status === "pending" || currentJob.status === "running"),
      render,
      storageKey: JOB_KEY,
      isQueue: false,
      onFinish: (job) => {
        if (job.status === "completed" && job.chapter_id) {
          window.showToast?.("Chế độ AI đã hoàn tất, đang mở chương.", "success");
          window.openLetteredChapter = job.chapter_id;
          window.resumeChapter?.(job.chapter_id);
        }
      },
    }, jobId, generation, (generation) => poll(jobId, generation));
  }

  function pollQueue(queueId, generation) {
    return pollTarget({
      url: (id) => `/api/ai_mode/queue/${encodeURIComponent(id)}`,
      wasRunning: (id) => currentQueue?.queue_id === id && (currentQueue.status === "pending" || currentQueue.status === "running"),
      render: renderQueue,
      storageKey: QUEUE_KEY,
      isQueue: true,
      onFinish: (queue) => {
        if (queue.status === "completed") {
          window.showToast?.(`Hàng đợi hoàn tất ${queue.completed_count}/${queue.total} chương.`, "success");
        } else if (queue.status === "completed_with_errors") {
          window.showToast?.(`Hàng đợi xong ${queue.completed_count}/${queue.total} chương (${queue.failed_count} lỗi).`, "info");
        }
      },
    }, queueId, generation, (generation) => pollQueue(queueId, generation));
  }

  function buildPayloadOptions() {
    const provider = $("ai-mode-provider").value;
    return {
      provider,
      model: $("ai-mode-model").value.trim() || null,
      target_lang: $("ai-mode-target").value,
      budget_usd: Number($("ai-mode-budget").value || 0.3),
      workers: typeof window.getWorkersSetting === "function" ? window.getWorkersSetting() : 2,
      story_notes: $("ai-mode-notes")?.value.trim() || "",
      polish: provider === CLOUD && Boolean($("ai-mode-polish")?.checked),
      images_per_request: Math.min(8, Math.max(1, Math.round(Number($("ai-mode-images")?.value) || 2))),
    };
  }

  function applyDiscoveredRange(fromOne, toOne) {
    if (!discoveredChapters.length) return;
    const total = discoveredChapters.length;
    const start = Math.max(1, Math.min(total, Math.round(Number(fromOne) || 1)));
    let end = Math.max(start, Math.min(total, Math.round(Number(toOne) || total)));
    if (end - start + 1 > MAX_QUEUE_URLS) end = start + MAX_QUEUE_URLS - 1;
    const slice = discoveredChapters.slice(start - 1, end);
    const fromInput = $("ai-mode-range-from");
    const toInput = $("ai-mode-range-to");
    if (fromInput) {
      fromInput.max = String(total);
      fromInput.value = String(start);
    }
    if (toInput) {
      toInput.max = String(total);
      toInput.value = String(end);
    }
    const info = $("ai-mode-range-info");
    if (info) info.textContent = `Đã quét ${total} chương · đang chọn ${start}–${end}`;
    const box = $("ai-mode-queue-urls");
    if (box) {
      box.value = slice.map((c) => c.url).join("\n");
      updateQueueCount();
    }
  }

  async function discoverChapters() {
    const url = $("ai-mode-toc-url")?.value.trim() || "";
    if (!url) {
      window.showToast?.("Vui lòng dán liên kết trang mục lục truyện trước.", "error");
      return;
    }
    const btn = $("ai-mode-discover-btn");
    if (btn) btn.disabled = true;
    try {
      const data = await postJson("/api/ai_mode/queue/discover", { url });
      const chapters = data.chapters || [];
      if (!chapters.length) {
        window.showToast?.("Không tìm thấy liên kết chương nào từ trang này.", "info");
        return;
      }
      discoveredChapters = chapters;
      const rangeBar = $("ai-mode-range-bar");
      if (rangeBar) rangeBar.hidden = chapters.length <= 1;
      const defaultEnd = Math.min(chapters.length, MAX_QUEUE_URLS);
      applyDiscoveredRange(1, defaultEnd);
      const extraNote = chapters.length > MAX_QUEUE_URLS
        ? ` (đang nạp ${defaultEnd} chương đầu; dùng thanh Từ/Đến bên dưới để chọn các chương tiếp theo)`
        : "";
      window.showToast?.(`Tìm thấy ${chapters.length} chương đúng của bộ truyện${extraNote}.`, "success");
      if ($("ai-mode-series")?.open) loadSeries();
    } catch (err) {
      window.showToast?.("Không quét được mục lục: " + err.message, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function start(event) {
    event.preventDefault();
    const base = buildPayloadOptions();
    if (scopeMode === "queue") {
      const urls = parseQueueUrls();
      if (!urls.length) {
        window.showToast?.("Vui lòng dán ít nhất 1 liên kết chương.", "error");
        return;
      }
      if (urls.length > MAX_QUEUE_URLS) {
        window.showToast?.(`Mỗi hàng đợi tối đa ${MAX_QUEUE_URLS} chương (hiện đang có ${urls.length}).`, "error");
        return;
      }
      setRunning(true, { isQueue: true });
      try {
        const queue = await postJson("/api/ai_mode/queue/start", {
          ...base,
          urls,
          continue_on_error: Boolean($("ai-mode-continue-on-error")?.checked ?? true),
        });
        storage("remove", JOB_KEY);
        storage("set", QUEUE_KEY, queue.queue_id);
        renderQueue(queue);
        pollQueue(queue.queue_id);
        if (base.provider === CLOUD) loadAccount();
      } catch (err) {
        setRunning(false, { isQueue: true });
        window.showToast?.("Không thể chạy hàng đợi AI: " + err.message, "error");
      }
      return;
    }

    const url = $("ai-mode-url")?.value.trim();
    if (!url) return;
    setRunning(true, { isQueue: false });
    try {
      const job = await postJson("/api/ai_mode/start", { url, ...base });
      storage("remove", QUEUE_KEY);
      storage("set", JOB_KEY, job.job_id);
      render(job);
      poll(job.job_id);
      if (base.provider === CLOUD) loadAccount();
    } catch (err) {
      setRunning(false, { isQueue: false });
      window.showToast?.("Không thể chạy Chế độ AI: " + err.message, "error");
    }
  }

  async function loadSeries() {
    const url = currentSeriesUrl();
    const box = $("ai-mode-series-text");
    if (!url || !box) return;
    try {
      const data = await requestJson("/api/ai_mode/series/read", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }),
      });
      box.value = data.text;
      box.placeholder = data.chapters ? "" : "Bộ truyện này chưa có chương nào chạy Chế độ AI; bạn có thể nhập trước tên riêng và cách xưng hô.";
    } catch (err) {
      box.value = "";
      box.placeholder = "Không đọc được bộ nhớ: " + err.message;
    }
  }

  async function saveSeries() {
    const url = currentSeriesUrl();
    if (!url) return window.showToast?.("Vui lòng dán liên kết chương trước.", "error");
    try {
      const data = await requestJson("/api/ai_mode/series/save", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, text: $("ai-mode-series-text").value }),
      });
      $("ai-mode-series-text").value = data.text;
      window.showToast?.("Đã lưu bộ nhớ bộ truyện.", "success");
    } catch (err) {
      window.showToast?.("Không lưu được: " + err.message, "error");
    }
  }

  async function restore() {
    try {
      const active = await requestJson("/api/ai_mode/active");
      const queueId = active.queue?.queue_id || storage("get", QUEUE_KEY);
      if (active.queue?.queue_id || (!active.job?.job_id && queueId)) {
        setScopeMode("queue");
        pollQueue(queueId);
        return;
      }
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
    $("ai-mode-scope-single")?.addEventListener("click", () => setScopeMode("single"));
    $("ai-mode-scope-queue")?.addEventListener("click", () => setScopeMode("queue"));
    $("ai-mode-queue-urls")?.addEventListener("input", updateQueueCount);
    $("ai-mode-discover-btn")?.addEventListener("click", discoverChapters);
    $("ai-mode-range-apply")?.addEventListener("click", () => {
      applyDiscoveredRange($("ai-mode-range-from")?.value, $("ai-mode-range-to")?.value);
    });
    $("ai-mode-range-next")?.addEventListener("click", () => {
      if (!discoveredChapters.length) return;
      const curFrom = Math.max(1, Math.round(Number($("ai-mode-range-from")?.value) || 1));
      const curTo = Math.max(curFrom, Math.round(Number($("ai-mode-range-to")?.value) || discoveredChapters.length));
      const nextFrom = (curTo - curFrom + 1) > RANGE_STEP ? 1 : (curTo >= discoveredChapters.length ? 1 : curTo + 1);
      applyDiscoveredRange(nextFrom, Math.min(discoveredChapters.length, nextFrom + RANGE_STEP - 1));
    });
    $("ai-mode-range-all")?.addEventListener("click", () => {
      if (!discoveredChapters.length) return;
      applyDiscoveredRange(1, Math.min(discoveredChapters.length, MAX_QUEUE_URLS));
    });
    $("ai-mode-series")?.addEventListener("toggle", (event) => { if (event.target.open) loadSeries(); });
    $("ai-mode-url")?.addEventListener("change", () => { if ($("ai-mode-series")?.open) loadSeries(); });
    $("ai-mode-series-save")?.addEventListener("click", saveSeries);
    $("ai-mode-login-form").addEventListener("submit", submitLogin);
    $("ai-mode-signout").addEventListener("click", signOut);
    $("ai-mode-signout-all").addEventListener("click", signOutEverywhere);
    $("ai-mode-delete-account").addEventListener("click", deleteAccount);
    $("ai-mode-payments").addEventListener("click", showPayments);
    $("ai-mode-model-trigger")?.addEventListener("click", (event) => {
      event.stopPropagation();
      const pop = $("ai-mode-model-pop");
      const trigger = $("ai-mode-model-trigger");
      if (!pop || !trigger) return;
      const willOpen = pop.hidden;
      if (willOpen) {
        renderVisionModelMenu();
        pop.hidden = false;
        trigger.setAttribute("aria-expanded", "true");
        void loadVisionModels(provider.value);
        pop.querySelector("input")?.focus();
      } else {
        closeVisionModelMenu();
      }
    });
    document.addEventListener("click", (event) => {
      const field = $("ai-mode-model-field");
      if (field && !field.contains(event.target)) closeVisionModelMenu();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeVisionModelMenu();
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
    $("ai-mode-queue-cancel")?.addEventListener("click", async () => {
      if (!currentQueue) return;
      try {
        renderQueue(await postJson(`/api/ai_mode/queue/${encodeURIComponent(currentQueue.queue_id)}/cancel`));
      } catch (err) {
        window.showToast?.("Không hủy được hàng đợi: " + err.message, "error");
      }
    });
    $("ai-mode-queue-retry")?.addEventListener("click", () => {
      if (!currentQueue) return;
      const failedUrls = (currentQueue.items || [])
        .filter((i) => ["failed", "cancelled", "skipped"].includes(i.status))
        .map((i) => i.url);
      if (!failedUrls.length) return;
      setScopeMode("queue");
      const box = $("ai-mode-queue-urls");
      if (box) box.value = failedUrls.join("\n");
      updateQueueCount();
      window.showToast?.(`Đã nạp lại ${failedUrls.length} chương chưa hoàn tất.`, "info");
    });
    $("ai-mode-open").addEventListener("click", () => {
      if (!currentJob?.chapter_id) return;
      if (currentJob.status === "completed") window.openLetteredChapter = currentJob.chapter_id;
      window.resumeChapter?.(currentJob.chapter_id);
    });
    syncProviderFields();
    setScopeMode("single");
    window.syncAIProviderSelects?.();
    loadAccount().then(watchReturnFromCheckout);
    restore();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
