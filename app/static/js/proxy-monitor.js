(() => {
  "use strict";

  const CALLER_LABELS = {
    ai_mode: "Chế độ AI",
    translation: "Dịch ảnh",
    chaos_agent: "Chaos",
    local_proxy: "Local Endpoint",
  };

  const STATUS_LABELS = {
    ok: "OK",
    rescued: "Đã cứu",
    skipped_quota: "Bỏ qua 429/Quota",
    failed: "Lỗi",
  };

  let timer = null;
  let hostMounted = false;
  let lastSnapshotKey = "";

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function callerText(raw) {
    const base = String(raw || "app").split(":")[0];
    return CALLER_LABELS[base] || raw || "App";
  }

  function formatTime(epochSec) {
    try {
      return new Date(Number(epochSec || 0) * 1000).toLocaleTimeString([], {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch (_) {
      return "";
    }
  }

  function renderMonitor(data, force = false) {
    const host = document.getElementById("proxy-monitor-host");
    if (!host) return;
    hostMounted = true;
    const key = JSON.stringify({
      stats: data?.stats || {},
      ep: data?.local_endpoint || "",
      wrapped: (data?.wrapped_targets || []).map((w) => `${w.api_base}:${w.model}:${w.rescued_count}:${(w.rules || []).join(",")}`),
      ids: (data?.events || []).map((e) => `${e.id}:${e.status}:${e.attempts}`),
    });
    if (!force && key === lastSnapshotKey) return;
    lastSnapshotKey = key;

    const card = el("section", "proxy-monitor-card");
    const head = el("div", "proxy-monitor-head");
    const titleWrap = el("div", "proxy-monitor-title-wrap");
    titleWrap.append(
      el("h3", "proxy-monitor-title", "Proxy cứu lỗi API"),
      el("span", "proxy-monitor-pill", "Tự động bật · bỏ qua 429/quota"),
    );

    const actions = el("div", "proxy-monitor-actions");
    const refreshBtn = el("button", "ui-btn ui-btn-ghost ui-btn-compact", "Làm mới");
    refreshBtn.type = "button";
    refreshBtn.addEventListener("click", () => loadProxyStatus());
    const clearBtn = el("button", "ui-btn ui-btn-ghost ui-btn-compact", "Xóa log");
    clearBtn.type = "button";
    clearBtn.addEventListener("click", async () => {
      try {
        await fetch("/api/proxy/logs/clear", { method: "POST" });
        await loadProxyStatus();
      } catch (_) {}
    });
    actions.append(refreshBtn, clearBtn);
    head.append(titleWrap, actions);

    const endpoint = data?.local_endpoint || `${window.location.origin}/api/proxy/v1`;
    const epRow = el("div", "proxy-monitor-endpoint");
    const epCode = el("code", "proxy-monitor-code", endpoint);
    const copyBtn = el("button", "ui-btn ui-btn-ghost ui-btn-compact", "Sao chép");
    copyBtn.type = "button";
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(endpoint);
        window.showToast?.("Đã sao chép địa chỉ Proxy API.", "info");
      } catch (_) {}
    });
    epRow.append(el("span", "proxy-monitor-ep-label", "Endpoint nội bộ:"), epCode, copyBtn);

    const stats = data?.stats || {};
    const statsGrid = el("div", "proxy-monitor-stats");
    [
      ["Tổng gọi", String(stats.total_calls || 0), ""],
      ["Đã cứu", String(stats.rescued_calls || 0), "is-rescued"],
      ["Bỏ qua 429", String(stats.skipped_quota_calls || 0), "is-skipped"],
      ["Lỗi khác", String(stats.failed_calls || 0), "is-failed"],
    ].forEach(([label, val, cls]) => {
      const cell = el("div", `proxy-stat-cell ${cls}`.trim());
      cell.append(el("span", "proxy-stat-val", val), el("span", "proxy-stat-label", label));
      statsGrid.append(cell);
    });

    const wrappedBox = el("div", "proxy-wrapped-section");
    const wrapped = Array.isArray(data?.wrapped_targets) ? data.wrapped_targets : [];
    if (wrapped.length) {
      wrappedBox.append(el("p", "proxy-wrapped-heading", `Đang bọc tự động (${wrapped.length} model từng gặp lỗi)`));
      wrapped.forEach((item) => {
        const wRow = el("div", "proxy-wrapped-row");
        const wHead = el("div", "proxy-wrapped-top");
        wHead.append(
          el("span", "proxy-log-model", [item.provider, item.model].filter(Boolean).join(" · ")),
          el("span", "proxy-log-meta", `${item.rescued_count || 1} lần cứu`),
        );
        const unwrapBtn = el("button", "ui-btn ui-btn-ghost ui-btn-compact", "Gỡ bọc");
        unwrapBtn.type = "button";
        unwrapBtn.addEventListener("click", async () => {
          try {
            await fetch("/api/proxy/unwrap", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ api_base: item.api_base, model: item.model }),
            });
            await loadProxyStatus(true);
          } catch (_) {}
        });
        wHead.append(unwrapBtn);
        wRow.append(wHead);
        const subParts = [item.api_base, item.key_hint ? `Key ${item.key_hint}` : ""].filter(Boolean).join(" · ");
        if (subParts) wRow.append(el("p", "proxy-log-err", subParts));
        if (Array.isArray(item.rules) && item.rules.length) {
          wRow.append(el("p", "proxy-log-rescue", `Quy tắc bọc: ${item.rules.join(" · ")}`));
        }
        wrappedBox.append(wRow);
      });
    }

    const list = el("div", "proxy-monitor-log");
    const events = Array.isArray(data?.events) ? data.events : [];
    const interesting = events.filter((e) => e.status !== "ok");
    const shown = (interesting.length ? interesting : events).slice(0, 40);

    if (!shown.length) {
      list.append(el("p", "proxy-monitor-empty", "Chưa ghi nhận lỗi hay lượt cứu nào trong phiên chạy máy chủ này."));
    } else {
      shown.forEach((ev) => {
        const row = el("div", `proxy-log-row is-${ev.status || "ok"}`);
        const top = el("div", "proxy-log-top");
        top.append(
          el("span", "proxy-log-badge", STATUS_LABELS[ev.status] || ev.status),
          el("span", "proxy-log-caller", callerText(ev.caller)),
          el("span", "proxy-log-model", [ev.provider, ev.model].filter(Boolean).join(" · ") || "API"),
          el("span", "proxy-log-meta", `${formatTime(ev.time)} · ${ev.latency_ms || 0}ms${ev.attempts > 1 ? ` · ${ev.attempts} lần` : ""}`),
        );
        row.append(top);
        const firstErr = Array.isArray(ev.errors_seen) && ev.errors_seen.length ? ev.errors_seen[0] : null;
        if (firstErr?.detail || ev.detail) {
          row.append(el("p", "proxy-log-err", `Lỗi: ${(firstErr?.detail || ev.detail || "").slice(0, 180)}`));
        }
        if (Array.isArray(ev.rescues_applied) && ev.rescues_applied.length) {
          row.append(el("p", "proxy-log-rescue", `Cứu: ${ev.rescues_applied.join(" → ")}`));
        }
        list.append(row);
      });
    }

    card.append(head, epRow, statsGrid);
    if (wrapped.length) card.append(wrappedBox);
    card.append(list);
    host.replaceChildren(card);
  }

  async function loadProxyStatus(force = false) {
    try {
      const resp = await fetch("/api/proxy/status?limit=60");
      if (!resp.ok) return;
      const data = await window.parseApiResponse(resp);
      renderMonitor(data, force);
    } catch (_) {}
  }

  function startRefresh() {
    const host = document.getElementById("proxy-monitor-host");
    const visible = Boolean(host && !host.closest("[hidden]"));
    if (visible) loadProxyStatus(true);
    clearInterval(timer);
    timer = setInterval(() => {
      const currentHost = document.getElementById("proxy-monitor-host");
      if (document.body.classList.contains("settings-open") && currentHost && !currentHost.closest("[hidden]")) {
        loadProxyStatus(false);
      } else {
        clearInterval(timer);
      }
    }, 3000);
  }

  window.refreshProxyMonitor = (force = false) => loadProxyStatus(force);

  document.addEventListener("DOMContentLoaded", () => {
    if (!hostMounted && document.getElementById("proxy-monitor-host")) {
      loadProxyStatus(true);
    }
  });
  document.addEventListener("settings-opened", startRefresh);
  document.addEventListener("settings-closed", () => clearInterval(timer));
})();
