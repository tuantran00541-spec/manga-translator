(() => {
  function parseResponse(response) {
    return typeof window.parseApiResponse === "function"
      ? window.parseApiResponse(response)
      : response.json().catch(() => ({}));
  }

  function errorMessage(status, data) {
    return typeof window.getErrorMessage === "function"
      ? window.getErrorMessage(status, data)
      : data?.detail || `HTTP ${status}`;
  }

  function buildControls(toolbar) {
    if (!toolbar || toolbar.querySelector(".chapter-translate-controls")) return;
    const controls = document.createElement("details");
    controls.className = "chapter-translate-controls command-disclosure";
    const summary = document.createElement("summary");
    summary.className = "ui-btn ui-btn-ghost";
    summary.textContent = "Dịch tự động";
    const options = document.createElement("div");
    options.className = "command-disclosure-panel chapter-translate-options";

    const target = document.createElement("select");
    target.className = "chapter-translate-target";
    target.setAttribute("aria-label", "Ngôn ngữ bản dịch");
    [
      ["vi", "→ Tiếng Việt"],
      ["en", "→ English"],
    ].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      target.appendChild(option);
    });

    const budget = document.createElement("input");
    budget.type = "number";
    budget.min = "0.001";
    budget.max = "0.25";
    budget.step = "0.001";
    budget.value = "0.02";
    budget.className = "chapter-translate-budget";
    budget.title = "Ngân sách tối đa ước tính cho lần dịch chương (USD)";
    budget.setAttribute("aria-label", "Ngân sách dịch chương bằng USD");
    const targetLabel = document.createElement("label");
    targetLabel.className = "ui-field";
    targetLabel.textContent = "Dịch sang";
    targetLabel.appendChild(target);
    const budgetLabel = document.createElement("label");
    budgetLabel.className = "ui-field";
    budgetLabel.textContent = "Giới hạn chi phí (USD)";
    budgetLabel.appendChild(budget);

    const run = document.createElement("button");
    run.type = "button";
    run.className = "ui-btn ui-btn-primary chapter-translate-run";
    run.textContent = "Dịch tự động";

    run.addEventListener("click", async () => {
      const chapterId = window.currentChapterId;
      if (!chapterId) return;
      run.disabled = true;
      run.textContent = "Đang dịch…";
      summary.textContent = "Đang dịch…";
      try {
        if (typeof window.flushAllPendingPersists === "function") {
          await window.flushAllPendingPersists();
        } else if (typeof window.flushTextObjectPersist === "function") {
          await window.flushTextObjectPersist();
        }
        if (chapterId !== window.currentChapterId) return;

        const response = await fetch("/api/translate/chapter", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chapter_id: chapterId,
            source_lang: document.getElementById("lang-select")?.value || "ja",
            target_lang: target.value,
            budget_usd: Number(budget.value || 0.02),
            force: false,
          }),
        });
        const data = await parseResponse(response);
        if (!response.ok) throw new Error(errorMessage(response.status, data));
        if (chapterId !== window.currentChapterId) return;
        window.currentManifest = data;
        const info = data.translation_run || {};
        const cost = Number(info.estimated_cost_usd || 0).toFixed(4);
        if (typeof window.showToast === "function") {
          window.showToast(`Đã dịch ${info.translated || 0} vùng · chi phí ~$${cost}`, "info");
        }
        if (typeof window.renderEditor === "function") window.renderEditor();
      } catch (err) {
        if (typeof window.showToast === "function") {
          window.showToast("Dịch tự động thất bại: " + err.message, "error");
        }
      } finally {
        run.disabled = false;
        run.textContent = "Dịch tự động";
        summary.textContent = "Dịch tự động";
      }
    });

    options.append(targetLabel, budgetLabel, run);
    controls.append(summary, options);
    const renderButton = toolbar.querySelector(".editor-render-btn");
    if (renderButton) toolbar.insertBefore(controls, renderButton);
    else toolbar.appendChild(controls);
  }

  const observer = new MutationObserver(() => {
    document.querySelectorAll(".translation-sticky-toolbar").forEach(buildControls);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".translation-sticky-toolbar").forEach(buildControls);
  });
})();
