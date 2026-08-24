(() => {
  const STAGES = ["landing", "preview", "review", "editor"];
  const STAGE_LABELS = {
    landing: "Nhập nội dung",
    preview: "Xử lý ảnh",
    review: "Kiểm tra chất lượng",
    editor: "Biên tập bản dịch",
  };

  function setAppContext(text) {
    const el = document.getElementById("app-context");
    if (el) el.textContent = text || "Chưa mở chương";
  }

  function setAppStage(stage) {
    const resolved = STAGES.includes(stage) ? stage : "landing";
    document.body.dataset.appStage = resolved;

    const landing = document.getElementById("landing-view");
    const workspace = document.getElementById("page-view");
    if (landing) {
      landing.hidden = resolved !== "landing";
      landing.classList.toggle("app-stage-active", resolved === "landing");
    }
    if (workspace) workspace.hidden = resolved === "landing";

    const activeIndex = STAGES.indexOf(resolved);
    document.querySelectorAll(".workflow-step").forEach((step) => {
      const idx = STAGES.indexOf(step.dataset.stage);
      step.classList.toggle("active", idx === activeIndex);
      step.classList.toggle("complete", idx >= 0 && idx < activeIndex);
      if (idx === activeIndex) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });

    if (resolved === "landing") {
      setAppContext("Chưa mở chương");
    } else if (window.currentChapterId) {
      setAppContext(`Chương ${window.currentChapterId}`);
    } else {
      setAppContext(STAGE_LABELS[resolved]);
    }
  }

  function mountAISettings(configEl) {
    const host = document.getElementById("ai-settings-host");
    if (!host || !configEl) return;
    host.innerHTML = "";
    host.appendChild(configEl);
  }

  function openSettings() {
    const drawer = document.getElementById("settings-drawer");
    const backdrop = document.getElementById("settings-backdrop");
    if (!drawer || !backdrop) return;
    drawer.hidden = false;
    backdrop.hidden = false;
    document.body.classList.add("settings-open");
    const closeBtn = document.getElementById("settings-close");
    if (closeBtn) closeBtn.focus();
  }

  function closeSettings() {
    const drawer = document.getElementById("settings-drawer");
    const backdrop = document.getElementById("settings-backdrop");
    if (!drawer || !backdrop) return;
    drawer.hidden = true;
    backdrop.hidden = true;
    document.body.classList.remove("settings-open");
    const toggle = document.getElementById("settings-toggle");
    if (toggle) toggle.focus();
  }

  function wrapRenderer(name, stage) {
    const original = window[name];
    if (typeof original !== "function" || original._uiSystemWrapped) return;
    const wrapped = function uiSystemStageRenderer(...args) {
      setAppStage(stage);
      const result = original.apply(this, args);
      if (window.currentChapterId) setAppContext(`Chương ${window.currentChapterId}`);
      return result;
    };
    wrapped._uiSystemWrapped = true;
    wrapped._uiSystemOriginal = original;
    window[name] = wrapped;
  }

  function setupShellEvents() {
    const toggle = document.getElementById("settings-toggle");
    const close = document.getElementById("settings-close");
    const backdrop = document.getElementById("settings-backdrop");
    if (toggle) toggle.addEventListener("click", openSettings);
    if (close) close.addEventListener("click", closeSettings);
    if (backdrop) backdrop.addEventListener("click", closeSettings);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && document.body.classList.contains("settings-open")) {
        closeSettings();
      }
    });
  }

  window.setAppStage = setAppStage;
  window.setAppContext = setAppContext;
  window.mountAISettings = mountAISettings;
  window.openAppSettings = openSettings;
  window.closeAppSettings = closeSettings;

  wrapRenderer("renderPreview", "preview");
  wrapRenderer("renderReview", "review");
  wrapRenderer("renderEditor", "editor");

  document.addEventListener("DOMContentLoaded", () => {
    setupShellEvents();
    if (!window.currentChapterId) setAppStage("landing");
  });
})();
