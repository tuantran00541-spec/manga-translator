(() => {
  "use strict";

  const STORAGE_KEY = "mt_theme";
  const MODES = new Set(["light", "dark", "system"]);
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function savedMode() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return MODES.has(value) ? value : "system";
    } catch (_) {
      return "system";
    }
  }

  const MODE_LABELS = { system: "Hệ thống", light: "Sáng", dark: "Tối" };
  const MODE_ICONS = {
    system: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    light: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/>',
    dark: '<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/>',
  };

  function syncControl(mode) {
    const select = document.getElementById("theme-select");
    if (select && select.value !== mode) select.value = mode;
    const label = document.getElementById("theme-control-label");
    if (label) label.textContent = MODE_LABELS[mode] || MODE_LABELS.system;
    const icon = document.querySelector(".theme-control-icon");
    if (icon) icon.innerHTML = MODE_ICONS[mode] || MODE_ICONS.system;
  }

  function applyMode(mode, persist = true) {
    const normalized = MODES.has(mode) ? mode : "system";
    const resolved = normalized === "system"
      ? (media.matches ? "dark" : "light")
      : normalized;

    document.documentElement.dataset.theme = normalized;
    document.documentElement.dataset.resolvedTheme = resolved;
    document.documentElement.style.colorScheme = resolved;
    syncControl(normalized);

    if (persist) {
      try { localStorage.setItem(STORAGE_KEY, normalized); } catch (_) {}
    }

    window.dispatchEvent(new CustomEvent("app-theme-change", {
      detail: { mode: normalized, resolved },
    }));
    return normalized;
  }

  function bindThemeSelect() {
    const select = document.getElementById("theme-select");
    if (!select || select.dataset.themeBound === "true") return;
    select.dataset.themeBound = "true";
    select.addEventListener("change", () => applyMode(select.value));
    syncControl(document.documentElement.dataset.theme || "system");
  }

  applyMode(savedMode(), false);

  const onSystemChange = () => {
    if ((document.documentElement.dataset.theme || "system") === "system") {
      applyMode("system", false);
    }
  };
  if (typeof media.addEventListener === "function") {
    media.addEventListener("change", onSystemChange);
  } else {
    media.addListener(onSystemChange);
  }

  document.addEventListener("DOMContentLoaded", bindThemeSelect);
  window.setAppTheme = applyMode;
})();
