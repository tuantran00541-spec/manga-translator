(() => {
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

  function applyMode(mode, persist = true) {
    const normalized = MODES.has(mode) ? mode : "system";
    const resolved = normalized === "system" ? (media.matches ? "dark" : "light") : normalized;
    document.documentElement.dataset.theme = normalized;
    document.documentElement.dataset.resolvedTheme = resolved;
    document.documentElement.style.colorScheme = resolved;
    if (persist) {
      try { localStorage.setItem(STORAGE_KEY, normalized); } catch (_) {}
    }
    const select = document.getElementById("theme-select");
    if (select && select.value !== normalized) select.value = normalized;
    window.dispatchEvent(new CustomEvent("app-theme-change", { detail: { mode: normalized, resolved } }));
    return normalized;
  }

  applyMode(savedMode(), false);
  const onSystemChange = () => {
    if ((document.documentElement.dataset.theme || "system") === "system") applyMode("system", false);
  };
  if (typeof media.addEventListener === "function") media.addEventListener("change", onSystemChange);
  else media.addListener(onSystemChange);

  document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("theme-select");
    if (!select) return;
    select.value = document.documentElement.dataset.theme || "system";
    select.addEventListener("change", () => applyMode(select.value));
  });

  window.setAppTheme = applyMode;
  window.getAppTheme = () => document.documentElement.dataset.theme || "system";
})();
