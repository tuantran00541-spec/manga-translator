(() => {
  const STORAGE_KEY = "mt_theme";
  const MODES = new Set(["light", "dark", "system"]);
  const MODE_ORDER = ["system", "light", "dark"];
  const MODE_LABELS = {
    system: "Hệ thống",
    light: "Sáng",
    dark: "Tối",
  };
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function savedMode() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return MODES.has(value) ? value : "system";
    } catch (_) {
      return "system";
    }
  }

  function syncThemeControls(mode) {
    const normalized = MODES.has(mode) ? mode : "system";
    const select = document.getElementById("theme-select");
    if (select && select.value !== normalized) select.value = normalized;

    const trigger = document.getElementById("theme-trigger");
    if (trigger) {
      const label = trigger.querySelector(".theme-trigger-label");
      if (label) label.textContent = MODE_LABELS[normalized];
      trigger.setAttribute("aria-label", `Giao diện: ${MODE_LABELS[normalized]}`);
    }

    document.querySelectorAll("#theme-menu [data-theme-mode]").forEach((option) => {
      option.setAttribute(
        "aria-selected",
        option.dataset.themeMode === normalized ? "true" : "false",
      );
    });
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
    syncThemeControls(normalized);
    window.dispatchEvent(new CustomEvent("app-theme-change", { detail: { mode: normalized, resolved } }));
    return normalized;
  }

  function buildThemePicker() {
    const select = document.getElementById("theme-select");
    if (!select) return;
    const legacyHost = select.closest(".theme-control");
    if (!legacyHost || legacyHost.dataset.themePickerReady === "true") return;

    const picker = document.createElement("div");
    picker.className = `${legacyHost.className} theme-picker`;
    picker.dataset.themePickerReady = "true";

    const trigger = document.createElement("button");
    trigger.id = "theme-trigger";
    trigger.className = "theme-trigger";
    trigger.type = "button";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", "theme-menu");

    const sourceIcon = legacyHost.querySelector("svg");
    if (sourceIcon) {
      sourceIcon.setAttribute("aria-hidden", "true");
      trigger.appendChild(sourceIcon);
    }

    const triggerLabel = document.createElement("span");
    triggerLabel.className = "theme-trigger-label";
    trigger.appendChild(triggerLabel);

    const menu = document.createElement("div");
    menu.id = "theme-menu";
    menu.className = "theme-menu";
    menu.setAttribute("role", "listbox");
    menu.setAttribute("aria-label", "Chọn giao diện");
    menu.hidden = true;

    MODE_ORDER.forEach((mode) => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "theme-option";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.dataset.themeMode = mode;
      option.tabIndex = -1;
      option.textContent = MODE_LABELS[mode];
      menu.appendChild(option);
    });

    select.classList.add("theme-select-native");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");

    picker.appendChild(trigger);
    picker.appendChild(menu);
    picker.appendChild(select);
    legacyHost.replaceWith(picker);

    const getOptions = () => [...menu.querySelectorAll("[data-theme-mode]")];
    const selectedOption = () => (
      menu.querySelector('[aria-selected="true"]') || getOptions()[0] || null
    );

    const closeMenu = (returnFocus = false) => {
      if (menu.hidden) return;
      menu.hidden = true;
      picker.classList.remove("open");
      trigger.setAttribute("aria-expanded", "false");
      if (returnFocus) trigger.focus();
    };

    const openMenu = () => {
      if (!menu.hidden) return;
      menu.hidden = false;
      picker.classList.add("open");
      trigger.setAttribute("aria-expanded", "true");
      requestAnimationFrame(() => selectedOption()?.focus());
    };

    trigger.addEventListener("click", () => {
      if (menu.hidden) openMenu();
      else closeMenu(false);
    });

    trigger.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        openMenu();
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeMenu(true);
      }
    });

    menu.addEventListener("click", (event) => {
      const option = event.target.closest("[data-theme-mode]");
      if (!option) return;
      applyMode(option.dataset.themeMode);
      closeMenu(true);
    });

    menu.addEventListener("keydown", (event) => {
      const options = getOptions();
      const current = event.target.closest("[data-theme-mode]");
      const index = Math.max(0, options.indexOf(current));
      let target = null;

      if (event.key === "ArrowDown") target = options[(index + 1) % options.length];
      if (event.key === "ArrowUp") target = options[(index - 1 + options.length) % options.length];
      if (event.key === "Home") target = options[0];
      if (event.key === "End") target = options[options.length - 1];
      if (event.key === "Escape") {
        event.preventDefault();
        closeMenu(true);
        return;
      }
      if (event.key === "Tab") {
        closeMenu(false);
        return;
      }
      if (target) {
        event.preventDefault();
        target.focus();
      }
    });

    document.addEventListener("pointerdown", (event) => {
      if (!menu.hidden && !picker.contains(event.target)) closeMenu(false);
    });
    document.addEventListener("focusin", (event) => {
      if (!menu.hidden && !picker.contains(event.target)) closeMenu(false);
    });
    window.addEventListener("resize", () => closeMenu(false));

    select.addEventListener("change", () => applyMode(select.value));
    syncThemeControls(document.documentElement.dataset.theme || "system");
  }

  applyMode(savedMode(), false);
  const onSystemChange = () => {
    if ((document.documentElement.dataset.theme || "system") === "system") applyMode("system", false);
  };
  if (typeof media.addEventListener === "function") media.addEventListener("change", onSystemChange);
  else media.addListener(onSystemChange);

  document.addEventListener("DOMContentLoaded", buildThemePicker);

  window.setAppTheme = applyMode;
  window.getAppTheme = () => document.documentElement.dataset.theme || "system";
})();
