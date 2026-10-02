// Named text styles kept on the server and applied to one or many text objects.
(() => {
  let presets = [];
  let loaded = null;

  const STYLE_KEYS = Object.keys(window.DEFAULT_TEXT_OBJECT_STYLE || {});

  function notify() {
    document.dispatchEvent(new CustomEvent("style-presets-changed", { detail: { presets } }));
  }

  function load(force = false) {
    if (loaded && !force) return loaded;
    loaded = fetch("/api/style_presets")
      .then((r) => (r.ok ? r.json() : { presets: [] }))
      .then((data) => { presets = data.presets || []; notify(); return presets; })
      .catch(() => { presets = []; return presets; });
    return loaded;
  }

  async function save(next) {
    const response = await fetch("/api/style_presets", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ presets: next }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
    presets = data.presets || [];
    loaded = Promise.resolve(presets);
    notify();
    return presets;
  }

  function styleOf(obj) {
    const style = {};
    STYLE_KEYS.forEach((key) => { style[key] = obj.style?.[key] ?? window.DEFAULT_TEXT_OBJECT_STYLE[key]; });
    if (obj.font_selection_mode === "auto") style.font = "auto";
    return style;
  }

  async function saveFrom(name, obj) {
    const clean = String(name || "").trim();
    if (!clean) throw new Error("Đặt tên cho kiểu chữ mẫu");
    const next = presets.filter((p) => p.name !== clean);
    next.push({ name: clean, style: styleOf(obj) });
    return save(next);
  }

  function remove(name) {
    return save(presets.filter((p) => p.name !== name));
  }

  function applyTo(obj, preset) {
    obj.style = Object.assign({}, window.DEFAULT_TEXT_OBJECT_STYLE, obj.style || {}, preset.style);
    obj.font_selection_mode = preset.style.font === "auto" ? "auto" : "user";
    obj.font_ai_id = null;
    obj.font_match = null;
  }

  // Applies a preset to the given objects as one undo step and saves them.
  function applyPreset(preset, targets) {
    const live = targets.filter(({ pageIndex, id }) => window.findTextObject?.(pageIndex, id));
    if (!preset || !live.length) return 0;
    const run = () => live.forEach(({ pageIndex, id }) => {
      applyTo(window.findTextObject(pageIndex, id), preset);
      window.scheduleTextObjectPersist?.(pageIndex, id);
    });
    if (window.editorHistory) window.editorHistory.batch(live, run);
    else run();
    return live.length;
  }

  function buildSection(body, obj, pageIndex) {
    const row = document.createElement("div");
    row.className = "ui-control-row style-preset-row";
    const select = document.createElement("select");
    select.className = "ui-select style-preset-select";
    select.setAttribute("aria-label", "Kiểu chữ mẫu");
    const apply = document.createElement("button");
    apply.type = "button";
    apply.className = "ui-btn ui-btn-ghost ui-btn-compact style-preset-apply";
    apply.textContent = "Áp dụng";
    const remove_ = document.createElement("button");
    remove_.type = "button";
    remove_.className = "ui-icon-btn ui-btn-ghost ui-btn-compact style-preset-delete";
    remove_.setAttribute("aria-label", "Xóa kiểu chữ mẫu đang chọn");
    remove_.title = "Xóa kiểu chữ mẫu đang chọn";
    remove_.append(window.createUiIcon("trash"));
    const saveRow = document.createElement("div");
    saveRow.className = "ui-control-row style-preset-save-row";
    const name = document.createElement("input");
    name.className = "ui-input style-preset-name";
    name.placeholder = "Tên mẫu: Lời thoại, Dẫn truyện…";
    name.maxLength = 40;
    name.setAttribute("aria-label", "Tên kiểu chữ mẫu mới");
    const store = document.createElement("button");
    store.type = "button";
    store.className = "ui-btn ui-btn-ghost ui-btn-compact style-preset-store";
    store.textContent = "Lưu kiểu này";
    store.title = "Lưu kiểu chữ của vùng đang chọn thành mẫu";

    const fill = () => {
      select.replaceChildren(new Option(presets.length ? "Chọn kiểu mẫu…" : "Chưa có kiểu mẫu", ""));
      presets.forEach((p, i) => select.add(new Option(i < 9 ? `${i + 1}. ${p.name}` : p.name, p.name)));
      apply.disabled = remove_.disabled = true;
    };
    select.addEventListener("change", () => { apply.disabled = remove_.disabled = !select.value; });
    apply.addEventListener("click", () => {
      const preset = presets.find((p) => p.name === select.value);
      if (!applyPreset(preset, [{ pageIndex, id: obj.id }])) return;
      window.renderEditorPanel?.(pageIndex);
      window.showToast?.(`Đã áp kiểu "${preset.name}".`, "success");
    });
    remove_.addEventListener("click", async () => {
      const target = select.value;
      if (!target) return;
      try {
        await remove(target);
        window.showToast?.(`Đã xóa kiểu "${target}".`, "info");
      } catch (err) {
        window.showToast?.("Không xóa được kiểu mẫu: " + err.message, "error");
      }
    });
    const doStore = async () => {
      try {
        await saveFrom(name.value, obj);
        window.showToast?.(`Đã lưu kiểu "${name.value.trim()}".`, "success");
        name.value = "";
      } catch (err) {
        window.showToast?.("Không lưu được kiểu mẫu: " + err.message, "error");
      }
    };
    store.addEventListener("click", doStore);
    name.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); doStore(); } });
    const onChange = () => { if (!select.isConnected) { document.removeEventListener("style-presets-changed", onChange); return; } fill(); };
    document.addEventListener("style-presets-changed", onChange);

    fill();
    load();
    row.append(select, apply, remove_);
    saveRow.append(name, store);
    body.append(row, saveRow);
  }

  window.stylePresets = {
    load,
    list: () => presets,
    applyPreset,
    buildSection,
  };
})();
