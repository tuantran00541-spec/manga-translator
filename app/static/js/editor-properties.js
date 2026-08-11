// editor-properties.js - UI-05 selection + single-object properties panel
(() => {
  const legacyRenderEditor = window.renderEditor;
  if (typeof legacyRenderEditor !== "function") return;

  let selectedBoxKey = null;

  function clearBoxSelection() {
    selectedBoxKey = null;
    document.querySelectorAll(".box-overlay.selected").forEach((el) => el.classList.remove("selected"));
    document.querySelectorAll(".properties-box-list-item.active").forEach((el) => el.classList.remove("active"));
  }
  window.clearBoxSelection = clearBoxSelection;

  function selectBox(pageIndex, boxIndex) {
    const key = `${pageIndex}_${boxIndex}`;
    selectedBoxKey = key;

    const host = document.querySelector(".translation-panel-host");
    const wrapper = document.querySelector(".translation-canvas-host .page-block-wrapper");
    if (!host || !wrapper) return;

    const items = [...document.querySelectorAll(".box-item")];
    const overlays = [...wrapper.querySelectorAll(".box-overlay:not(.drawing)")];
    const listItems = [...document.querySelectorAll(".properties-box-list-item")];
    const editorHost = host.querySelector(".properties-editor-host");
    const titleSpan = host.querySelector(".properties-panel-heading span");

    const targetItem = items.find((el) => el.dataset.propertyKey === key);
    if (!targetItem) return;

    items.forEach((el) => el.classList.toggle("property-hidden", el !== targetItem));
    overlays.forEach((el) => el.classList.toggle("selected", `${el.dataset.pageIndex}_${el.dataset.boxIndex}` === key));

    if (editorHost && targetItem.parentElement !== editorHost) {
      editorHost.innerHTML = "";
      editorHost.appendChild(targetItem);
    }

    if (titleSpan) {
      const boxNum = Number(boxIndex) + 1;
      titleSpan.textContent = `Vùng ${boxNum} · chỉnh nội dung và kiểu chữ.`;
    }

    listItems.forEach((el) => el.classList.toggle("active", el.dataset.key === key));
  }
  window.selectBox = selectBox;

  function setupProperties() {
    const host = document.querySelector(".translation-panel-host");
    const wrapper = document.querySelector(".translation-canvas-host .page-block-wrapper");
    if (!host || !wrapper) return;

    const panel = wrapper.querySelector(".box-panel");
    if (!panel) return;

    const items = [...panel.querySelectorAll(".box-item")];
    const overlays = [...wrapper.querySelectorAll(".box-overlay:not(.drawing)")];

    const title = document.createElement("div");
    title.className = "properties-panel-heading";
    title.innerHTML = '<strong>Thuộc tính</strong><span>Chọn một vùng chữ trên ảnh để chỉnh.</span>';

    const list = document.createElement("div");
    list.className = "properties-box-list";

    const editorHost = document.createElement("div");
    editorHost.className = "properties-editor-host";

    const addBoxBtn = panel.querySelector(".add-box-btn");
    const renderBtn = panel.querySelector(".render-btn");
    const result = panel.querySelector(".render-result");

    host.innerHTML = "";
    host.append(title, list, editorHost);

    const empty = document.createElement("div");
    empty.className = "properties-empty";
    empty.textContent = items.length ? "Chọn vùng chữ trên ảnh hoặc trong danh sách." : "Trang này chưa có vùng chữ.";
    editorHost.appendChild(empty);

    const select = (item, key) => {
      const parts = key.split("_");
      selectBox(parts[0], parts[1]);
    };

    items.forEach((item, index) => {
      const textarea = item.querySelector("textarea[data-box-index]");
      const key = textarea ? `${textarea.dataset.pageIndex}_${textarea.dataset.boxIndex}` : `${host.dataset.pageIndex}_${index}`;
      item.dataset.propertyKey = key;

      const entry = document.createElement("button");
      entry.type = "button";
      entry.className = "properties-box-list-item";
      entry.dataset.key = key;
      const original = item.querySelector(".original");
      entry.innerHTML = `<span class="properties-box-number">${index + 1}</span><span class="properties-box-summary">${(original?.textContent || "Vùng chữ").slice(0, 34)}</span>`;
      entry.addEventListener("click", () => select(item, key));
      list.appendChild(entry);

      item.addEventListener("click", (event) => {
        if (event.target.closest("button, input, select, textarea")) return;
        select(item, key);
      });
    });

    overlays.forEach((overlay) => {
      const key = `${overlay.dataset.pageIndex}_${overlay.dataset.boxIndex}`;
      overlay.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const item = items.find((el) => el.dataset.propertyKey === key);
        if (item) select(item, key);
      });
    });

    if (selectedBoxKey && items.some((el) => el.dataset.propertyKey === selectedBoxKey)) {
      const parts = selectedBoxKey.split("_");
      selectBox(parts[0], parts[1]);
    } else if (items.length) {
      select(items[0], items[0].dataset.propertyKey);
    }

    // Keep page-level actions available without repeating them for every box.
    if (addBoxBtn) host.appendChild(addBoxBtn);
    if (renderBtn) host.appendChild(renderBtn);
    if (result) host.appendChild(result);

    // Keep the original panel out of the visible layout; its children are moved into the properties host.
    panel.style.display = "none";
  }

  window.renderEditor = function propertiesEditorRender() {
    legacyRenderEditor();
    requestAnimationFrame(setupProperties);
  };
})();
