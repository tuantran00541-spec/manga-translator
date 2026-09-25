function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    try {
      const dataUrl = canvas.toDataURL("image/png");
      const parts = dataUrl.split(",");
      if (parts.length < 2) {
        throw new Error("Không thể tạo dữ liệu vùng đánh dấu từ canvas");
      }
      const binStr = atob(parts[1]);
      const len = binStr.length;
      const u8arr = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        u8arr[i] = binStr.charCodeAt(i);
      }
      resolve(new Blob([u8arr], { type: "image/png" }));
    } catch (err) {
      reject(err);
    }
  });
}
window.canvasToBlob = canvasToBlob;

function chooseRepaintMode() {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.className = "repaint-mode-backdrop";

    const dialog = document.createElement("section");
    dialog.className = "repaint-mode-dialog";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.setAttribute("aria-labelledby", "repaint-mode-title");

    const eyebrow = document.createElement("p");
    eyebrow.className = "repaint-mode-eyebrow";
    eyebrow.textContent = "Tái xử lý vùng đánh dấu";

    const title = document.createElement("h2");
    title.id = "repaint-mode-title";
    title.textContent = "Chọn phương thức xử lý";

    const description = document.createElement("p");
    description.className = "repaint-mode-description";
    description.textContent = "Lựa chọn này áp dụng cho toàn bộ vùng đánh dấu hiện có trên trang.";

    const options = document.createElement("div");
    options.className = "repaint-mode-options";
    let selectedMode = "standard";

    const selectMode = (mode) => {
      selectedMode = mode;
      for (const option of options.querySelectorAll(".repaint-mode-option")) {
        const isSelected = option.dataset.mode === mode;
        option.classList.toggle("is-selected", isSelected);
        option.setAttribute("aria-pressed", String(isSelected));
      }
    };

    const createOption = (mode, label, detail, badge = "") => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "repaint-mode-option";
      option.dataset.mode = mode;
      option.setAttribute("aria-pressed", "false");

      const optionHeader = document.createElement("span");
      optionHeader.className = "repaint-mode-option-header";
      const optionLabel = document.createElement("strong");
      optionLabel.textContent = label;
      optionHeader.appendChild(optionLabel);
      if (badge) {
        const badgeEl = document.createElement("span");
        badgeEl.className = "ui-badge repaint-mode-badge";
        badgeEl.textContent = badge;
        optionHeader.appendChild(badgeEl);
      }

      const optionDetail = document.createElement("span");
      optionDetail.className = "repaint-mode-option-detail";
      optionDetail.textContent = detail;
      option.append(optionHeader, optionDetail);
      option.addEventListener("click", () => selectMode(mode));
      return option;
    };

    const standardOption = createOption(
      "standard",
      "Xử lý mặc định",
      "Ưu tiên Smart Fill trên nền đồng nhất và chỉ chuyển sang LaMa khi vùng ảnh cần tái tạo phức tạp hơn.",
      "Khuyến nghị"
    );
    const lamaOption = createOption(
      "lama",
      "Tái inpaint bằng LaMa",
      "Bỏ qua Smart Fill và buộc LaMa tái tạo toàn bộ vùng đã đánh dấu. Phương án này có thể mất nhiều thời gian hơn."
    );
    options.append(standardOption, lamaOption);
    selectMode(selectedMode);

    const actions = document.createElement("div");
    actions.className = "repaint-mode-actions";
    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "ui-btn ui-btn-ghost repaint-mode-cancel";
    cancelBtn.textContent = "Hủy";
    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "ui-btn ui-btn-primary repaint-mode-confirm";
    confirmBtn.textContent = "Bắt đầu xử lý";
    actions.append(cancelBtn, confirmBtn);

    let closed = false;
    const close = (value) => {
      if (closed) return;
      closed = true;
      document.removeEventListener("keydown", onKeyDown);
      document.body.classList.remove("repaint-mode-open");
      backdrop.remove();
      resolve(value);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") close(null);
    };

    cancelBtn.addEventListener("click", () => close(null));
    confirmBtn.addEventListener("click", () => {
      close(selectedMode);
    });
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) close(null);
    });

    dialog.append(eyebrow, title, description, options, actions);
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);
    document.body.classList.add("repaint-mode-open");
    document.addEventListener("keydown", onKeyDown);
    standardOption.focus();
  });
}
window.chooseRepaintMode = chooseRepaintMode;
