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

    const aiConfig = document.createElement("div");
    aiConfig.className = "repaint-ai-config";
    aiConfig.hidden = true;

    const aiProviderLabel = document.createElement("label");
    aiProviderLabel.className = "repaint-ai-row";
    const aiProviderText = document.createElement("span");
    aiProviderText.textContent = "Nhà cung cấp";
    const aiProviderSelect = document.createElement("select");
    aiProviderSelect.className = "ui-select";
    aiProviderSelect.add(new Option("Tự chọn nhà cung cấp khả dụng", ""));
    aiProviderLabel.append(aiProviderText, aiProviderSelect);

    const aiModelLabel = document.createElement("label");
    aiModelLabel.className = "repaint-ai-row";
    const aiModelText = document.createElement("span");
    aiModelText.textContent = "Mô hình";
    const aiModelInput = document.createElement("input");
    aiModelInput.type = "text";
    aiModelInput.className = "ui-input";
    aiModelInput.value = "openai/gpt-image-2.6-flare";
    aiModelInput.placeholder = "openai/gpt-image-2.6-flare";
    aiModelLabel.append(aiModelText, aiModelInput);

    const aiQualityLabel = document.createElement("label");
    aiQualityLabel.className = "repaint-ai-row";
    const aiQualityText = document.createElement("span");
    aiQualityText.textContent = "Chất lượng";
    const aiQualitySelect = document.createElement("select");
    aiQualitySelect.className = "ui-select";
    [
      ["low", "Tiết kiệm (low — khuyến nghị)"],
      ["medium", "Cân bằng (medium)"],
      ["high", "Chi tiết cao (high)"],
    ].forEach(([val, lbl]) => aiQualitySelect.add(new Option(lbl, val)));
    aiQualitySelect.value = "low";
    aiQualityLabel.append(aiQualityText, aiQualitySelect);

    const aiHint = document.createElement("p");
    aiHint.className = "repaint-ai-hint";
    aiHint.hidden = true;

    aiConfig.append(aiProviderLabel, aiModelLabel, aiQualityLabel, aiHint);

    let aiSettingsLoaded = false;
    const loadAiConfig = async () => {
      if (aiSettingsLoaded) return;
      aiSettingsLoaded = true;
      try {
        const res = await fetch("/api/inpaint/settings");
        const data = await res.json().catch(() => ({}));
        if (!res.ok) return;
        const providers = Array.isArray(data.providers) ? data.providers : [];
        aiProviderSelect.replaceChildren();
        if (!providers.length) {
          aiProviderSelect.add(new Option("Chưa cấu hình nhà cung cấp OpenAI", ""));
          aiHint.textContent = "Hãy thêm nhà cung cấp giao thức OpenAI và khóa API trong Cài đặt, mục Mô hình AI.";
          aiHint.hidden = false;
        } else {
          aiProviderSelect.add(new Option(`Tự động (${providers[0].label})`, ""));
          for (const p of providers) {
            aiProviderSelect.add(new Option(p.label || p.id, p.id));
          }
          if (data.provider_id && providers.some((p) => p.id === data.provider_id)) {
            aiProviderSelect.value = data.provider_id;
          }
          aiHint.hidden = true;
        }
        if (data.model) aiModelInput.value = data.model;
        if (data.quality) aiQualitySelect.value = data.quality;
      } catch (_err) {
        // Keep defaults if settings endpoint is unavailable.
      }
    };

    const selectMode = (mode) => {
      selectedMode = mode;
      for (const option of options.querySelectorAll(".repaint-mode-option")) {
        const isSelected = option.dataset.mode === mode;
        option.classList.toggle("is-selected", isSelected);
        option.setAttribute("aria-pressed", String(isSelected));
      }
      aiConfig.hidden = mode !== "ai";
      if (mode === "ai") void loadAiConfig();
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
      "Vá nền chuyên sâu bằng LaMa",
      "Bỏ qua Smart Fill và buộc LaMa tái tạo toàn bộ vùng đã đánh dấu. Phương án này có thể mất nhiều thời gian hơn."
    );
    const aiOption = createOption(
      "ai",
      "Vẽ lại bằng AI (API)",
      "Gửi vùng đã tô sang mô hình ảnh API đã cấu hình. Tự gom nhiều vùng thành lưới động 2×2 / 3×3 (tối đa 9 vùng/lượt) để giảm chi phí và căn chỉnh sát nét vẽ gốc.",
      "Đẹp nhất cho SFX"
    );
    const restoreOption = createOption(
      "restore",
      "Khôi phục ảnh gốc",
      "Trả vùng đã tô về đúng ảnh gốc, dùng khi quá trình tẩy nền vô tình xóa mất chi tiết tranh. Vùng này được giữ nguyên qua các lần xử lý sau."
    );
    options.append(standardOption, lamaOption, aiOption, restoreOption);
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
      if (selectedMode === "ai") {
        const providerId = aiProviderSelect.value || "";
        const model = aiModelInput.value.trim() || "openai/gpt-image-2.6-flare";
        const quality = aiQualitySelect.value || "low";
        fetch("/api/inpaint/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider_id: providerId, model, quality }),
        }).catch(() => {});
        close({ mode: "ai", providerId, model, quality });
        return;
      }
      close(selectedMode);
    });
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) close(null);
    });

    dialog.append(eyebrow, title, description, options, aiConfig, actions);
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);
    document.body.classList.add("repaint-mode-open");
    document.addEventListener("keydown", onKeyDown);
    standardOption.focus();
  });
}
window.chooseRepaintMode = chooseRepaintMode;
