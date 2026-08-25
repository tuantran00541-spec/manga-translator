(() => {
  function helpers() {
    return {
      parse: typeof window.parseApiResponse === "function"
        ? window.parseApiResponse
        : async (response) => response.json().catch(() => ({})),
      getError: typeof window.getErrorMessage === "function"
        ? window.getErrorMessage
        : (status, data) => data?.detail || `Máy chủ trả về ${status}`,
    };
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const { parse, getError } = helpers();
    const data = await parse(response);
    if (!response.ok) throw new Error(getError(response.status, data));
    return data;
  }

  function statusText(provider) {
    if (provider?.configured) {
      return `DeepSeek: Sẵn sàng · ${provider.model || "Vision Exp"}`;
    }
    if (provider?.source === "unavailable") return "DeepSeek: Kho bí mật chưa sẵn sàng";
    return "DeepSeek: Chưa cấu hình";
  }

  async function refresh(block) {
    const status = block.querySelector(".deepseek-qc-status");
    const clear = block.querySelector(".deepseek-key-clear-btn");
    try {
      const settings = await requestJson("/api/visual_qc/settings");
      const provider = settings?.providers?.deepseek || {};
      window.deepseekVisualQCConfigured = Boolean(provider.configured);
      if (status) {
        status.textContent = statusText(provider);
        status.classList.toggle("configured", Boolean(provider.configured));
      }
      if (clear) clear.disabled = !provider.configured || provider.source === "environment";
    } catch (err) {
      window.deepseekVisualQCConfigured = false;
      if (status) {
        status.textContent = "DeepSeek: Lỗi cấu hình";
        status.classList.remove("configured");
      }
      console.warn("DeepSeek QC settings check failed:", err);
    }
  }

  function buildControls(config) {
    const heading = document.createElement("strong");
    heading.className = "gemini-qc-privacy-note deepseek-qc-heading";
    heading.textContent = "DeepSeek Vision · kiểm tra toàn chương";

    const status = document.createElement("span");
    status.className = "gemini-qc-status deepseek-qc-status";
    status.textContent = "DeepSeek: Đang kiểm tra cấu hình…";

    const input = document.createElement("input");
    input.type = "password";
    input.className = "gemini-key-input deepseek-key-input";
    input.placeholder = "DeepSeek API key";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.setAttribute("aria-label", "DeepSeek API key");

    const save = document.createElement("button");
    save.type = "button";
    save.className = "gemini-key-save-btn deepseek-key-save-btn";
    save.textContent = "Lưu khóa DeepSeek";

    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "gemini-key-clear-btn deepseek-key-clear-btn";
    clear.textContent = "Xóa khóa DeepSeek";

    const privacy = document.createElement("span");
    privacy.className = "gemini-qc-privacy-note deepseek-qc-privacy-note";
    privacy.textContent = "Chỉ khi chọn DeepSeek cho kiểm tra toàn chương, contact sheet sẽ được gửi đến DeepSeek.";

    config.append(heading, status, input, save, clear, privacy);
    return { status, input, save, clear };
  }

  function bindConfig(config) {
    if (!config || config.dataset.deepseekQcBound === "1") return;
    config.dataset.deepseekQcBound = "1";
    const { input, save, clear } = buildControls(config);

    save.addEventListener("click", async () => {
      const apiKey = input.value.trim();
      if (!apiKey) {
        window.showToast?.("Nhập DeepSeek API key trước khi lưu.", "error");
        return;
      }
      save.disabled = true;
      save.textContent = "Đang lưu…";
      try {
        await requestJson("/api/visual_qc/deepseek/key", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: apiKey }),
        });
        input.value = "";
        window.showToast?.("Đã lưu DeepSeek API key trong kho bí mật của hệ điều hành.", "success");
      } catch (err) {
        window.showToast?.("Không thể lưu DeepSeek API key: " + err.message, "error");
      } finally {
        save.disabled = false;
        save.textContent = "Lưu khóa DeepSeek";
        await refresh(config);
      }
    });

    clear.addEventListener("click", async () => {
      clear.disabled = true;
      try {
        const data = await requestJson("/api/visual_qc/deepseek/key", { method: "DELETE" });
        const message = data.source === "environment"
          ? "DeepSeek API key đang đến từ biến môi trường; hãy xóa tại môi trường chạy ứng dụng."
          : "Đã xóa DeepSeek API key khỏi kho bí mật.";
        window.showToast?.(message, data.source === "environment" ? "info" : "success");
      } catch (err) {
        window.showToast?.("Không thể xóa DeepSeek API key: " + err.message, "error");
      } finally {
        await refresh(config);
      }
    });

    refresh(config);
  }

  function scan() {
    document.querySelectorAll(".gemini-qc-config").forEach(bindConfig);
  }

  const observer = new MutationObserver(scan);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", scan);
  scan();
})();
