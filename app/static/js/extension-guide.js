(() => {
  const guide = document.getElementById("extension-guide");
  if (!guide) return;

  const copy = async (text) => {
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error("Clipboard API unavailable");
      }
      await navigator.clipboard.writeText(text);
      window.showToast?.("Đã sao chép.", "success");
    } catch (_) {
      window.showToast?.(
        "Không thể sao chép tự động, vui lòng chọn và sao chép thủ công.",
        "error",
      );
    }
  };

  async function openGuide() {
    if (!guide.open) {
      if (typeof guide.showModal === "function") {
        guide.showModal();
      } else {
        guide.setAttribute("open", "");
      }
    }
    try {
      const res = await fetch("/api/extension");
      if (!res.ok) return;
      const info = await res.json();
      const outdated = document.getElementById("extension-outdated");
      if (outdated) {
        outdated.hidden = !info?.outdated;
      }
    } catch (_) {
      // Non-fatal: modal still opens even if status check fails.
    }
  }

  document
    .getElementById("extension-guide-open")
    ?.addEventListener("click", openGuide);
  document
    .getElementById("extension-copy-page")
    ?.addEventListener("click", () => copy("chrome://extensions"));

  window.suggestExtension = () => {
    const hint = document.getElementById("extension-hint");
    if (!hint) return;
    const message =
      "Không tải được từ liên kết? Trang có thể chặn tải tự động. ";
    if (hint.firstChild && hint.firstChild.nodeType === Node.TEXT_NODE) {
      hint.firstChild.textContent = message;
    } else {
      hint.prepend(document.createTextNode(message));
    }
    hint.classList.add("attention");
  };
})();
