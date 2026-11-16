
(() => {
  const guide = document.getElementById("extension-guide");
  if (!guide) return;

  const copy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      window.showToast?.("Đã sao chép.", "success");
    } catch (_) {
      window.showToast?.("Không thể sao chép tự động, vui lòng chọn và sao chép thủ công.", "error");
    }
  };

  async function openGuide() {
    try {
      const info = await (await fetch("/api/extension")).json();
      document.getElementById("extension-outdated").hidden = !info.outdated;
    } catch (_) { }
    guide.showModal();
  }

  document.getElementById("extension-guide-open").addEventListener("click", openGuide);
  document.getElementById("extension-copy-page").addEventListener("click", () => copy("chrome://extensions"));

  window.suggestExtension = () => {
    const hint = document.getElementById("extension-hint");
    hint.firstChild.textContent = "Không tải được từ liên kết? Trang có thể chặn tải tự động. ";
    hint.classList.add("attention");
  };
})();
