// Shows how to install the browser extension and whether the installed copy needs a reload.
(() => {
  const guide = document.getElementById("extension-guide");
  if (!guide) return;
  const path = document.getElementById("extension-path");

  const copy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      window.showToast?.("Đã sao chép.", "success");
    } catch (_) {
      window.showToast?.("Không sao chép được, hãy chọn và chép tay.", "error");
    }
  };

  async function openGuide() {
    try {
      const info = await (await fetch("/api/extension")).json();
      path.textContent = info.path;
      document.getElementById("extension-outdated").hidden = !info.outdated;
    } catch (_) { /* the steps still help without the path */ }
    guide.showModal();
  }

  document.getElementById("extension-guide-open").addEventListener("click", openGuide);
  document.getElementById("extension-copy-page").addEventListener("click", () => copy("chrome://extensions"));
  document.getElementById("extension-copy-path").addEventListener("click", () => copy(path.textContent));
  document.getElementById("extension-open-folder").addEventListener("click", async () => {
    const reply = await fetch("/api/extension/open", { method: "POST" }).catch(() => null);
    if (!reply?.ok) window.showToast?.("Không mở được thư mục, hãy sao chép đường dẫn.", "error");
  });

  // A link the app cannot download is the moment the extension helps most.
  window.suggestExtension = () => {
    const hint = document.getElementById("extension-hint");
    hint.firstChild.textContent = "Không tải được từ liên kết? Trang có thể chặn tải tự động. ";
    hint.classList.add("attention");
  };
})();
