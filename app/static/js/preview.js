// preview.js - Quản lý Giao diện Xem trước & Chọn trang

function pageLabel(pages, pageIndex) {
  const page = pages[pageIndex];
  const total = pages.filter((p) => p.source_page === page.source_page).length;
  if (total <= 1) return "Trang " + (page.source_page + 1);
  return "Trang " + (page.source_page + 1) + " - Lát " + (page.slice_index + 1) + "/" + total;
}

function renderPreview() {
  const container = document.getElementById("page-view");
  if (!container) return;
  container.innerHTML = "";

  const toolbar = document.createElement("div");
  toolbar.id = "preview-toolbar";

  const processBtn = document.createElement("button");
  processBtn.textContent = "Xử lý các trang đã chọn (bỏ qua trang đã đánh dấu)";
  processBtn.addEventListener("click", processSelectedPages);
  toolbar.appendChild(processBtn);

  container.appendChild(toolbar);

  currentManifest.pages.forEach((page, pageIndex) => {
    const card = document.createElement("div");
    card.className = "preview-card";
    card.dataset.pageIndex = pageIndex;

    const img = document.createElement("img");
    img.src = page.original;
    card.appendChild(img);

    const label = document.createElement("div");
    label.className = "preview-label";
    label.textContent = pageLabel(currentManifest.pages, pageIndex);
    card.appendChild(label);

    const skipBtn = document.createElement("button");
    skipBtn.className = "skip-btn";
    skipBtn.textContent = page.skipped ? "Đã bỏ qua (bấm để hủy)" : "Bỏ qua trang này";
    if (page.skipped) card.classList.add("skipped");
    skipBtn.addEventListener("click", () => toggleSkip(pageIndex, card, skipBtn));
    card.appendChild(skipBtn);

    container.appendChild(card);
  });
}
