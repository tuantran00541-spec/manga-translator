function renderEditor() {
  const container = document.getElementById("page-view");
  if (!container) return;
  container.innerHTML = "";

  currentManifest.pages.forEach((page, pageIndex) => {
    if (page.skipped) return;

    const wrapper = document.createElement("div");
    wrapper.className = "page-block-wrapper";

    const label = document.createElement("div");
    label.className = "page-block-label";
    label.textContent = pageLabel(currentManifest.pages, pageIndex);
    wrapper.appendChild(label);

    const block = document.createElement("div");
    block.className = "page-block";
    block.dataset.pageIndex = pageIndex;
    wrapper.appendChild(block);

    const imgWrap = document.createElement("div");
    imgWrap.className = "page-image-wrap";

    const img = document.createElement("img");
    img.src = page.clean;
    imgWrap.appendChild(img);

    const panel = document.createElement("div");
    panel.className = "box-panel";

    img.onload = () => {
      const scaleX = img.clientWidth / img.naturalWidth;
      const scaleY = img.clientHeight / img.naturalHeight;

      page.boxes.forEach((box, boxIndex) => {
        if (box.removed) return;

        const overlay = document.createElement("div");
        overlay.className = "box-overlay";
        overlay.dataset.pageIndex = pageIndex;
        overlay.dataset.boxIndex = boxIndex;
        overlay.style.left = box.x1 * scaleX + "px";
        overlay.style.top = box.y1 * scaleY + "px";
        overlay.style.width = (box.x2 - box.x1) * scaleX + "px";
        overlay.style.height = (box.y2 - box.y1) * scaleY + "px";
        imgWrap.appendChild(overlay);

        const item = createBoxItem(pageIndex, boxIndex);
        panel.appendChild(item);
      });
    };

    const addBoxBtn = document.createElement("button");
    addBoxBtn.className = "add-box-btn";
    addBoxBtn.textContent = "Thêm vùng thoại bị bỏ sót";
    addBoxBtn.addEventListener("click", () => {
      imgWrap.classList.toggle("draw-mode");
      addBoxBtn.textContent = imgWrap.classList.contains("draw-mode")
        ? "Kéo chuột trên ảnh để khoanh vùng..."
        : "Thêm vùng thoại bị bỏ sót";
    });
    panel.appendChild(addBoxBtn);

    enableManualDraw(imgWrap, img, pageIndex, addBoxBtn);

    const renderBtn = document.createElement("button");
    renderBtn.className = "render-btn";
    renderBtn.textContent = "Chèn chữ vào ảnh";
    renderBtn.addEventListener("click", () => renderTranslations(pageIndex));
    panel.appendChild(renderBtn);

    block.appendChild(imgWrap);
    block.appendChild(panel);
    container.appendChild(wrapper);
  });
}

function enableManualDraw(imgWrap, img, pageIndex, addBoxBtn) {
  let drawing = false;
  let startX = 0, startY = 0;
  let drawBox = null;

  imgWrap.addEventListener("mousedown", (e) => {
    if (!imgWrap.classList.contains("draw-mode")) return;
    if (e.target !== img && e.target !== imgWrap) return;
    drawing = true;
    const rect = img.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;

    drawBox = document.createElement("div");
    drawBox.className = "draw-box";
    drawBox.style.left = startX + "px";
    drawBox.style.top = startY + "px";
    drawBox.style.width = "0px";
    drawBox.style.height = "0px";
    imgWrap.appendChild(drawBox);
    e.preventDefault();
  });

  imgWrap.addEventListener("mousemove", (e) => {
    if (!drawing || !drawBox) return;
    const rect = img.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    const left = Math.min(startX, curX);
    const top = Math.min(startY, curY);
    const w = Math.abs(curX - startX);
    const h = Math.abs(curY - startY);
    drawBox.style.left = left + "px";
    drawBox.style.top = top + "px";
    drawBox.style.width = w + "px";
    drawBox.style.height = h + "px";
  });

  imgWrap.addEventListener("mouseup", async (e) => {
    if (!drawing || !drawBox) return;
    drawing = false;
    imgWrap.classList.remove("draw-mode");
    addBoxBtn.textContent = "Thêm vùng thoại bị bỏ sót";

    const left = parseFloat(drawBox.style.left);
    const top = parseFloat(drawBox.style.top);
    const w = parseFloat(drawBox.style.width);
    const h = parseFloat(drawBox.style.height);
    drawBox.remove();
    if (w < 6 || h < 6) return;

    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;

    await submitManualBox(
      pageIndex,
      Math.round(left * scaleX),
      Math.round(top * scaleY),
      Math.round((left + w) * scaleX),
      Math.round((top + h) * scaleY)
    );
  });
}

function refreshPageAfterAddBox(pageIndex, newPage) {
  const block = document.querySelector(`.page-block[data-page-index="${pageIndex}"]`);
  if (!block) return;
  const imgWrap = block.querySelector(".page-image-wrap");
  const img = imgWrap.querySelector("img");
  const panel = block.querySelector(".box-panel");
  const renderBtn = panel.querySelector(".render-btn");

  img.src = newPage.clean + "?t=" + Date.now();

  const boxIndex = newPage.boxes.length - 1;
  const box = newPage.boxes[boxIndex];

  img.onload = () => {
    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;
    const overlay = document.createElement("div");
    overlay.className = "box-overlay";
    overlay.dataset.pageIndex = pageIndex;
    overlay.dataset.boxIndex = boxIndex;
    overlay.style.left = box.x1 * scaleX + "px";
    overlay.style.top = box.y1 * scaleY + "px";
    overlay.style.width = (box.x2 - box.x1) * scaleX + "px";
    overlay.style.height = (box.y2 - box.y1) * scaleY + "px";
    imgWrap.appendChild(overlay);
  };

  const item = createBoxItem(pageIndex, boxIndex);
  panel.insertBefore(item, renderBtn);
}

function showRenderResult(pageIndex, outputPath) {
  const panel = document.querySelector(
    `.page-block[data-page-index="${pageIndex}"] .box-panel`
  );
  if (!panel) return;

  let resultBox = panel.querySelector(".render-result");
  if (!resultBox) {
    resultBox = document.createElement("div");
    resultBox.className = "render-result";
    panel.appendChild(resultBox);
  }

  const cacheBust = "?t=" + Date.now();
  resultBox.innerHTML = "";

  const label = document.createElement("div");
  label.className = "render-result-label";
  label.textContent = "Kết quả:";
  resultBox.appendChild(label);

  const img = document.createElement("img");
  img.src = outputPath + cacheBust;
  resultBox.appendChild(img);

  const link = document.createElement("a");
  link.href = outputPath + cacheBust;
  link.download = outputPath.split("/").pop();
  link.className = "download-link";
  link.textContent = "Tải ảnh này về";
  resultBox.appendChild(link);
}
