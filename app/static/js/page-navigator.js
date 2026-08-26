(() => {
  function clampIndex(value, total) {
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed) || total <= 0) return 0;
    return Math.max(0, Math.min(parsed, total - 1));
  }

  function createPageNavigator({
    items = [],
    activeIndex = 0,
    onSelect,
    title = "Trang",
    ariaLabel = "Điều hướng trang",
    busy = false,
  } = {}) {
    let currentItems = Array.isArray(items) ? items.slice() : [];
    let currentIndex = clampIndex(activeIndex, currentItems.length);
    let locked = Boolean(busy);

    const root = document.createElement("aside");
    root.className = "page-navigator";
    root.setAttribute("aria-label", ariaLabel);

    const header = document.createElement("div");
    header.className = "page-navigator-header";
    const heading = document.createElement("div");
    heading.className = "page-navigator-heading";
    const headingTitle = document.createElement("strong");
    headingTitle.textContent = title;
    const count = document.createElement("span");
    heading.append(headingTitle, count);

    const stepper = document.createElement("div");
    stepper.className = "page-navigator-stepper";
    const prev = document.createElement("button");
    prev.type = "button";
    prev.className = "page-navigator-step";
    prev.textContent = "←";
    prev.title = "Trang trước";
    prev.setAttribute("aria-label", "Trang trước");
    const jump = document.createElement("input");
    jump.type = "number";
    jump.min = "1";
    jump.className = "page-navigator-jump";
    jump.setAttribute("aria-label", "Nhảy đến trang");
    const next = document.createElement("button");
    next.type = "button";
    next.className = "page-navigator-step";
    next.textContent = "→";
    next.title = "Trang sau";
    next.setAttribute("aria-label", "Trang sau");
    stepper.append(prev, jump, next);
    header.append(heading, stepper);

    const list = document.createElement("div");
    list.className = "page-navigator-list";
    root.append(header, list);

    const select = (index) => {
      if (locked || !currentItems.length) return;
      const target = clampIndex(index, currentItems.length);
      if (target === currentIndex) return;
      if (typeof onSelect === "function") onSelect(target, currentItems[target]);
    };

    const render = () => {
      const total = currentItems.length;
      currentIndex = clampIndex(currentIndex, total);
      count.textContent = total ? `${currentIndex + 1} / ${total}` : "0 / 0";
      jump.max = String(Math.max(1, total));
      jump.value = total ? String(currentIndex + 1) : "";
      jump.disabled = locked || total === 0;
      prev.disabled = locked || currentIndex <= 0 || total === 0;
      next.disabled = locked || currentIndex >= total - 1 || total === 0;
      root.classList.toggle("page-navigator-busy", locked);

      list.replaceChildren();
      currentItems.forEach((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "page-navigator-item";
        button.dataset.pageNavigatorIndex = String(index);
        if (item.key !== undefined && item.key !== null) button.dataset.pageKey = String(item.key);
        if (item.state) button.dataset.state = item.state;
        button.classList.toggle("active", index === currentIndex);
        button.disabled = locked;
        if (index === currentIndex) button.setAttribute("aria-current", "page");
        button.title = item.label || `Trang ${index + 1}`;

        const thumb = document.createElement("span");
        thumb.className = "page-navigator-thumb";
        if (item.image) {
          const img = document.createElement("img");
          img.src = item.image;
          img.alt = "";
          img.loading = "lazy";
          thumb.appendChild(img);
        } else {
          thumb.textContent = String(index + 1).padStart(2, "0");
        }

        const text = document.createElement("span");
        text.className = "page-navigator-item-text";
        const label = document.createElement("strong");
        label.textContent = item.label || `Trang ${index + 1}`;
        text.appendChild(label);
        if (item.meta || item.stateLabel) {
          const meta = document.createElement("span");
          meta.textContent = [item.meta, item.stateLabel].filter(Boolean).join(" · ");
          text.appendChild(meta);
        }

        button.append(thumb, text);
        button.addEventListener("click", () => select(index));
        list.appendChild(button);
      });

      requestAnimationFrame(() => {
        list.querySelector('.page-navigator-item[aria-current="page"]')?.scrollIntoView({ block: "nearest" });
      });
    };

    const commitJump = () => {
      const target = parseInt(jump.value, 10);
      if (!Number.isFinite(target) || target < 1 || target > currentItems.length) {
        jump.value = currentItems.length ? String(currentIndex + 1) : "";
        return;
      }
      select(target - 1);
    };

    prev.addEventListener("click", () => select(currentIndex - 1));
    next.addEventListener("click", () => select(currentIndex + 1));
    jump.addEventListener("change", commitJump);
    jump.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitJump();
      }
    });

    render();

    return {
      element: root,
      setBusy(value) {
        locked = Boolean(value);
        render();
      },
      setActive(index) {
        currentIndex = clampIndex(index, currentItems.length);
        render();
      },
      setItems(nextItems, nextActiveIndex = currentIndex) {
        currentItems = Array.isArray(nextItems) ? nextItems.slice() : [];
        currentIndex = clampIndex(nextActiveIndex, currentItems.length);
        render();
      },
    };
  }

  window.createPageNavigator = createPageNavigator;
})();
