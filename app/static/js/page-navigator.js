(() => {
  "use strict";

  const CACHE_LIMIT = 8;
  const navigatorCache = new Map();

  function clampIndex(value, total) {
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed) || total <= 0) return 0;
    return Math.max(0, Math.min(parsed, total - 1));
  }

  function itemSignature(items) {
    return (items || []).map((item, index) => [
      item?.key ?? index,
      item?.label || "",
      item?.image || "",
      item?.meta || "",
      item?.state || "",
      item?.stateLabel || "",
    ].join("\u001f")).join("\u001e");
  }

  function cacheKey(title, ariaLabel) {
    const chapterId = window.currentChapterId || window.currentManifest?.chapter_id || "none";
    return `${chapterId}\u001f${title}\u001f${ariaLabel}`;
  }

  function evictCache() {
    while (navigatorCache.size > CACHE_LIMIT) {
      const [key, controller] = navigatorCache.entries().next().value || [];
      if (!key) break;
      navigatorCache.delete(key);
      controller?.dispose?.({ remove: false, fromCache: true });
    }
  }

  function buildNavigator({
    items = [],
    activeIndex = 0,
    onSelect,
    title = "Trang",
    ariaLabel = "Điều hướng trang",
    busy = false,
    cacheId = null,
  } = {}) {
    let currentItems = Array.isArray(items) ? items.slice() : [];
    let currentIndex = clampIndex(activeIndex, currentItems.length);
    let locked = Boolean(busy);
    let selecting = false;
    let selectHandler = typeof onSelect === "function" ? onSelect : null;
    let signature = itemSignature(currentItems);
    let disposed = false;
    let activeButton = null;
    const buttons = [];

    const root = document.createElement("aside");
    root.className = "page-navigator";
    root.setAttribute("aria-label", ariaLabel);
    root.dataset.navigatorCacheId = cacheId || "";

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
    prev.append(window.createUiIcon("chevron-left"));
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
    next.append(window.createUiIcon("chevron-right"));
    next.title = "Trang sau";
    next.setAttribute("aria-label", "Trang sau");
    stepper.append(prev, jump, next);
    header.append(heading, stepper);

    const list = document.createElement("div");
    list.className = "page-navigator-list";
    root.append(header, list);

    let thumbObserver = null;
    const makeObserver = () => {
      thumbObserver?.disconnect();
      thumbObserver = null;
      if (typeof window.IntersectionObserver === "undefined") return;
      thumbObserver = new IntersectionObserver((entries, observer) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const img = entry.target;
          const src = img.dataset.src;
          if (src) {
            img.src = src;
            img.removeAttribute("data-src");
          }
          observer.unobserve(img);
        }
      }, { root: list, rootMargin: "180px 0px" });
    };
    makeObserver();

    const updateControlsState = () => {
      const total = currentItems.length;
      currentIndex = clampIndex(currentIndex, total);
      const disabled = locked || selecting || total === 0;
      count.textContent = total ? `${currentIndex + 1} / ${total}` : "0 / 0";
      jump.max = String(Math.max(1, total));
      jump.value = total ? String(currentIndex + 1) : "";
      jump.disabled = disabled;
      prev.disabled = disabled || currentIndex <= 0;
      next.disabled = disabled || currentIndex >= total - 1;
      root.classList.toggle("page-navigator-busy", locked || selecting);
      root.setAttribute("aria-busy", String(locked || selecting));
    };

    const setButtonActive = (button, active) => {
      if (!button) return;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    };

    const updateActiveItem = (previousIndex, scrollActive = true) => {
      updateControlsState();
      const previous = Number.isInteger(previousIndex) ? buttons[previousIndex] : activeButton;
      if (previous && previous !== buttons[currentIndex]) setButtonActive(previous, false);
      activeButton = buttons[currentIndex] || null;
      setButtonActive(activeButton, true);
      if (scrollActive && activeButton) {
        requestAnimationFrame(() => {
          if (!disposed && activeButton?.isConnected) activeButton.scrollIntoView({ block: "nearest" });
        });
      }
    };

    const createItemButton = (item, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "page-navigator-item";
      button.dataset.pageNavigatorIndex = String(index);
      if (item?.key !== undefined && item?.key !== null) button.dataset.pageKey = String(item.key);
      if (item?.state) button.dataset.state = item.state;
      button.disabled = locked || selecting;
      button.setAttribute("aria-setsize", String(currentItems.length));
      button.setAttribute("aria-posinset", String(index + 1));
      button.title = item?.label || `Trang ${index + 1}`;

      const thumb = document.createElement("span");
      thumb.className = "page-navigator-thumb";
      if (item?.image) {
        const img = document.createElement("img");
        img.alt = "";
        img.decoding = "async";
        if (thumbObserver) {
          img.dataset.src = item.image;
          thumbObserver.observe(img);
        } else {
          img.src = item.image;
          img.loading = "lazy";
        }
        thumb.appendChild(img);
      } else {
        thumb.textContent = String(index + 1).padStart(2, "0");
      }

      const text = document.createElement("span");
      text.className = "page-navigator-item-text";
      const label = document.createElement("strong");
      label.textContent = item?.label || `Trang ${index + 1}`;
      text.appendChild(label);
      if (item?.meta || item?.stateLabel) {
        const meta = document.createElement("span");
        meta.textContent = [item.meta, item.stateLabel].filter(Boolean).join(" · ");
        text.appendChild(meta);
      }
      button.append(thumb, text);
      button.addEventListener("click", () => { void select(index); });
      return button;
    };

    const renderList = ({ preserveScroll = false, preserveFocus = false } = {}) => {
      const oldScroll = list.scrollTop;
      const focused = preserveFocus && root.contains(document.activeElement)
        ? Number(document.activeElement?.dataset?.pageNavigatorIndex)
        : null;
      thumbObserver?.disconnect();
      makeObserver();
      buttons.length = 0;
      list.replaceChildren();
      currentItems.forEach((item, index) => {
        const button = createItemButton(item, index);
        buttons.push(button);
        list.appendChild(button);
      });
      activeButton = buttons[currentIndex] || null;
      setButtonActive(activeButton, true);
      updateControlsState();
      if (preserveScroll) list.scrollTop = oldScroll;
      if (Number.isInteger(focused) && buttons[focused]) requestAnimationFrame(() => buttons[focused]?.focus());
    };

    async function select(index) {
      if (disposed || locked || selecting || !currentItems.length) return false;
      const target = clampIndex(index, currentItems.length);
      if (target === currentIndex) return true;
      const previous = currentIndex;
      currentIndex = target;
      updateActiveItem(previous, true);
      if (!selectHandler) return true;
      try {
        let result = selectHandler(target, currentItems[target]);
        if (result && typeof result.then === "function") {
          selecting = true;
          buttons.forEach((button) => { button.disabled = true; });
          updateControlsState();
          result = await result;
        }
        if (result === false) {
          const failedTarget = currentIndex;
          currentIndex = previous;
          updateActiveItem(failedTarget, true);
          return false;
        }
        return true;
      } catch (err) {
        console.warn("Navigator onSelect warning:", err);
        const failedTarget = currentIndex;
        currentIndex = previous;
        updateActiveItem(failedTarget, true);
        return false;
      } finally {
        selecting = false;
        buttons.forEach((button) => { button.disabled = locked; });
        updateControlsState();
      }
    }

    const commitJump = () => {
      const target = parseInt(jump.value, 10);
      if (!Number.isFinite(target) || target < 1 || target > currentItems.length) {
        jump.value = currentItems.length ? String(currentIndex + 1) : "";
        return;
      }
      void select(target - 1);
    };

    prev.addEventListener("click", () => { void select(currentIndex - 1); });
    next.addEventListener("click", () => { void select(currentIndex + 1); });
    jump.addEventListener("change", commitJump);
    jump.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commitJump();
      }
    });

    const controller = {
      element: root,
      select,
      selectByKey(key) {
        const index = currentItems.findIndex((item) => String(item?.key) === String(key));
        if (index >= 0) return select(index);
        return Promise.resolve(false);
      },
      setBusy(value) {
        locked = Boolean(value);
        buttons.forEach((button) => { button.disabled = locked || selecting; });
        updateControlsState();
      },
      setActive(index, { scroll = true } = {}) {
        const target = clampIndex(index, currentItems.length);
        if (target === currentIndex) {
          updateControlsState();
          return;
        }
        const previous = currentIndex;
        currentIndex = target;
        updateActiveItem(previous, scroll);
      },
      setItems(nextItems, nextActiveIndex = currentIndex) {
        const normalized = Array.isArray(nextItems) ? nextItems.slice() : [];
        const nextSignature = itemSignature(normalized);
        const previous = currentIndex;
        currentItems = normalized;
        currentIndex = clampIndex(nextActiveIndex, currentItems.length);
        if (nextSignature === signature && buttons.length === currentItems.length) {
          signature = nextSignature;
          updateActiveItem(previous, false);
          return;
        }
        signature = nextSignature;
        renderList({ preserveScroll: true, preserveFocus: true });
      },
      setOnSelect(handler) {
        selectHandler = typeof handler === "function" ? handler : null;
      },
      reconfigure(options = {}) {
        if (options.onSelect !== undefined) controller.setOnSelect(options.onSelect);
        if (options.busy !== undefined) locked = Boolean(options.busy);
        headingTitle.textContent = options.title || headingTitle.textContent;
        root.setAttribute("aria-label", options.ariaLabel || root.getAttribute("aria-label") || "Điều hướng trang");
        controller.setItems(options.items || currentItems, options.activeIndex ?? currentIndex);
        controller.setBusy(locked);
      },
      dispose({ remove = true, fromCache = false } = {}) {
        if (disposed) return;
        disposed = true;
        thumbObserver?.disconnect();
        thumbObserver = null;
        if (remove) root.remove();
        if (!fromCache && cacheId && navigatorCache.get(cacheId) === controller) navigatorCache.delete(cacheId);
        buttons.length = 0;
        activeButton = null;
        selectHandler = null;
      },
      get activeIndex() { return currentIndex; },
      get itemCount() { return currentItems.length; },
    };

    renderList();
    return controller;
  }

  function createPageNavigator(options = {}) {
    const title = options.title || "Trang";
    const ariaLabel = options.ariaLabel || "Điều hướng trang";
    const key = cacheKey(title, ariaLabel);
    const cached = navigatorCache.get(key);
    if (cached) {
      navigatorCache.delete(key);
      navigatorCache.set(key, cached);
      cached.reconfigure({ ...options, title, ariaLabel });
      return cached;
    }
    const controller = buildNavigator({ ...options, title, ariaLabel, cacheId: key });
    navigatorCache.set(key, controller);
    evictCache();
    return controller;
  }

  window.disposePageNavigators = function disposePageNavigators(chapterId = null) {
    for (const [key, controller] of [...navigatorCache.entries()]) {
      if (chapterId && !key.startsWith(`${chapterId}\u001f`)) continue;
      navigatorCache.delete(key);
      controller.dispose({ remove: false, fromCache: true });
    }
  };
  window.pageNavigatorDebug = {
    cache: navigatorCache,
    signature: itemSignature,
    clear: window.disposePageNavigators,
  };
  window.createPageNavigator = createPageNavigator;
})();
