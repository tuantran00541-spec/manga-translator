// Undo and redo for text objects: edits, moves, styles, creation and deletion.
(() => {
  const LIMIT = 100;
  const IDLE_MS = 700;
  const undoStack = [];
  const redoStack = [];
  const shadow = new Map();
  const open = new Map();
  let chapter = null;
  let applying = 0;
  let busy = false;

  const keyOf = (pageIndex, id) => `${pageIndex}:${id}`;
  const find = (pageIndex, id) => window.findTextObject?.(pageIndex, id) || null;
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  function snapshot(obj) {
    return JSON.parse(JSON.stringify({
      shape: obj.shape || "rectangle",
      region: obj.region || null,
      ocr_text: obj.ocr_text ?? "",
      translation: obj.translation ?? "",
      style: obj.style || window.DEFAULT_TEXT_OBJECT_STYLE,
      font_selection_mode: obj.font_selection_mode ?? null,
      font_match: obj.font_match ?? null,
      font_ai_id: obj.font_ai_id ?? null,
    }));
  }

  function syncChapter() {
    const current = window.currentChapterId || null;
    if (current === chapter) return;
    chapter = current;
    undoStack.length = 0;
    redoStack.length = 0;
    shadow.clear();
    open.forEach((txn) => clearTimeout(txn.timer));
    open.clear();
    notify();
  }

  function notify() {
    document.dispatchEvent(new CustomEvent("editor-history-changed", {
      detail: { canUndo: undoStack.length > 0, canRedo: redoStack.length > 0 },
    }));
  }

  function push(entry) {
    undoStack.push(entry);
    if (undoStack.length > LIMIT) undoStack.shift();
    redoStack.length = 0;
    notify();
  }

  // Remembers how an object looks before the user starts changing it.
  function watch(pageIndex, obj) {
    syncChapter();
    if (!obj?.id) return;
    const key = keyOf(pageIndex, obj.id);
    if (!open.has(key)) shadow.set(key, snapshot(obj));
  }

  function commit(key) {
    const txn = open.get(key);
    if (!txn) return;
    open.delete(key);
    clearTimeout(txn.timer);
    const obj = find(txn.pageIndex, txn.id);
    if (!obj) return;
    const after = snapshot(obj);
    shadow.set(key, after);
    if (!same(txn.before, after)) push({ type: "edit", items: [{ pageIndex: txn.pageIndex, id: txn.id, before: txn.before, after }] });
  }

  function commitAll() {
    [...open.keys()].forEach(commit);
  }

  // Called by every save path; groups a burst of changes to one object into one step.
  function touch(pageIndex, id) {
    if (applying) return;
    syncChapter();
    const key = keyOf(pageIndex, id);
    let txn = open.get(key);
    if (!txn) {
      const before = shadow.get(key);
      if (!before) return;
      txn = { pageIndex, id, before, timer: null };
      open.set(key, txn);
    }
    clearTimeout(txn.timer);
    txn.timer = setTimeout(() => commit(key), IDLE_MS);
  }

  // Runs one change over many objects as a single undo step.
  function batch(targets, mutate) {
    syncChapter();
    commitAll();
    const live = targets.map(({ pageIndex, id }) => ({ pageIndex, id, obj: find(pageIndex, id) })).filter((t) => t.obj);
    const befores = live.map((t) => snapshot(t.obj));
    applying += 1;
    try {
      mutate();
    } finally {
      applying -= 1;
    }
    const items = [];
    live.forEach((t, i) => {
      const after = snapshot(t.obj);
      shadow.set(keyOf(t.pageIndex, t.id), after);
      if (!same(befores[i], after)) items.push({ pageIndex: t.pageIndex, id: t.id, before: befores[i], after });
    });
    if (items.length) push({ type: "edit", items });
    return items.length;
  }

  function recordCreate(pageIndex, id) {
    if (applying) return;
    syncChapter();
    commitAll();
    push({ type: "create", pageIndex, id, snapshot: null });
  }

  function recordDelete(pageIndex, obj) {
    if (applying || !obj) return;
    syncChapter();
    commitAll();
    push({ type: "delete", pageIndex, id: obj.id, snapshot: snapshot(obj) });
  }

  function remap(pageIndex, oldId, newId) {
    [...undoStack, ...redoStack].forEach((entry) => {
      if (entry.type === "edit") {
        entry.items.forEach((item) => { if (item.pageIndex === pageIndex && item.id === oldId) item.id = newId; });
      } else if (entry.pageIndex === pageIndex && entry.id === oldId) {
        entry.id = newId;
      }
    });
  }

  function applySnapshot(pageIndex, id, snap) {
    const obj = find(pageIndex, id);
    if (!obj) return false;
    Object.assign(obj, JSON.parse(JSON.stringify(snap)));
    shadow.set(keyOf(pageIndex, id), snapshot(obj));
    window.scheduleTextObjectPersist?.(pageIndex, id);
    window.scheduleGeomPersist?.(pageIndex, id);
    window.syncOverlayForObject?.(pageIndex, id);
    return true;
  }

  async function run(entry, direction) {
    if (entry.type === "edit") {
      const items = direction === "undo" ? [...entry.items].reverse() : entry.items;
      let missing = 0;
      items.forEach((item) => {
        if (!applySnapshot(item.pageIndex, item.id, direction === "undo" ? item.before : item.after)) missing += 1;
      });
      if (missing) window.showToast?.("Một số vùng chữ không còn trên trang nên không hoàn tác được.", "info");
      const first = items[0];
      return first ? { pageIndex: first.pageIndex, id: first.id } : null;
    }
    const removes = (entry.type === "create") === (direction === "undo");
    if (removes) {
      const obj = find(entry.pageIndex, entry.id);
      if (obj) entry.snapshot = snapshot(obj);
      await window.deleteTextObject?.(entry.pageIndex, entry.id);
      return { pageIndex: entry.pageIndex, id: null };
    }
    const newId = await window.restoreTextObject?.(entry.pageIndex, entry.snapshot);
    if (!newId) throw new Error("Không khôi phục được vùng chữ");
    shadow.set(keyOf(entry.pageIndex, newId), JSON.parse(JSON.stringify(entry.snapshot)));
    remap(entry.pageIndex, entry.id, newId);
    entry.id = newId;
    return { pageIndex: entry.pageIndex, id: newId };
  }

  async function step(direction) {
    if (busy) return false;
    syncChapter();
    commitAll();
    const from = direction === "undo" ? undoStack : redoStack;
    const to = direction === "undo" ? redoStack : undoStack;
    const entry = from.pop();
    if (!entry) {
      window.showToast?.(direction === "undo" ? "Không còn thao tác để hoàn tác." : "Không còn thao tác để làm lại.", "info");
      return false;
    }
    busy = true;
    applying += 1;
    try {
      const focus = await run(entry, direction);
      to.push(entry);
      document.dispatchEvent(new CustomEvent("editor-history-applied", { detail: { focus, direction } }));
      return true;
    } catch (err) {
      from.push(entry);
      window.showToast?.((direction === "undo" ? "Không thể hoàn tác: " : "Không thể làm lại: ") + err.message, "error");
      return false;
    } finally {
      applying -= 1;
      busy = false;
      notify();
    }
  }

  window.editorHistory = {
    watch,
    touch,
    batch,
    recordCreate,
    recordDelete,
    undo: () => step("undo"),
    redo: () => step("redo"),
    canUndo: () => undoStack.length > 0,
    canRedo: () => redoStack.length > 0,
  };
})();
