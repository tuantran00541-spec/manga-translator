let _currentSaveStatus = "saved";
let _textSaving = 0;
let _textHasError = false;

function setSaveStatusContent(el, status) {
  const copy = {
    saved: ["check", "Đã lưu"],
    dirty: ["unsaved", "Chưa lưu"],
    saving: ["spinner", "Đang lưu…"],
    error: ["alert", "Lưu thất bại"],
  }[status];
  if (!copy) return;
  const [icon, text] = copy;
  el.replaceChildren(
    window.createUiIcon(icon, icon === "spinner" ? "ui-status-icon ui-icon-spinner" : "ui-status-icon"),
    document.createTextNode(text),
  );
}

function updateSaveStatus(status) {
  _currentSaveStatus = status;
  const statusEls = document.querySelectorAll(".editor-save-status");
  statusEls.forEach((el) => {
    el.className = `editor-save-status save-status-${status}`;
    if (status === "saved") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Tất cả thay đổi đã được lưu");
    } else if (status === "dirty") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Có thay đổi chưa lưu");
    } else if (status === "saving") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Đang lưu thay đổi");
    } else if (status === "error") {
      setSaveStatusContent(el, status);
      el.setAttribute("aria-label", "Lưu thay đổi thất bại");
    }
  });
}
window.updateSaveStatus = updateSaveStatus;

function refreshSaveStatus() {
  let status = "saved";
  const hasGeomDirty = typeof window.hasPendingGeom === "function" ? window.hasPendingGeom() : false;
  const isGeomSaving = typeof window.isGeomSaving === "function" ? window.isGeomSaving() : false;
  const hasGeomError = typeof window.hasGeomError === "function" ? window.hasGeomError() : false;

  if (_textHasError || hasGeomError) {
    status = "error";
  } else if (_textSaving > 0 || isGeomSaving) {
    status = "saving";
  } else if (_textDirty.size > 0 || hasGeomDirty) {
    status = "dirty";
  } else {
    status = "saved";
  }
  updateSaveStatus(status);
  return status;
}
window.refreshSaveStatus = refreshSaveStatus;

const _textDirty = new Map();
let _textTimer = null;
let _textPersistChain = Promise.resolve();

function _captureTextState(obj) {
  return {
    ocr_text: obj.ocr_text != null ? obj.ocr_text : "",
    translation: obj.translation != null ? obj.translation : "",
    style: obj.style
      ? JSON.parse(JSON.stringify(obj.style))
      : JSON.parse(JSON.stringify(DEFAULT_TEXT_OBJECT_STYLE)),
    font_selection_mode: obj.font_selection_mode || null,
    font_match: obj.font_match ? JSON.parse(JSON.stringify(obj.font_match)) : null,
    font_ai_id: obj.font_ai_id || null,
  };
}

function scheduleTextObjectPersist(pageIndex, id) {
  const obj = findTextObject(pageIndex, id);
  if (!obj) return;
  _textDirty.set(`${pageIndex}:${id}`, Object.assign({ pageIndex, id }, _captureTextState(obj)));
  _textHasError = false;
  refreshSaveStatus();
  clearTimeout(_textTimer);
  _textTimer = setTimeout(() => { flushTextObjectPersist().catch(() => {}); }, 800);
}
window.scheduleTextObjectPersist = scheduleTextObjectPersist;

async function _persistTextObjectsBulk(chapterId, items) {
  const resp = await fetch("/api/text_object/update_bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chapter_id: chapterId,
      updates: items.map((p) => ({
        page_index: p.pageIndex,
        id: p.id,
        ocr_text: p.ocr_text,
        translation: p.translation,
        style: p.style,
        font_selection_mode: p.font_selection_mode,
        font_match: p.font_match,
        font_ai_id: p.font_ai_id,
      })),
    }),
  });
  if (!resp.ok) {
    throw new Error(getErrorMessage(resp.status, await parseApiResponse(resp)));
  }
  return "bulk";
}

async function _persistTextObjectsIndividually(chapterId, items) {
  const failures = [];
  await Promise.all(items.map(async (p) => {
    const obj = findTextObject(p.pageIndex, p.id);
    if (!obj) return;
    try {
      await apiTextObject("update", {
        chapter_id: chapterId,
        page_index: p.pageIndex,
        id: p.id,
        ocr_text: p.ocr_text,
        translation: p.translation,
        style: p.style,
        font_selection_mode: p.font_selection_mode,
        font_match: p.font_match,
        font_ai_id: p.font_ai_id,
      });
    } catch (err) {
      failures.push(err);
      if (chapterId === currentChapterId) {
        _textHasError = true;
        _textDirty.set(`${p.pageIndex}:${p.id}`, Object.assign({ pageIndex: p.pageIndex, id: p.id }, _captureTextState(obj)));
      }
    }
  }));
  return failures;
}

async function _flushTextObjectPersistNow(pageIndex) {
  clearTimeout(_textTimer);
  _textTimer = null;
  if (_textDirty.size === 0) return;
  const items = [];
  _textDirty.forEach((v) => {
    if (pageIndex === undefined || v.pageIndex === pageIndex) items.push(v);
  });
  if (items.length === 0) return;
  const chapterId = currentChapterId;
  if (!chapterId) return;
  items.forEach((v) => _textDirty.delete(`${v.pageIndex}:${v.id}`));
  _textSaving += items.length;
  refreshSaveStatus();
  let failures = [];
  try {
    if (items.length === 1) {
      failures = await _persistTextObjectsIndividually(chapterId, items);
    } else {
      try {
        await _persistTextObjectsBulk(chapterId, items);
      } catch (bulkErr) {
        console.warn("Bulk text-object save failed; falling back per object:", bulkErr);
        failures = await _persistTextObjectsIndividually(chapterId, items);
      }
    }
  } finally {
    _textSaving -= items.length;
    refreshSaveStatus();
  }
  if (chapterId !== currentChapterId) return;
  if (failures.length) {
    showToast("Không lưu được nội dung: " + failures[0].message, "error");
    throw new Error("Không lưu được nội dung");
  }
}

function flushTextObjectPersist(pageIndex) {
  const job = _textPersistChain.catch(() => {}).then(() => _flushTextObjectPersistNow(pageIndex));
  _textPersistChain = job;
  return job;
}
window.flushTextObjectPersist = flushTextObjectPersist;

function cancelTextObjectPersist() {
  clearTimeout(_textTimer);
  _textTimer = null;
}

window.removePendingPersist = function removePendingPersist(pageIndex, id) {
  _textDirty.delete(`${pageIndex}:${id}`);
  if (typeof window.removePendingGeom === "function") window.removePendingGeom(pageIndex, id);
  refreshSaveStatus();
};

window.flushAllPendingPersists = async function flushAllPendingPersists(pageIndex) {
  const jobs = [flushTextObjectPersist(pageIndex)];
  if (typeof window.flushGeomPersist === "function") jobs.push(window.flushGeomPersist(pageIndex));
  await Promise.all(jobs);
};

window.cancelPendingPersist = function cancelPendingPersist() {
  cancelTextObjectPersist();
  _textDirty.clear();
  _textHasError = false;
  if (typeof window.cancelGeomPersist === "function") window.cancelGeomPersist();
  if (typeof window.clearPendingGeom === "function") window.clearPendingGeom();
  refreshSaveStatus();
};
