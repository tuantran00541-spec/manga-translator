(() => {
  const LABELS = { ja: "Tiếng Nhật", ch: "Tiếng Trung", korean: "Tiếng Hàn", en: "Tiếng Anh" };
  const ORIGINS = { auto: "tự nhận", site: "theo nguồn truyện", manual: "đã chọn" };
  const pending = new Map();
  const uncertain = new Set();

  const known = (lang) => (Object.prototype.hasOwnProperty.call(LABELS, lang) ? lang : null);

  window.SOURCE_LANG_LABELS = LABELS;
  window.sourceLangOriginLabel = (origin) => ORIGINS[origin] || "đã lưu";
  window.currentSourceLang = () => known(window.currentManifest?.source_lang);

  async function request(url, options) {
    const response = await fetch(url, options);
    const parse = window.parseApiResponse || (async (r) => r.json().catch(() => ({})));
    const data = await parse(response);
    if (!response.ok) {
      throw new Error(window.getErrorMessage?.(response.status, data) || data.detail || `HTTP ${response.status}`);
    }
    return data;
  }

  function apply(chapterId, data) {
    if (chapterId !== window.currentChapterId || !window.currentManifest) return;
    window.currentManifest.source_lang = data.source_lang;
    window.currentManifest.source_lang_origin = data.source_lang_origin;
    document.dispatchEvent(new CustomEvent("source-lang-changed", { detail: data }));
  }

  window.detectSourceLang = (force = false) => {
    const chapterId = window.currentChapterId;
    if (!chapterId) return Promise.resolve(null);
    if (!force && window.currentSourceLang()) return Promise.resolve(window.currentSourceLang());
    if (!force && uncertain.has(chapterId)) return Promise.resolve(null);
    if (pending.has(chapterId)) return pending.get(chapterId);
    const job = request(`/api/chapters/${encodeURIComponent(chapterId)}/language/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force }),
    }).then((data) => {
      const lang = known(data.source_lang);
      if (!lang && data.detection?.samples) uncertain.add(chapterId);
      apply(chapterId, data);
      return lang;
    }).finally(() => pending.delete(chapterId));
    pending.set(chapterId, job);
    return job;
  };

  window.resolveSourceLang = async () => {
    let lang = null;
    try {
      lang = await window.detectSourceLang();
    } catch (err) {
      console.warn("Source language detection failed", err);
    }
    if (lang) return lang;
    document.dispatchEvent(new CustomEvent("source-lang-needed"));
    throw new Error("Chưa nhận diện được ngôn ngữ gốc. Hãy chọn ở mục Ngôn ngữ gốc trong menu ⋯");
  };

  window.setSourceLang = async (lang) => {
    const chapterId = window.currentChapterId;
    if (!chapterId || !known(lang)) return null;
    const data = await request(`/api/chapters/${encodeURIComponent(chapterId)}/language`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_lang: lang }),
    });
    uncertain.delete(chapterId);
    apply(chapterId, data);
    return known(data.source_lang);
  };
})();
