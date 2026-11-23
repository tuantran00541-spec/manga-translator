export const state = {
  variant: "clean",
  tool: "select",
  zoom: 100,
  fitWidth: true,
  showOverlays: true,
  livePreview: true,
  peekOriginal: false,
  brushRadius: 24,
  selectedKeys: new Set(),
  chapter: null,
  renderToken: 0,
};


export const BRUSH_CHUNK_H = 4096;

export const ZOOM_STEPS = [25, 50, 75, 100, 125, 150, 200, 300, 400];

export const MIN_BOX = 10;

export const SHORTCUTS = { v: "select", r: "rectangle", o: "ellipse", b: "brush", j: "remove", e: "eraser", i: "eyedropper", h: "hand", z: "zoom" };

export const selectionKey = (pageIndex, id) => `${Number(pageIndex)}:${String(id)}`;

export function parseSelectionKey(key) {
  const raw = String(key || "");
  const sep = raw.indexOf(":");
  if (sep < 0) return null;
  const pageIndex = Number(raw.slice(0, sep));
  const id = raw.slice(sep + 1);
  if (!Number.isInteger(pageIndex) || !id) return null;
  return { pageIndex, id };
}

const snapshots = new Map();

export const storeSnapshot = (key, value) => { snapshots.set(key, value); };

export const readSnapshot = (key) => snapshots.get(key);

export const deleteSnapshot = (key) => { snapshots.delete(key); };

export const hasSnapshotPrefix = (prefix) => [...snapshots.keys()].some((key) => key.startsWith(prefix));

const autoSynced = new Set();

export const isAutoSynced = (key) => autoSynced.has(key);

export const markAutoSynced = (key) => { autoSynced.add(key); };

export const chapterKey = () => String(window.currentChapterId || "");

export const snapshotKey = () => `${chapterKey()}:strip`;

export const livePage = (item) => window.currentManifest?.pages?.[Number(item?.canonicalIndex)] || null;

export function resetChapterState() {
  const key = chapterKey();
  if (state.chapter === key) return;
  state.chapter = key;
  state.variant = "clean";
  state.tool = "select";
  state.zoom = 100;
  state.fitWidth = true;
  state.selectedKeys.clear();
  snapshots.clear();
  autoSynced.clear();
}
