export const state = { variant: "clean", baseSyncOverlayForObject: null, tool: "select", zoom: 100, fitWidth: true, chapter: null, renderToken: 0 };


export const BRUSH_CHUNK_H = 4096;

export const ZOOM_STEPS = [25, 50, 75, 100, 125, 150, 200, 300, 400];

export const MIN_BOX = 10;

export const SHORTCUTS = { v: "select", r: "rectangle", o: "ellipse", b: "brush", e: "eraser", h: "hand", z: "zoom" };

export const snapshots = new Map();

export const autoSynced = new Set();

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
  snapshots.clear();
  autoSynced.clear();
}
