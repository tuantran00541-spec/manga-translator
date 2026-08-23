#!/usr/bin/env python3
"""
Fast local Ground-Truth reviewer for the detect -> box -> mask benchmark.

Dependencies:
    Python 3.10+
    Pillow

Tkinter is normally included with the official Windows Python installer.

This tool DOES NOT run detector models and DOES NOT modify production code.
It reviews the existing raw bubble/text candidates already stored in the GT
template, lets the user accept/reject/flag them, and lets the user draw
missed GT boxes. The result is written back to the same JSON schema.

Typical Windows usage:
    python tools/detect_box_mask_reviewer.py ^
        --images bench\images ^
        --gt bench\speech_gt_template_v1.json

Keyboard:
    1 = target
    2 = non-target
    3 = uncertain
    Tab = next candidate
    Shift+Tab = previous candidate
    B = bubble candidate mode
    T = text candidate mode
    M = draw missed bubble
    N = draw missed text
    A = assign selected text to a bubble by clicking the bubble
    Delete = remove selected missed box
    S = save
    Left/Right = previous/next image
    Esc = cancel drawing mode

Mouse:
    Left click candidate = select it
    Mouse wheel = zoom
    Middle click/drag = pan
    In draw mode: left-drag a rectangle

The reviewer autosaves after every label/add/remove action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageTk, ImageDraw, ImageOps
except ImportError:
    print("ERROR: Pillow is required. Install with: python -m pip install pillow")
    raise SystemExit(1)

import tkinter as tk
from tkinter import filedialog, messagebox


SCHEMA_VERSION = "1.0.0"
LABELS = ("target_speech_bubble", "non_target", "uncertain")
TEXT_LABELS = ("target_speech_text", "non_target", "uncertain")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(raw_path: str, images_dir: Path) -> Path | None:
    p = Path(raw_path)
    candidates = [
        p,
        images_dir / p.name,
        images_dir / p.stem,
        images_dir / f"{p.stem}.webp",
        images_dir / f"{p.stem}.png",
        images_dir / f"{p.stem}.jpg",
        images_dir / f"{p.stem}.jpeg",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> list[int]:
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    return [
        max(0, min(w, x1)),
        max(0, min(h, y1)),
        max(0, min(w, x2)),
        max(0, min(h, y2)),
    ]


class Reviewer:
    def __init__(self, root: tk.Tk, gt_path: Path, images_dir: Path, start_index: int = 0):
        self.root = root
        self.gt_path = gt_path
        self.images_dir = images_dir

        with gt_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

        if self.data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={self.data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )

        self.items: list[dict[str, Any]] = self.data.get("images", [])
        if not self.items:
            raise ValueError("GT file contains no images")

        self.current = max(0, min(start_index, len(self.items) - 1))
        self.mode = "bubble"
        self.selected_index: int | None = None
        self.selected_kind: str | None = None
        self.draw_mode: str | None = None
        self.assign_mode = False

        self.zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 8.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_start = None
        self._draw_start = None
        self._photo = None
        self._display_image = None
        self._image = None

        self._build_ui()
        self.load_image()
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.root.title("Detect → Box → Mask Ground-Truth Reviewer")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 700)

        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        self.image_label = tk.Label(top, text="")
        self.image_label.pack(side="left")

        self.progress_label = tk.Label(top, text="", font=("Segoe UI", 10, "bold"))
        self.progress_label.pack(side="left", padx=18)

        self.mode_label = tk.Label(top, text="", font=("Segoe UI", 10, "bold"))
        self.mode_label.pack(side="left", padx=18)

        self.status_label = tk.Label(top, text="", anchor="w")
        self.status_label.pack(side="left", padx=18, fill="x", expand=True)

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(body, bg="#222222", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        side = tk.Frame(body, width=330)
        side.pack(side="right", fill="y", padx=8)
        side.pack_propagate(False)

        tk.Label(side, text="Candidate review", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(4, 6)
        )

        self.listbox = tk.Listbox(side, width=45, height=26, exportselection=False)
        self.listbox.pack(fill="both", expand=False)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        controls = tk.Frame(side)
        controls.pack(fill="x", pady=8)

        for label, cmd in [
            ("1 Target", lambda: self.set_label("target")),
            ("2 Non-target", lambda: self.set_label("non_target")),
            ("3 Uncertain", lambda: self.set_label("uncertain")),
        ]:
            tk.Button(controls, text=label, command=cmd).pack(fill="x", pady=2)

        tk.Button(controls, text="Assign selected text → bubble", command=self.start_assign).pack(
            fill="x", pady=(7, 2)
        )
        tk.Button(controls, text="Draw missed bubble (M)", command=lambda: self.start_draw("bubble")).pack(
            fill="x", pady=2
        )
        tk.Button(controls, text="Draw missed text (N)", command=lambda: self.start_draw("text")).pack(
            fill="x", pady=2
        )
        tk.Button(controls, text="Delete selected missed box", command=self.delete_selected).pack(
            fill="x", pady=2
        )
        tk.Button(controls, text="Save (S)", command=self.save).pack(fill="x", pady=(8, 2))

        nav = tk.Frame(side)
        nav.pack(fill="x", pady=8)
        tk.Button(nav, text="← Previous", command=self.prev_image).pack(side="left", expand=True, fill="x")
        tk.Button(nav, text="Next →", command=self.next_image).pack(side="left", expand=True, fill="x", padx=(5, 0))

        tk.Label(
            side,
            text=(
                "Shortcuts\n"
                "1/2/3 label candidate\n"
                "B/T switch candidate list\n"
                "M/N draw missed GT box\n"
                "A assign text → bubble\n"
                "Tab / Shift+Tab next/prev candidate\n"
                "S save\n"
                "←/→ image\n"
                "Wheel zoom • middle-drag pan\n"
                "Esc cancel draw/assign"
            ),
            justify="left",
            anchor="nw",
        ).pack(fill="x", pady=8)

        self.canvas.bind("<Button-1>", self.on_canvas_left)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.15))
        self.canvas.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1 / 1.15))
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

    def current_item(self) -> dict[str, Any]:
        return self.items[self.current]

    def current_path(self) -> Path | None:
        return resolve_image_path(self.current_item().get("image", {}).get("path", ""), self.images_dir)

    def current_candidates(self) -> list[tuple[str, int, dict[str, Any]]]:
        item = self.current_item()
        key = "raw_bubble_candidates" if self.mode == "bubble" else "raw_text_candidates"
        return [(self.mode, i, x) for i, x in enumerate(item.get(key, []))]

    def load_image(self):
        self.selected_index = None
        self.selected_kind = None
        self.draw_mode = None
        self.assign_mode = False
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        path = self.current_path()
        if path is None:
            self._image = None
            self.status_label.config(text="IMAGE MISSING — fix --images path")
            self.redraw()
            self.refresh_list()
            return

        try:
            self._image = Image.open(path).convert("RGB")
        except Exception as exc:
            self._image = None
            self.status_label.config(text=f"Cannot open image: {exc}")
            self.redraw()
            self.refresh_list()
            return

        # Update image metadata without changing the benchmark's labels.
        arr_shape = [self._image.height, self._image.width, 3]
        meta = self.current_item().setdefault("image", {})
        meta["path"] = str(path)
        meta["shape"] = arr_shape
        try:
            meta["sha256"] = sha256_file(path)
        except Exception:
            pass

        self.refresh_list()
        self.redraw()

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        candidates = self.current_candidates()
        for kind, i, c in candidates:
            label = c.get("label", "uncertain")
            conf = c.get("confidence")
            if conf is None:
                conf_s = ""
            else:
                conf_s = f"  conf={float(conf):.3f}"
            self.listbox.insert(tk.END, f"{i:03d}  {label:20s}{conf_s}")

        if self.selected_index is not None and self.selected_index < len(candidates):
            self.listbox.selection_set(self.selected_index)
            self.listbox.see(self.selected_index)

        self.progress_label.config(
            text=f"Image {self.current + 1}/{len(self.items)}  "
                 f"({Path(self.current_path()).name if self.current_path() else 'MISSING'})"
        )
        self.mode_label.config(text=f"MODE: {self.mode.upper()}")

        item = self.current_item()
        b = len(item.get("raw_bubble_candidates", []))
        t = len(item.get("raw_text_candidates", []))
        mb = len(item.get("missed_gt_bubbles", []))
        mt = len(item.get("missed_gt_text", []))
        self.status_label.config(
            text=f"raw bubbles={b}, raw text={t}, missed GT bubbles={mb}, missed GT text={mt}"
        )

    def redraw(self):
        self.canvas.delete("all")
        if self._image is None:
            self.canvas.create_text(
                300, 200, text="Image not found", fill="white", font=("Segoe UI", 18)
            )
            return

        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())

        base_scale = min(cw / self._image.width, ch / self._image.height)
        scale = max(self.min_zoom, min(self.max_zoom, base_scale * self.zoom))
        dw = max(1, int(self._image.width * scale))
        dh = max(1, int(self._image.height * scale))

        self._display_image = self._image.resize((dw, dh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(self._display_image)

        x0 = (cw - dw) / 2 + self.pan_x
        y0 = (ch - dh) / 2 + self.pan_y
        self._origin = (x0, y0, scale)

        self.canvas.create_image(x0, y0, anchor="nw", image=self._photo)

        # Raw candidates
        for kind, i, c in self.current_candidates():
            color = "#55ff55" if kind == "bubble" else "#55aaff"
            if c.get("label") == "non_target":
                color = "#ff5555"
            elif c.get("label") == "uncertain":
                color = "#ffff55"

            self.draw_box(c, color, width=2, tag=f"{kind}_{i}")

            if i == self.selected_index and kind == self.selected_kind:
                self.draw_box(c, "#ffffff", width=4, tag="selected")

        # Missed GT boxes
        item = self.current_item()
        for i, c in enumerate(item.get("missed_gt_bubbles", [])):
            self.draw_box(c, "#ff00ff", width=4, tag=f"missed_bubble_{i}")
        for i, c in enumerate(item.get("missed_gt_text", [])):
            self.draw_box(c, "#00ffff", width=4, tag=f"missed_text_{i}")

        # Draw mode hint
        if self.draw_mode:
            self.canvas.create_text(
                12, 12, anchor="nw",
                text=f"DRAW {self.draw_mode.upper()}: drag a rectangle • Esc cancels",
                fill="white", font=("Segoe UI", 11, "bold")
            )
        elif self.assign_mode:
            self.canvas.create_text(
                12, 12, anchor="nw",
                text="ASSIGN: click the target bubble • Esc cancels",
                fill="white", font=("Segoe UI", 11, "bold")
            )

    def draw_box(self, c: dict[str, Any], color: str, width: int = 2, tag: str | None = None):
        if not hasattr(self, "_origin"):
            return
        x0, y0, scale = self._origin
        x1 = x0 + float(c["x1"]) * scale
        y1 = y0 + float(c["y1"]) * scale
        x2 = x0 + float(c["x2"]) * scale
        y2 = y0 + float(c["y2"]) * scale
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=tag)

    def canvas_to_image(self, x: float, y: float) -> tuple[int, int]:
        x0, y0, scale = self._origin
        ix = round((x - x0) / scale)
        iy = round((y - y0) / scale)
        return ix, iy

    def on_canvas_left(self, event):
        if self._image is None:
            return

        if self.draw_mode:
            self._draw_start = (event.x, event.y)
            self.canvas.bind("<B1-Motion>", self.on_draw_move)
            self.canvas.bind("<ButtonRelease-1>", self.on_draw_release)
            return

        if self.assign_mode:
            ix, iy = self.canvas_to_image(event.x, event.y)
            if self.mode != "text" or self.selected_index is None:
                self.assign_mode = False
                self.redraw()
                return
            item = self.current_item()
            bubbles = item.get("raw_bubble_candidates", [])
            hits = []
            for bi, b in enumerate(bubbles):
                if b["x1"] <= ix <= b["x2"] and b["y1"] <= iy <= b["y2"]:
                    hits.append(bi)
            if len(hits) == 1:
                t = item["raw_text_candidates"][self.selected_index]
                t["assigned_bubble_indices"] = hits
                self.status_label.config(text=f"Assigned text {self.selected_index} → bubble {hits[0]}")
            elif len(hits) > 1:
                # Preserve ambiguity rather than guessing.
                item["raw_text_candidates"][self.selected_index]["assigned_bubble_indices"] = hits
                self.status_label.config(text=f"Assigned text {self.selected_index} → ambiguous bubbles {hits}")
            else:
                item["raw_text_candidates"][self.selected_index]["assigned_bubble_indices"] = []
                self.status_label.config(text="Text assigned to no bubble")
            self.assign_mode = False
            self.save(silent=True)
            self.redraw()
            return

        # Candidate hit-test
        ix, iy = self.canvas_to_image(event.x, event.y)
        candidates = self.current_candidates()
        hits = []
        for kind, i, c in candidates:
            if c["x1"] <= ix <= c["x2"] and c["y1"] <= iy <= c["y2"]:
                hits.append((kind, i, c))
        if hits:
            kind, i, _ = hits[-1]
            self.mode = kind
            self.selected_kind = kind
            self.selected_index = i
            self.refresh_list()
            self.redraw()

    def on_draw_move(self, event):
        if not self._draw_start:
            return
        self.redraw()
        x1, y1 = self._draw_start
        self.canvas.create_rectangle(x1, y1, event.x, event.y, outline="#ffffff", width=3, dash=(6, 4))

    def on_draw_release(self, event):
        if not self._draw_start or self._image is None:
            return
        x1, y1 = self._draw_start
        self._draw_start = None
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

        ix1, iy1 = self.canvas_to_image(x1, y1)
        ix2, iy2 = self.canvas_to_image(event.x, event.y)
        box = clamp_box(ix1, iy1, ix2, iy2, self._image.width, self._image.height)
        if box[2] - box[0] < 3 or box[3] - box[1] < 3:
            self.status_label.config(text="Ignored tiny box")
            self.redraw()
            return

        item = self.current_item()
        if self.draw_mode == "bubble":
            item.setdefault("missed_gt_bubbles", []).append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "label": "target_speech_bubble",
                "source": "manual",
            })
        else:
            item.setdefault("missed_gt_text", []).append({
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "label": "target_speech_text",
                "source": "manual",
                "assigned_bubble_indices": [],
            })

        self.draw_mode = None
        self.save(silent=True)
        self.refresh_list()
        self.redraw()

    def set_label(self, which: str):
        if self.selected_index is None or self.selected_kind is None:
            return
        key = "raw_bubble_candidates" if self.selected_kind == "bubble" else "raw_text_candidates"
        arr = self.current_item().get(key, [])
        if self.selected_index >= len(arr):
            return
        arr[self.selected_index]["label"] = {
            "target": "target_speech_bubble" if self.selected_kind == "bubble" else "target_speech_text",
            "non_target": "non_target",
            "uncertain": "uncertain",
        }[which]
        self.save(silent=True)
        self.refresh_list()
        self.redraw()

        # Move to next candidate for rapid review.
        if self.selected_index + 1 < len(arr):
            self.selected_index += 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.selected_index)
            self.listbox.see(self.selected_index)
            self.redraw()

    def start_assign(self):
        if self.mode != "text" or self.selected_index is None:
            self.status_label.config(text="Select a raw text candidate first")
            return
        self.assign_mode = True
        self.draw_mode = None
        self.redraw()

    def start_draw(self, kind: str):
        self.draw_mode = kind
        self.assign_mode = False
        self.redraw()

    def delete_selected(self):
        if self.selected_kind not in ("missed_bubble", "missed_text") or self.selected_index is None:
            self.status_label.config(text="Click a magenta/cyan missed-GT box first")
            return
        key = "missed_gt_bubbles" if self.selected_kind == "missed_bubble" else "missed_gt_text"
        arr = self.current_item().get(key, [])
        if 0 <= self.selected_index < len(arr):
            arr.pop(self.selected_index)
        self.selected_kind = None
        self.selected_index = None
        self.save(silent=True)
        self.refresh_list()
        self.redraw()

    def on_list_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.selected_index = sel[0]
        self.selected_kind = self.mode
        self.redraw()

    def switch_mode(self, mode: str):
        self.mode = mode
        self.selected_kind = mode
        self.selected_index = 0 if self.current_candidates() else None
        self.refresh_list()
        self.redraw()

    def next_candidate(self, delta: int):
        candidates = self.current_candidates()
        if not candidates:
            return
        if self.selected_index is None:
            self.selected_index = 0
        else:
            self.selected_index = (self.selected_index + delta) % len(candidates)
        self.selected_kind = self.mode
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.selected_index)
        self.listbox.see(self.selected_index)
        self.redraw()

    def prev_image(self):
        if self.current > 0:
            self.current -= 1
            self.load_image()

    def next_image(self):
        if self.current + 1 < len(self.items):
            self.current += 1
            self.load_image()

    def zoom_at(self, cx: float, cy: float, factor: float):
        if self._image is None or not hasattr(self, "_origin"):
            return
        old_scale = self._origin[2]
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        new_scale = min(
            max(1e-6, min(self.canvas.winfo_width() / self._image.width,
                          self.canvas.winfo_height() / self._image.height)),
            1e9
        ) * new_zoom
        if new_scale <= 0:
            return

        # Keep the image coordinate under the cursor stable.
        x0, y0, _ = self._origin
        ix = (cx - x0) / old_scale
        iy = (cy - y0) / old_scale

        self.zoom = new_zoom
        self.redraw()

        nx0, ny0, ns = self._origin
        target_x0 = cx - ix * ns
        target_y0 = cy - iy * ns
        self.pan_x += target_x0 - nx0
        self.pan_y += target_y0 - ny0
        self.redraw()

    def on_wheel(self, event):
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.zoom_at(event.x, event.y, factor)

    def on_pan_start(self, event):
        self._drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def on_pan_move(self, event):
        if not self._drag_start:
            return
        sx, sy, px, py = self._drag_start
        self.pan_x = px + event.x - sx
        self.pan_y = py + event.y - sy
        self.redraw()

    def save(self, silent: bool = False):
        self.data["schema_version"] = SCHEMA_VERSION
        self.data["review_tool"] = "detect_box_mask_reviewer"
        self.data["review_tool_version"] = "1.0.0"
        tmp = self.gt_path.with_suffix(self.gt_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, self.gt_path)
        if not silent:
            self.status_label.config(text=f"Saved: {self.gt_path}")

    def on_key(self, event):
        key = event.keysym.lower()
        char = event.char.lower() if event.char else ""

        if key == "escape":
            self.draw_mode = None
            self.assign_mode = False
            self.redraw()
            return

        if char == "1":
            self.set_label("target")
        elif char == "2":
            self.set_label("non_target")
        elif char == "3":
            self.set_label("uncertain")
        elif char == "b":
            self.switch_mode("bubble")
        elif char == "t":
            self.switch_mode("text")
        elif char == "m":
            self.start_draw("bubble")
        elif char == "n":
            self.start_draw("text")
        elif char == "a":
            self.start_assign()
        elif char == "s":
            self.save()
        elif key == "tab":
            self.next_candidate(-1 if event.state & 0x1 else 1)
        elif key == "left":
            self.prev_image()
        elif key == "right":
            self.next_image()
        elif key == "delete":
            self.delete_selected()

    def on_close(self):
        try:
            self.save(silent=True)
        finally:
            self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Directory containing manga images")
    ap.add_argument("--gt", required=True, help="Ground-truth template/result JSON")
    ap.add_argument("--start", type=int, default=0, help="Image index to open first")
    args = ap.parse_args()

    images_dir = Path(args.images).resolve()
    gt_path = Path(args.gt).resolve()

    if not images_dir.exists():
        print(f"ERROR: images directory does not exist: {images_dir}")
        raise SystemExit(2)
    if not gt_path.exists():
        print(f"ERROR: GT JSON does not exist: {gt_path}")
        raise SystemExit(2)

    root = tk.Tk()
    try:
        Reviewer(root, gt_path, images_dir, args.start)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("Reviewer error", str(exc))
        raise


if __name__ == "__main__":
    main()
