from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.detector import ctd_mask, kiuyha_detector as kd
from ctd_fake import InkModel


class _MockFixedSession:
    def __init__(self, shape, boxes_fn):
        self.shape = shape
        self.boxes_fn = boxes_fn
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=self.shape)]

    def run(self, _names, feeds):
        self.calls += 1
        blob = feeds["images"]
        rows = np.zeros((1, 300, 6), np.float32)
        for idx, (x1, y1, x2, y2, score) in enumerate(self.boxes_fn(self.calls, blob)):
            rows[0, idx] = [x1, y1, x2, y2, score, 0]
        return [rows]


def test_line_art_preserves_white_filled_dark_outlined_text_connected_to_background():
    # White letters with dark outline touching a dark panel border across the crop
    img = np.full((240, 420, 3), 235, np.uint8)
    # Dark panel band spanning >=90% width and >=60% height (simulated action/border lines)
    cv2.rectangle(img, (10, 20), (410, 210), (30, 30, 30), 6)
    cv2.line(img, (10, 120), (410, 120), (30, 30, 30), 6)
    # White-filled, thick-black-outlined text crossing the dark line
    cv2.putText(img, "HERO", (60, 155), cv2.FONT_HERSHEY_DUPLEX, 3.0, (25, 25, 25), 18)
    cv2.putText(img, "HERO", (60, 155), cv2.FONT_HERSHEY_DUPLEX, 3.0, (250, 250, 250), 8)
    seed = np.zeros((240, 420), bool)
    seed[60:175, 50:370] = True
    prob = np.where(seed, 0.90, 0.0).astype(np.float32)

    kept = ctd_mask._line_art(img, seed, (0, 0), prob=prob)
    assert kept.sum() == seed.sum()


def test_white_bubble_display_mask_falls_back_when_surrounded_by_dark_panel():
    # Large white bubble sitting inside a solid black panel (border is dark, not white page margin)
    img = np.full((500, 900, 3), 10, np.uint8)
    cv2.ellipse(img, (450, 250), (320, 170), 0, 0, 360, (250, 250, 250), -1)
    cv2.putText(img, "UNION", (260, 280), cv2.FONT_HERSHEY_DUPLEX, 3.5, (15, 15, 15), 12)
    assert kd._white_bubble_display_mask(img, (100, 60, 800, 440)) is None


def test_coarse_boxes_chunks_super_tall_webtoon_slices_instead_of_crushing():
    tall = np.full((7200, 900, 3), 255, np.uint8)
    sess = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [])
    det = kd.KiuyhaTextDetector("unused", session=sess)
    det._coarse_boxes(tall)
    assert sess.calls >= 2, "super-tall 7200px slice must be chunked into multiple coarse passes"

    normal = np.full((2400, 800, 3), 255, np.uint8)
    sess2 = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [])
    det2 = kd.KiuyhaTextDetector("unused", session=sess2)
    det2._coarse_boxes(normal)
    assert sess2.calls == 1, "standard 2400px slice stays a single coarse pass"


def test_native_aux_pass_catches_small_dots_and_rejects_single_solid_dot(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=100))

    # Case 1: 6-dot ellipsis "......" on a 1280x1280 canvas (1:1 scale -> no letterbox offset)
    dots_img = np.full((1280, 1280, 3), 255, np.uint8)
    for x in range(110, 210, 16):
        cv2.circle(dots_img, (x, 60), 5, (10, 10, 10), -1)

    primary = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [])
    aux = _MockFixedSession(
        [1, 3, 1280, 1280],
        # Native tile call is call 2 (after overview call 1)
        lambda c, _b: [(100, 45, 220, 75, 0.26)] if c == 2 else [],
    )
    det = kd.KiuyhaTextDetector("unused", session=primary, aux_session=aux)
    boxes = det.text_boxes(dots_img)
    assert len(boxes) == 1 and not boxes[0].needs_review

    # Case 2: Single tiny isolated camera dot at 0.26 -> MUST be rejected
    dot_img = np.full((1280, 1280, 3), 255, np.uint8)
    cv2.circle(dot_img, (160, 60), 4, (10, 10, 10), -1)
    primary2 = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [])
    aux2 = _MockFixedSession(
        [1, 3, 1280, 1280],
        lambda c, _b: [(140, 40, 180, 80, 0.26)] if c == 2 else [],
    )
    det2 = kd.KiuyhaTextDetector("unused", session=primary2, aux_session=aux2)
    assert det2.text_boxes(dot_img) == []


def test_packed_rotated_aux_catches_vertical_sign_even_when_upright_text_exists(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=100))
    img = np.full((1200, 800, 3), 255, np.uint8)
    # Upright bubble at top-left
    cv2.putText(img, "HELLO", (80, 200), cv2.FONT_HERSHEY_DUPLEX, 1.5, (10, 10, 10), 4)
    # Vertical sign at bottom-right
    sign = np.full((120, 360, 3), 255, np.uint8)
    cv2.putText(sign, "SIGN", (40, 90), cv2.FONT_HERSHEY_DUPLEX, 2.2, (10, 10, 10), 6)
    img[700:1060, 620:740] = cv2.rotate(sign, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Primary returns only the upright box
    primary = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [(60, 130, 320, 230, 0.90)])
    # Aux returns the rotated sign on the packed rotated canvas
    rot_scale = min(1.0, 1280 / 1200, ((1280 - kd.COLUMN_GAP) // 2) / 800)
    ox = (1280 - int(round(1200 * rot_scale))) // 2
    # In CW rotation: (x, y) -> (h-1-y, x), so (620..740, 700..1060) maps to tx in (139..499), ty in (620..740)
    rx1 = ox + int(139 * rot_scale)
    rx2 = ox + int(499 * rot_scale)
    ry1 = int(620 * rot_scale)
    ry2 = int(740 * rot_scale)

    aux = _MockFixedSession(
        [1, 3, 1280, 1280],
        # Packed rotated call is after overview (1) + native tile (2) -> call 3
        lambda c, _b: [(rx1, ry1, rx2, ry2, 0.72)] if c == 3 else [],
    )
    det = kd.KiuyhaTextDetector("unused", session=primary, aux_session=aux)
    boxes = det.text_boxes(img)
    assert len(boxes) == 2
    assert any(b.x1 >= 550 and b.y1 >= 650 for b in boxes)


def test_rescue_handles_bright_text_on_dark_background(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(light_above=200))
    img = np.full((600, 800, 3), (20, 20, 25), np.uint8)
    cv2.putText(img, "CHAT LINE", (120, 320), cv2.FONT_HERSHEY_DUPLEX, 1.0, (245, 245, 245), 2)
    sess = _MockFixedSession([1, 3, "height", "width"], lambda _c, _b: [(90, 270, 420, 350, 0.20)])
    det = kd.KiuyhaTextDetector("unused", session=sess)
    boxes = det.text_boxes(img)
    assert len(boxes) == 1
    assert boxes[0].x1 <= 130 and boxes[0].x2 >= 270


def test_tighten_box_trims_empty_padding_without_losing_mask_pixels():
    mask = np.zeros((200, 300), np.uint8)
    mask[60:110, 80:210] = 255
    letters = np.zeros((200, 300), bool)
    letters[65:105, 85:205] = True
    box = kd._text_box(100, 200, 400, 400, 0.9, mask, "kiuyha", letters, core=(150, 250, 350, 350))
    tight = kd._tighten_box(box, pad=4)
    assert (tight.x1, tight.y1, tight.x2, tight.y2) == (176, 256, 314, 314)
    assert tight.mask.shape == (58, 138)
    assert int(np.count_nonzero(tight.mask)) == int(np.count_nonzero(mask))
    assert int(np.count_nonzero(tight.letters)) == int(np.count_nonzero(letters))


def test_welded_words_rejects_square_ui_icons_while_keeping_wide_words():
    img = np.full((200, 400, 3), 240, np.uint8)
    # Square/circular UI button icon (30x28) -> must be rejected
    cv2.circle(img, (80, 100), 14, (40, 40, 40), -1)
    # Wide welded word (66x20, aspect=3.3) -> must be kept
    cv2.rectangle(img, (200, 90), (266, 110), (40, 40, 40), -1)
    words = kd._welded_words(img, (20, 40, 340, 160))
    assert len(words) == 1
    (wx1, wy1, wx2, wy2), _ = words[0]
    assert wx1 >= 195 and wx2 <= 270

