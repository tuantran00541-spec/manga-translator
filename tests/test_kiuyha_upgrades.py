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


def test_line_art_drops_single_axis_panel_borders():
    img = np.full((120, 320, 3), 240, np.uint8)
    # Single horizontal panel border spanning across crop width
    cv2.line(img, (0, 60), (319, 60), (15, 15, 15), 5)
    seed = np.zeros((120, 320), bool)
    seed[57:64, 10:310] = True
    kept = ctd_mask._line_art(img, seed, (0, 0))
    assert not kept.any()


def test_plausibility_rejects_periodic_screentone_grid_on_low_score():
    letters = np.zeros((160, 200), bool)
    for y in range(20, 140, 10):
        for x in range(20, 180, 10):
            letters[y:y + 3, x:x + 3] = True
    mask = np.where(letters, np.uint8(255), np.uint8(0))
    assert not kd._passes_glyph_plausibility((800, 600), (100, 100, 300, 260), mask, letters, 0.28)


def test_dense_tightened_mask_uses_box_letters_instead_of_silent_drop():
    from app.detector.boxes import BubbleBox
    from app.detector.mask_builder import build_mask

    crop = np.full((60, 120, 3), 220, np.uint8)
    cv2.putText(crop, "HI", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 20, 20), 2)
    mask = np.full((24, 36), 255, np.uint8)
    letters = np.zeros((24, 36), bool)
    letters[4:20, 4:32] = True
    box = BubbleBox(
        20, 15, 56, 39, 0.9, mask,
        letters=letters,
        source_role="text_segmenter",
        mask_source="text_segmenter",
        safe_to_inpaint=True,
    )
    built = build_mask((60, 120), [box], crop)
    assert np.all(built[19:35, 24:60 if 60 <= 56 else 52] > 127)


def test_dark_uniform_bubble_derives_bubble_mask_and_avoids_double_inset():
    from app.detector.bubble_mask import clip_bubble_mask, derive_bubble_mask

    img = np.full((300, 400, 3), 240, np.uint8)
    cv2.ellipse(img, (200, 150), (130, 90), 0, 0, 360, (25, 25, 25), -1)
    cv2.putText(img, "DARK", (130, 165), cv2.FONT_HERSHEY_DUPLEX, 1.5, (240, 240, 240), 3)
    mask = derive_bubble_mask(img, (80, 70, 320, 230), seed=(200, 150))
    assert mask["kind"] == "bubble"
    clipped = clip_bubble_mask(mask, (80, 70, 210, 230))
    assert clipped is not None
    w_orig = sum(b - a for r in mask["rows"] for a, b in r)
    w_clip = sum(b - a for r in clipped["rows"] for a, b in r)
    # Clipping half shouldn't double-inset away interior widths on each band
    assert w_clip >= 0.42 * w_orig


def test_side_by_side_bubbles_sharing_y_do_not_merge():
    from app.detector.boxes import BubbleBox

    img = np.full((240, 460, 3), 245, np.uint8)
    cv2.ellipse(img, (110, 120), (80, 60), 0, 0, 360, (20, 20, 20), 3)
    cv2.ellipse(img, (340, 120), (80, 60), 0, 0, 360, (20, 20, 20), 3)
    m1 = np.zeros((80, 160), np.uint8)
    l1 = np.zeros((80, 160), bool)
    l1[20:50, 20:100] = True
    m1[15:55, 15:110] = 255
    b1 = BubbleBox(60, 80, 220, 160, 0.9, m1, letters=l1, mask_source="text_segmenter", safe_to_inpaint=True)
    b2 = BubbleBox(190, 80, 350, 160, 0.9, m1.copy(), letters=l1.copy(), mask_source="text_segmenter", safe_to_inpaint=True)
    joined = kd._join_grown([b1, b2], image=img)
    assert len(joined) == 2


def test_wide_two_page_spread_chunks_horizontally():
    spread = np.full((1400, 2400, 3), 255, np.uint8)
    sess = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [])
    det = kd.KiuyhaTextDetector("unused", session=sess)
    det._coarse_boxes(spread)
    assert sess.calls == 2


def test_probability_upscales_15x_on_tiny_low_seed_crop(monkeypatch):
    seen_scales = []
    def fake_prob(im, scale):
        seen_scales.append(scale)
        return np.full(im.shape[:2], 0.9 if scale == 1.5 else 0.0, dtype=np.float32)

    monkeypatch.setattr(ctd_mask, "_prob", fake_prob)
    tiny = np.full((80, 160, 3), 240, np.uint8)
    prob = ctd_mask.probability(tiny)
    assert seen_scales == [1.0, 0.5, 1.5]
    assert float(prob.max()) >= 0.8


def test_overview_box_supersedes_partial_clipped_existing_box():
    img = np.full((2400, 900, 3), 10, np.uint8)
    # Compute column 0 mapping in _overview_boxes so stubbed canvas coords map back accurately
    h, w = img.shape[:2]
    size_h, size_w = 1280, 1280
    max_h = max(32, size_h - kd.OVERVIEW_PAD_Y)
    scaled = [
        (max(1, int(round(w * min(1.0, tw / w, max_h / h)))),
         max(1, int(round(h * min(1.0, tw / w, max_h / h)))),
         min(1.0, tw / w, max_h / h))
        for tw in kd.OVERVIEW_WIDTHS
    ]
    total_w = sum(d[0] for d in scaled) + kd.OVERVIEW_GAP * (len(scaled) - 1)
    ox = max(kd.OVERVIEW_GAP, (size_w - total_w) // 2)
    sw, sh, sc = scaled[0]
    oy = (size_h - sh) // 2
    ov_box = (
        ox + int(round(72 * sc)),
        oy + int(round(440 * sc)),
        ox + int(round(836 * sc)),
        oy + int(round(1588 * sc)),
        0.89,
    )

    primary = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [])
    aux = _MockFixedSession([1, 3, 1280, 1280], lambda _c, _b: [ov_box])
    det = kd.KiuyhaTextDetector("unused", session=primary, aux_session=aux)

    # Case 1: Single clipped partial lower-confidence box -> large overview box MUST supersede
    kept = det._overview_boxes(img, [(47, 724, 718, 1569, 0.50)])
    assert len(kept) == 1
    assert kept[0][1] <= 445 and kept[0][2] >= 830 and kept[0][4] == pytest.approx(0.89)

    # Case 2: Two disjoint existing boxes (separate adjacent bubbles) -> MUST NOT merge/supersede
    kept_disjoint = det._overview_boxes(img, [(80, 460, 400, 1500, 0.50), (460, 460, 820, 1500, 0.50)])
    assert kept_disjoint == []

    # Case 3: Existing box already high confidence (0.85, diff < 0.10) -> suppressed
    kept_high = det._overview_boxes(img, [(47, 724, 718, 1569, 0.85)])
    assert kept_high == []


def test_ctd_cut_scales_edge_band_with_large_text_size():
    # Large display text (height 200px): glyph ends 10px from top & right edges -> cut must fire
    large_seed = np.zeros((300, 300), bool)
    large_seed[10:210, 40:290] = True
    assert ctd_mask._cut(large_seed) == (False, True, True, False)

    # Normal text (height 24px): glyph 10px from edges -> 4px edge band must NOT false-trigger
    small_seed = np.zeros((100, 100), bool)
    small_seed[10:34, 40:90] = True
    assert ctd_mask._cut(small_seed) == (False, False, False, False)


def test_reach_cut_letters_expands_by_large_text_size(monkeypatch):
    monkeypatch.setattr(ctd_mask, "_session", InkModel(dark_below=None, light_above=200))
    image = np.full((600, 500, 3), 10, np.uint8)
    # Two tall lines: top line at y=80..260, bottom line starts 8px inside first[1]=278
    image[80:260, 100:400] = 245
    image[286:466, 100:400] = 245
    first = (80, 278, 420, 500)
    box, seed, _ = ctd_mask._reach_cut_letters(image, first, (48, 48))
    assert box[1] <= 80, f"expected reach expansion up to y<=80, got {box}"
    assert seed[max(0, 80 - box[1]):max(1, 260 - box[1]), :].any()



