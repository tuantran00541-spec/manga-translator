from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.ai_mode import clean_review, decider


def test_clean_review_skips_spotless_flat_bubbles_but_keeps_residue_and_art():
    orig = np.full((200, 300, 3), 255, dtype=np.uint8)
    clean = np.full((200, 300, 3), 255, dtype=np.uint8)

    # Box 0: spotless flat SmartFill white bubble (dark text in original, spotless white in clean)
    b0 = (20, 20, 120, 80)
    cv2.putText(orig, "HELLO", (28, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)

    # Box 1: white bubble with leftover dark stroke inside clean
    b1 = (150, 20, 260, 80)
    cv2.line(clean, (170, 45), (230, 55), (10, 10, 10), 3)

    # Box 2: textured artwork background inside clean
    b2 = (20, 110, 140, 180)
    rng = np.random.default_rng(7)
    clean[110:180, 20:140] = rng.integers(40, 210, size=(70, 120, 3), dtype=np.uint8)

    crops = clean_review.crops_for(0, orig, clean, [b0, b1, b2])
    rects = [c.rect for c in crops]
    assert b0 not in rects
    assert b1 in rects
    assert b2 in rects


def _decider_box(x1=20, y1=20, x2=120, y2=80, *, conf=0.92, bubble=True):
    return SimpleNamespace(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=conf,
        bubble_mask={"kind": "bubble"} if bubble else None,
        safe_to_inpaint=True,
        ocr_eligible=True,
        needs_review=False,
        deferred_reason=None,
    )


def test_art_boxes_skips_confident_speech_bubbles(monkeypatch):
    calls = []
    monkeypatch.setattr(decider, "_classify", lambda *a, **kw: (calls.append(1), "art")[1])
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    b = _decider_box(conf=0.91, bubble=True)
    assert decider.art_boxes(object(), "m", "k", img, [b]) == set()
    assert calls == []
    assert b.safe_to_inpaint is True


def test_art_boxes_still_defers_non_bubble_on_none_reply(monkeypatch):
    monkeypatch.setattr(decider, "_classify", lambda *a, **kw: None)
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    b = _decider_box(conf=0.65, bubble=False)
    assert decider.art_boxes(object(), "m", "k", img, [b]) == set()
    assert b.safe_to_inpaint is False
    assert b.deferred_reason == "lettering_unconfirmed"


def test_translate_pages_batch_combines_two_slices_into_separate_full_res_images(tmp_path, monkeypatch):
    from app.ai_providers import PROVIDERS
    from app.translation.vision import VisionPageTranslator, VisionTranslationResult

    seen_calls = []

    def fake_openai(self, system, prompt, marked, *, api_key, ids, max_tokens):
        seen_calls.append({"system": system, "prompt": prompt, "marked": marked, "ids": ids})
        return (
            VisionTranslationResult(
                translations={"1": "Xin chao", "2": "Tam biet"},
                model=self.model,
                usage={"total_tokens": 100},
                estimated_cost_usd=0.002,
            ),
            {"translations": [{"id": "1", "translated_text": "Xin chao"}, {"id": "2", "translated_text": "Tam biet"}]},
        )

    monkeypatch.setattr(VisionPageTranslator, "_openai", fake_openai)
    tr = VisionPageTranslator(PROVIDERS["openai"], "gpt-4o-mini")

    p1 = tmp_path / "p1.png"
    p2 = tmp_path / "p2.png"
    cv2.imwrite(str(p1), np.full((240, 200, 3), 255, dtype=np.uint8))
    cv2.imwrite(str(p2), np.full((240, 200, 3), 255, dtype=np.uint8))

    pages_spec = [
        {
            "page_index": 0,
            "original_path": p1,
            "cleaned_path": p1,
            "items": [{"id": "text_1", "text": "Hello", "region": (10, 10, 100, 60)}],
        },
        {
            "page_index": 1,
            "original_path": p2,
            "cleaned_path": p2,
            "items": [{"id": "text_2", "text": "Bye", "region": (10, 10, 100, 60)}],
        },
    ]
    out = tr.translate_pages_batch(pages_spec, api_key="test-key", source_lang="en", target_lang="vi")

    assert len(seen_calls) == 1
    assert isinstance(seen_calls[0]["marked"], list)
    assert len(seen_calls[0]["marked"]) == 2
    assert out[0].translations == {"text_1": "Xin chao"}
    assert out[1].translations == {"text_2": "Tam biet"}


def test_isolated_paper_ink_rejects_white_text_on_colored_system_panel():
    from app.detector.ink_mask import isolated_paper_ink

    # Simulate white glowing system text on a dark blue/red RPG panel (Lat 10 / Lat 31 bug)
    gray = np.full((240, 480), 82, dtype=np.uint8)
    cv2.putText(gray, "SKILLBOOK", (65, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 250, 5, cv2.LINE_AA)
    cv2.putText(gray, "KAGURAZAKA", (55, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 250, 5, cv2.LINE_AA)
    grown = np.zeros_like(gray, dtype=bool)
    grown[55:195, 45:435] = True
    letters = (gray >= 180) | grown

    assert isolated_paper_ink(gray, grown, letters) is None


def test_protect_free_text_art_skips_white_text_on_black_speech_bubble():
    from app.detector.ink_mask import protect_free_text_art

    # Simulate white text inside a black speech bubble on a white page (Lat 45 / Lat 57 bug)
    gray = np.full((260, 360), 250, dtype=np.uint8)
    cv2.ellipse(gray, (180, 130), (145, 95), 0, 0, 360, 8, -1)
    cv2.putText(gray, "ON THE", (95, 118), cv2.FONT_HERSHEY_DUPLEX, 1.3, 248, 4, cv2.LINE_AA)
    cv2.putText(gray, "GROUND.", (85, 168), cv2.FONT_HERSHEY_DUPLEX, 1.3, 248, 4, cv2.LINE_AA)
    letters = (gray > 200) & (np.hypot((np.indices(gray.shape)[1] - 180) / 145, (np.indices(gray.shape)[0] - 130) / 95) < 0.85)
    grown = cv2.dilate(letters.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0

    assert protect_free_text_art(gray, grown, letters) is None


def test_renderer_recovers_invisible_raw_measured_white_on_white():
    from PIL import Image
    from app.render.text_renderer import render_text_in_box

    # Lat 69 bug: source="raw" mis-sampled #fbffff on a #ffffff bubble with no stroke
    img = Image.new("RGB", (240, 100), (255, 255, 255))
    profile = {"source": "raw", "sample_version": 1, "fill": "#fbffff"}
    render_text_in_box(
        img,
        "Mot thanh kiem?",
        (15, 15, 225, 85),
        fill="auto",
        font_size=28,
        stroke_width="auto",
        stroke_color="auto",
        color_profile=profile,
        container="bubble",
    )
    arr = np.asarray(img)
    dark_pixels = np.count_nonzero((arr[:, :, 0] < 80) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80))
    assert dark_pixels > 150, f"expected legible dark text, got only {dark_pixels} dark pixels"


def test_composite_api_region_rejects_white_wash_over_blue_panel():
    from app.inpaint.api_inpaint import composite_api_region

    # Blue RPG system panel (Lat 31 bug: AI image model painted solid white over blue panel)
    base = np.full((160, 260, 3), (230, 155, 25), dtype=np.uint8)
    mask = np.zeros((160, 260), dtype=np.uint8)
    mask[45:115, 50:210] = 255
    ai_white = np.full_like(base, 253)

    out = composite_api_region(base.copy(), ai_white, mask, (0, 0, 260, 160), feather=False)
    # Must preserve the blue panel instead of pasting a white blob!
    assert np.array_equal(out, base)
