from pathlib import Path


fast_path = Path("app/inpaint/fast_lama_inpainter.py")
text = fast_path.read_text(encoding="utf-8")
old = '            box.semantic_type == "speech_bubble"\n'
new = '            box.semantic_type in {"speech_bubble", "free_text"}\n'
if text.count(old) != 1:
    raise RuntimeError(f"candidate marker count={text.count(old)}")
fast_path.write_text(text.replace(old, new, 1), encoding="utf-8")


tests = Path("tests/test_fast_inpaint_paths.py")
text = tests.read_text(encoding="utf-8")
if "test_smooth_gradient_free_text_uses_reconstruction_without_lama" in text:
    raise RuntimeError("free-text test already exists")
text += r'''


def test_smooth_gradient_free_text_uses_reconstruction_without_lama():
    h, w = 180, 240
    yy, xx = np.mgrid[:h, :w]
    background = np.empty((h, w, 3), dtype=np.uint8)
    background[..., 0] = np.clip(225 + xx * 0.025 + yy * 0.018, 0, 255)
    background[..., 1] = np.clip(229 + xx * 0.022 + yy * 0.016, 0, 255)
    background[..., 2] = np.clip(234 + xx * 0.018 + yy * 0.013, 0, 255)
    image = background.copy()

    mask = np.zeros((70, 130), dtype=np.uint8)
    mask[16:20, 18:112] = 255
    mask[36:40, 30:100] = 255
    box = _speech_box(55, 50, 185, 120, mask)
    box.semantic_type = "free_text"
    image[50:120, 55:185][mask > 127] = 15

    expected_authority = build_mask(image.shape[:2], [box], image)
    inpainter = FastInpainter()
    inpainter._smart_fill_color = lambda crop, local_mask: None
    output = inpainter.inpaint(image, [box])
    metrics = inpainter.last_metrics()

    changed = np.any(output != image, axis=2)
    assert not np.any(changed & (expected_authority <= 127))
    assert metrics["bubble_fast_fill_gradient_regions"] == 1
    assert metrics["bubble_fast_fill_telea_regions"] == 0
    assert metrics["lama_model_runs"] == 0

    authority = expected_authority > 127
    mae = float(
        np.abs(output.astype(np.int16) - background.astype(np.int16))[authority].mean()
    )
    assert mae < 5.0
'''
tests.write_text(text, encoding="utf-8")
