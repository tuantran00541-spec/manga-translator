import cv2
import numpy as np
import pytest
from PIL import Image

from app.render import display_lettering
from app.render.font_metrics import get_font_path
from app.render.lettering_plane import reconstruct_lettering_plane


_SIZE = (560, 440)


def _projected_quad(pitch=0):
    yaw, pitch = np.radians((32, pitch))
    across = np.array([np.cos(yaw), 0, np.sin(yaw)])
    down = np.array([-np.sin(yaw) * np.sin(pitch), np.cos(pitch), np.cos(yaw) * np.sin(pitch)])
    points = np.array([[0, 0, 6.] + u * across + v * down
                       for u, v in ((-2, -.55), (2, -.55), (2, .55), (-2, .55))])
    f = max(_SIZE)
    centre = (np.array(_SIZE) - 1) / 2
    quad = f * points[:, :2] / points[:, 2, None] + centre
    return quad.astype(np.float32), float(points[:, 2].max() / points[:, 2].min())


@pytest.mark.parametrize("pitch", [0, 25])
def test_lettering_plane_recovers_surface_aspect_and_reprojects_the_four_anchors(pitch):
    quad, depth = _projected_quad(pitch)
    pixel_size = np.rint((np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))).astype(int)
    plane = reconstruct_lettering_plane(quad, _SIZE, tuple(pixel_size))
    assert plane is not None
    assert plane.depth_ratio == pytest.approx(depth, rel=1e-5)
    w, h = plane.size
    assert (w - 1) / (h - 1) == pytest.approx(4 / 1.1, abs=.01)
    corners = np.float32([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])
    projected = cv2.perspectiveTransform(corners, plane.projection())[0]
    np.testing.assert_allclose(projected, quad, atol=.02)


def test_lettering_plane_leaves_flat_and_nonrectangular_ink_shapes_in_the_image_renderer():
    flat = np.float32([[50, 120], [440, 120], [440, 260], [50, 260]])
    assert reconstruct_lettering_plane(flat, _SIZE, (390, 140)) is None
    skewed = np.float32([[50, 200], [440, 40], [440, 140], [50, 342]])
    assert reconstruct_lettering_plane(skewed, _SIZE, (390, 120)) is None


def test_surface_projection_preserves_image_space_alternative_when_it_hits_protected_ink(monkeypatch):
    quad, _ = _projected_quad()
    raster = display_lettering._text_alpha(get_font_path("emphasis.anton"), "THUMP", 120)
    matrix = cv2.getPerspectiveTransform(display_lettering._corners(*raster.size), quad)
    glyphs = cv2.warpPerspective(np.asarray(raster), matrix, _SIZE)
    clean = Image.new("RGB", _SIZE, "white")
    raw = clean.copy()
    outline = cv2.dilate(glyphs, np.ones((9, 9), np.uint8))
    raw.paste((230, 60, 30), (0, 0, *_SIZE), Image.fromarray(outline))
    raw.paste((0, 0, 0), (0, 0, *_SIZE), Image.fromarray(glyphs))
    blocked = np.zeros(glyphs.shape, dtype=bool)
    options = dict(source="THUMP", font="emphasis.anton", blocked=blocked,
                   color_profile={"fill": "#000000", "stroke_color": "#e63c1e"})
    result, stats = display_lettering.render_source_sfx(raw, clean, glyphs, "ẦM!", **options)
    assert stats["sfx_depth_ratio"] > 1.2
    with monkeypatch.context() as patch:
        patch.setattr(display_lettering, "reconstruct_lettering_plane", lambda *_: None)
        previous, _ = display_lettering.render_source_sfx(raw, clean, glyphs, "ẦM!", **options)
    introduced = np.any(np.asarray(result) < 245, axis=2) & np.all(np.asarray(previous) == 255, axis=2)
    assert introduced.any()
    blocked[tuple(np.argwhere(introduced)[0])] = True
    fallback, fallback_stats = display_lettering.render_source_sfx(raw, clean, glyphs, "ẦM!", **options)
    assert "sfx_depth_ratio" not in fallback_stats
    assert fallback.tobytes() == previous.tobytes()
