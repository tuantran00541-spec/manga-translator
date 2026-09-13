import unittest

import numpy as np
from PIL import Image, ImageDraw

from app.render.text_renderer import (
    _text_ink_bbox,
    _text_ink_width,
    get_font_object,
    get_font_path,
    render_text_in_box,
)


class TextRendererAlignmentRegressionTests(unittest.TestCase):
    def test_ink_width_uses_true_bbox_extent(self):
        image = Image.new("RGB", (500, 200), "white")
        draw = ImageDraw.Draw(image)
        font = get_font_object(str(get_font_path("Mac-dinh-3")), 64)
        bbox = draw.textbbox((0, 0), "Đó là...", font=font, stroke_width=1)
        self.assertEqual(
            _text_ink_width(draw, "Đó là...", font, stroke_w=1),
            bbox[2] - bbox[0],
        )

    def test_center_alignment_centers_visible_ink_with_nonzero_font_bearing(self):
        image = Image.new("RGB", (420, 220), "white")
        draw = ImageDraw.Draw(image)
        samples = [
            "Đó là...",
            "Sao có thể?!",
            "Lần này...",
            "Tuyệt kỹ!",
            "Bởi vì...",
            "Jý Á Vũ",
        ]
        font_names = [
            "Mac-dinh-3",
            "Mac-dinh-2",
            "Granite",
            "Manga-fonts",
            "Skill-fonts-1",
            "Curves-Regular",
        ]

        chosen = None
        for font_name in font_names:
            path = get_font_path(font_name)
            font = get_font_object(str(path), 56)
            for text in samples:
                bbox = _text_ink_bbox(draw, text, font, stroke_w=0)
                width = bbox[2] - bbox[0]
                if abs(bbox[0]) >= 2 and 0 < width < 300:
                    chosen = (font_name, text)
                    break
            if chosen:
                break

        self.assertIsNotNone(chosen, "expected at least one bundled manga font with non-zero left bearing")
        font_name, text = chosen
        box = (50, 40, 370, 180)
        rendered = render_text_in_box(
            Image.new("RGB", (420, 220), "white"),
            text,
            box,
            font_name=font_name,
            font_size=56,
            fill=(0, 0, 0),
            stroke_width=0,
            horizontal_align="center",
            vertical_align="middle",
        )

        arr = np.asarray(rendered.convert("L"))
        ys, xs = np.where(arr < 128)
        self.assertGreater(len(xs), 0)
        ink_center_x = (float(xs.min()) + float(xs.max())) / 2.0
        expected_center_x = (box[0] + box[2]) / 2.0
        self.assertLessEqual(
            abs(ink_center_x - expected_center_x),
            2.0,
            f"visible ink drifted from center for {font_name} / {text!r}",
        )


if __name__ == "__main__":
    unittest.main()
