from pathlib import Path

from PIL import Image

from app.routers.export import _render_request_from_page, _stitch_pngs


def test_render_request_uses_persisted_text_object_state():
    page = {
        "text_objects": [
            {
                "id": "obj1",
                "translation": "Xin chào",
                "style": {
                    "color": "#112233",
                    "font": "default",
                    "fontSize": "24",
                    "bold": True,
                    "strokeWidth": "2",
                    "strokeColor": "#ffffff",
                    "bgColor": "transparent",
                    "cornerRadius": "4",
                    "horizontalAlign": "left",
                    "verticalAlign": "top",
                },
            }
        ]
    }
    request = _render_request_from_page("deadbeef", 0, page)
    assert request.translations == {"obj1": "Xin chào"}
    assert request.font_sizes == {"obj1": "24"}
    assert request.bolds == {"obj1": True}
    assert request.horizontal_aligns == {"obj1": "left"}


def test_stitch_pngs_reconstructs_vertical_source_page(tmp_path: Path):
    top = tmp_path / "top.png"
    bottom = tmp_path / "bottom.png"
    Image.new("RGB", (20, 10), "white").save(top)
    Image.new("RGB", (20, 15), "black").save(bottom)

    payload = _stitch_pngs([top, bottom])
    output = tmp_path / "joined.png"
    output.write_bytes(payload)
    with Image.open(output) as image:
        assert image.size == (20, 25)
        assert image.getpixel((5, 5)) == (255, 255, 255)
        assert image.getpixel((5, 20)) == (0, 0, 0)
