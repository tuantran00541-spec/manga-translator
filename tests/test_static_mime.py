import mimetypes

from starlette.responses import FileResponse

from app import main
from app.config import BASE_DIR


def test_module_scripts_are_served_as_javascript():
    script = BASE_DIR / "app" / "static" / "js" / "review-stitch" / "index.js"

    assert FileResponse(script).media_type == "text/javascript"


def test_javascript_mime_type_overrides_a_text_plain_registry_entry(monkeypatch):
    monkeypatch.setitem(mimetypes._db.types_map[True], ".js", "text/plain")
    assert mimetypes.guess_type("index.js")[0] == "text/plain"

    main.register_javascript_mime_type()

    assert mimetypes.guess_type("index.js")[0] == "text/javascript"
