"""Two bounded live requests through the application's A.I scan/translation code."""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.ai_mode.page_scan import scan_slices
from app.ai_providers import resolve_provider
from app.translation.vision import VisionPageTranslator


def main():
    key = os.environ.get("MINIROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("The MINIROUTER_API_KEY Actions secret is not configured")
    model = "deepseek/deepseek-v4.1-flash-1017"
    provider = resolve_provider("minirouter", label="MiniRouter", protocol="openai",
                                api_base="https://api.minirouter.sh/v1")
    image = np.full((360, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (24, 35), (615, 280), (0, 0, 0), 3)
    cv2.putText(image, "HELLO FRIEND!", (95, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    scans, _ = scan_slices(provider, model, key, [(0, image)])
    assert len(scans) == 1 and scans[0].page_index == 0
    assert not scans[0].is_credit and not scans[0].no_text, "AI scan missed the text in the synthetic image"
    with tempfile.TemporaryDirectory() as tmp:
        original, clean = Path(tmp) / "original.png", Path(tmp) / "clean.png"
        assert cv2.imwrite(str(original), image)
        assert cv2.imwrite(str(clean), np.full_like(image, 255))
        result = VisionPageTranslator(provider, model).translate_page(
            original, clean, [{"id": "1", "text": "HELLO FRIEND!", "region": [50, 70, 590, 230]}],
            api_key=key, source_lang="en", target_lang="vi")
    assert result.translations.get("1", "").strip(), "AI returned no usable translation"
    print("PASS: AI scan detected text and the translation stage returned text")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Do not include provider error bodies or credentials in CI output.
        print(f"Live AI smoke failed ({type(exc).__name__}); inspect configuration and model availability.")
        raise SystemExit(1)
