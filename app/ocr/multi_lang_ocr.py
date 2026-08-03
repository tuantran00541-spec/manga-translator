import threading
import numpy as np
import cv2
from PIL import Image

MIN_OCR_DIM = 48
TARGET_OCR_HEIGHT = 64


class MultiLangOCR:
    def __init__(self):
        self._manga_ocr = None
        self._paddle_engines = {}
        self._manga_lock = threading.Lock()
        self._paddle_lock = threading.Lock()

    def read(self, image: np.ndarray, lang: str) -> str:
        if image is None or image.size == 0:
            return ""

        image = self._ensure_min_size(image)
        padded = self._add_white_padding(image, pad=24)

        raw_text = self._read_engine(padded, lang)
        if raw_text and len(raw_text.strip()) >= 2:
            return raw_text

        enhanced = self._enhance_for_ocr(padded)
        enhanced_text = self._read_engine(enhanced, lang)

        if not raw_text and not enhanced_text:
            return ""
        if not enhanced_text:
            return raw_text or ""
        if not raw_text:
            return enhanced_text
        return enhanced_text if len(enhanced_text) >= len(raw_text) else raw_text

    def _read_engine(self, image: np.ndarray, lang: str) -> str:
        if lang == "ja":
            return self._read_manga_ocr(image)
        return self._read_paddle(image, lang)

    @staticmethod
    def _ensure_min_size(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h >= MIN_OCR_DIM and w >= MIN_OCR_DIM:
            return image
        scale = max(MIN_OCR_DIM / max(h, 1), MIN_OCR_DIM / max(w, 1), 1.0)
        if h < TARGET_OCR_HEIGHT:
            scale = max(scale, TARGET_OCR_HEIGHT / max(h, 1))
        scale = min(scale, 4.0)
        new_w = max(MIN_OCR_DIM, int(w * scale))
        new_h = max(MIN_OCR_DIM, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _add_white_padding(image: np.ndarray, pad: int = 24) -> np.ndarray:
        h, w = image.shape[:2]
        if image.ndim == 2:
            padded = np.full((h + pad * 2, w + pad * 2), 255, dtype=np.uint8)
            padded[pad:pad + h, pad:pad + w] = image
        else:
            padded = np.full((h + pad * 2, w + pad * 2, 3), 255, dtype=np.uint8)
            padded[pad:pad + h, pad:pad + w, :] = image
        return padded

    @staticmethod
    def _enhance_for_ocr(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Measure inner region mean to detect dark background text correctly despite white padding
        h, w = gray.shape[:2]
        inner = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] if (h >= 8 and w >= 8) else gray
        if float(inner.mean()) < 130:
            gray = cv2.bitwise_not(gray)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    def _read_manga_ocr(self, image: np.ndarray) -> str:
        if self._manga_ocr is None:
            with self._manga_lock:
                if self._manga_ocr is None:
                    from manga_ocr import MangaOcr
                    self._manga_ocr = MangaOcr()
        pil_img = Image.fromarray(image)
        return self._manga_ocr(pil_img).strip()

    def _read_paddle(self, image: np.ndarray, lang: str) -> str:
        engine = self._get_paddle_engine(lang)
        result = engine.ocr(image, cls=True)
        if not result or not result[0]:
            return ""

        lines = []
        for line in result[0]:
            if line and len(line) >= 2 and line[1] and line[1][0]:
                text_content = line[1][0].strip()
                confidence = line[1][1] if len(line[1]) > 1 else 1.0
                if text_content and confidence > 0.4:
                    lines.append(text_content)
        return "\n".join(lines).strip()

    def _get_paddle_engine(self, lang: str):
        lang_map = {
            "korean": "korean",
            "ko": "korean",
            "ch": "ch",
            "zh": "ch",
            "en": "en",
            "ja": "japan",
            "japan": "japan",
        }
        target_lang = lang_map.get(lang.lower(), lang)
        if target_lang not in self._paddle_engines:
            with self._paddle_lock:
                if target_lang not in self._paddle_engines:
                    from paddleocr import PaddleOCR
                    self._paddle_engines[target_lang] = PaddleOCR(
                        lang=target_lang,
                        use_angle_cls=True,
                        show_log=False,
                        det_db_thresh=0.25,
                        det_db_box_thresh=0.45,
                    )
        return self._paddle_engines[target_lang]
