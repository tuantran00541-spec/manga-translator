import threading
import numpy as np
import cv2
from PIL import Image


class MultiLangOCR:
    def __init__(self):
        self._manga_ocr = None
        self._paddle_engines = {}
        self._manga_lock = threading.Lock()
        self._paddle_lock = threading.Lock()

    def read(self, image: np.ndarray, lang: str) -> str:
        if image is None or image.size == 0:
            return ""

        padded_raw = self._add_white_padding(image, pad=20)

        text = self._read_engine(padded_raw, lang)
        if text and len(text.strip()) > 0:
            return text

        prep = self._preprocess_for_ocr(padded_raw)
        return self._read_engine(prep, lang)

    def _read_engine(self, image: np.ndarray, lang: str) -> str:
        if lang == "ja":
            return self._read_manga_ocr(image)
        return self._read_paddle(image, lang)

    @staticmethod
    def _add_white_padding(image: np.ndarray, pad: int = 20) -> np.ndarray:
        h, w = image.shape[:2]
        if image.ndim == 2:
            padded = np.full((h + pad * 2, w + pad * 2), 255, dtype=np.uint8)
            padded[pad:pad + h, pad:pad + w] = image
        else:
            padded = np.full((h + pad * 2, w + pad * 2, 3), 255, dtype=np.uint8)
            padded[pad:pad + h, pad:pad + w, :] = image
        return padded

    @staticmethod
    def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Invert if text is light on a dark background
        if float(gray.mean()) < 135:
            gray = cv2.bitwise_not(gray)

        # Apply mild CLAHE to boost contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

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
                if text_content and confidence > 0.3:
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
                        lang=target_lang, use_angle_cls=True, show_log=False
                    )
        return self._paddle_engines[target_lang]
