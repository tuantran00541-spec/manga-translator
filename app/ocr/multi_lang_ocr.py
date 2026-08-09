import threading
import numpy as np
import cv2
from PIL import Image

MIN_OCR_DIM = 64
TARGET_OCR_HEIGHT = 128


class MultiLangOCR:
    def __init__(self):
        self._manga_ocr = None
        self._paddle_engines = {}
        self._manga_lock = threading.Lock()
        self._paddle_locks_guard = threading.Lock()
        self._paddle_locks = {}

    def read(self, image: np.ndarray, lang: str) -> str:
        if image is None or image.size == 0:
            return ""

        image = self._ensure_min_size(image)
        enhanced = self._enhance_for_ocr(image)

        padded_raw = self._add_white_padding(image, pad=32)
        padded_enhanced = self._add_white_padding(enhanced, pad=32)

        raw_text = self._read_engine(padded_raw, lang)
        if raw_text and len(raw_text.strip()) >= 2:
            return raw_text

        enhanced_text = self._read_engine(padded_enhanced, lang)

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
        if h >= TARGET_OCR_HEIGHT and w >= MIN_OCR_DIM:
            return image
        scale = max(MIN_OCR_DIM / max(w, 1), TARGET_OCR_HEIGHT / max(h, 1), 1.0)
        scale = min(scale, 4.0)
        new_w = max(MIN_OCR_DIM, int(w * scale))
        new_h = max(TARGET_OCR_HEIGHT, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _add_white_padding(image: np.ndarray, pad: int = 32) -> np.ndarray:
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

        h, w = gray.shape[:2]
        inner = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] if (h >= 8 and w >= 8) else gray
        if float(inner.mean()) < 130:
            gray = cv2.bitwise_not(gray)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        sharpen_kernel = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(enhanced_gray, -1, sharpen_kernel)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    def _read_manga_ocr(self, image: np.ndarray) -> str:
        with self._manga_lock:
            if self._manga_ocr is None:
                from manga_ocr import MangaOcr
                self._manga_ocr = MangaOcr()
            pil_img = Image.fromarray(image)
            return self._manga_ocr(pil_img).strip()

    @staticmethod
    def _split_lines(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        img_h, img_w = gray.shape[:2]
        edge_margin = 4

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 20), 1))
        dilated = cv2.dilate(binary, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 20 or h < 6:
                continue
            touches_top = y <= edge_margin
            touches_bottom = (y + h) >= (img_h - edge_margin)
            touches_left = x <= edge_margin
            touches_right = (x + w) >= (img_w - edge_margin)
            edges_touched = sum([touches_top, touches_bottom, touches_left, touches_right])
            if edges_touched >= 2:
                continue
            span_ratio = w / max(img_w, 1)
            if (touches_top or touches_bottom) and span_ratio > 0.85 and h < 15:
                continue
            boxes.append((x, y, x + w, y + h))

        boxes.sort(key=lambda b: b[1])
        return boxes

    def _read_paddle(self, image: np.ndarray, lang: str) -> str:
        engine, lang_lock = self._get_paddle_engine(lang)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        line_boxes = self._split_lines(gray)
        with lang_lock:
            if len(line_boxes) <= 1:
                return self._read_paddle_full(engine, image)

            lines = []
            for x1, y1, x2, y2 in line_boxes:
                pad = 6
                cy1 = max(0, y1 - pad)
                cy2 = min(image.shape[0], y2 + pad)
                cx1 = max(0, x1 - pad)
                cx2 = min(image.shape[1], x2 + pad)
                crop = image[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                text = self._read_paddle_line(engine, crop)
                if text:
                    lines.append(text)

            if not lines:
                return self._read_paddle_full(engine, image)

            return "\n".join(lines)

    @staticmethod
    def _read_paddle_line(engine, crop: np.ndarray) -> str:
        result = engine.ocr(crop, det=False, cls=True)
        if not result or not result[0]:
            return ""
        item = result[0][0] if isinstance(result[0], list) else result[0]
        text, confidence = item[0], item[1]
        return text.strip() if text and confidence > 0.25 else ""

    @staticmethod
    def _read_paddle_full(engine, image: np.ndarray) -> str:
        result = engine.ocr(image, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            if line and len(line) >= 2 and line[1] and line[1][0]:
                text_content = line[1][0].strip()
                confidence = line[1][1] if len(line[1]) > 1 else 1.0
                if text_content and confidence > 0.25:
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
        with self._paddle_locks_guard:
            if target_lang not in self._paddle_locks:
                self._paddle_locks[target_lang] = threading.Lock()
            lang_lock = self._paddle_locks[target_lang]
        if target_lang not in self._paddle_engines:
            with lang_lock:
                if target_lang not in self._paddle_engines:
                    from paddleocr import PaddleOCR
                    self._paddle_engines[target_lang] = PaddleOCR(
                        lang=target_lang,
                        use_angle_cls=True,
                        show_log=False,
                        det_db_thresh=0.15,
                        det_db_box_thresh=0.30,
                        det_db_unclip_ratio=2.0,
                    )
        return self._paddle_engines[target_lang], lang_lock
