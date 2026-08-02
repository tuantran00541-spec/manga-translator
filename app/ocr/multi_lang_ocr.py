import threading
import numpy as np
import cv2
from PIL import Image
from app.logging_config import logger


class MultiLangOCR:
    def __init__(self):
        self._manga_ocr = None
        self._manga_failed = False
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
        lang_l = (lang or "").lower()
        if lang_l in ("ja", "japan", "japanese"):
            try:
                return self._read_manga_ocr(image)
            except Exception as e:
                logger.warning(
                    "manga-ocr failed (%s); falling back to PaddleOCR japan", e
                )
                return self._read_paddle(image, "ja")
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
        if self._manga_failed:
            raise RuntimeError("manga-ocr previously failed to load")

        if self._manga_ocr is None:
            with self._manga_lock:
                if self._manga_ocr is None and not self._manga_failed:
                    self._manga_ocr = self._load_manga_ocr()

        pil_img = Image.fromarray(image)
        return self._manga_ocr(pil_img).strip()

    def _load_manga_ocr(self):
        """Load MangaOcr; try package first, then manual ViTImageProcessor path."""
        try:
            from manga_ocr import MangaOcr
            return MangaOcr()
        except Exception as e1:
            logger.warning("MangaOcr() failed: %s — trying manual load", e1)

        try:
            return self._load_manga_ocr_manual()
        except Exception as e2:
            self._manga_failed = True
            logger.error("manga-ocr manual load also failed: %s", e2)
            raise RuntimeError(
                "manga-ocr không tương thích với transformers hiện tại. "
                "Cài: pip install -U 'manga-ocr>=0.1.14' "
                "hoặc pin transformers==4.36.2"
            ) from e2

    @staticmethod
    def _load_manga_ocr_manual():
        """Bypass AutoFeatureExtractor; use ViTImageProcessor (transformers >= 4.x)."""
        import torch
        from transformers import (
            ViTImageProcessor,
            AutoTokenizer,
            VisionEncoderDecoderModel,
        )

        model_id = "kha-white/manga-ocr-base"
        logger.info("Loading manga-ocr manually from %s", model_id)

        processor = ViTImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
        else:
            device = "cpu"

        class _ManualMangaOcr:
            def __call__(self, img):
                if isinstance(img, str):
                    img = Image.open(img)
                img = img.convert("L").convert("RGB")
                pixel_values = processor(img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)
                with torch.no_grad():
                    generated = model.generate(pixel_values, max_length=300)
                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                # manga-ocr post-process: strip special spacing artifacts
                text = text.replace("‥", "…").strip()
                return text

        return _ManualMangaOcr()

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
            "japanese": "japan",
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
