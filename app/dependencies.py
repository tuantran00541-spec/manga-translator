"""Shared singleton instances and dependency injection for FastAPI routers."""

from app.pipeline import ChapterPipeline
from app.ocr.multi_lang_ocr import MultiLangOCR

pipeline = ChapterPipeline()
ocr = MultiLangOCR()
