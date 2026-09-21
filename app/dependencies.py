from app.runtime_responsiveness import configure_local_cpu_headroom

configure_local_cpu_headroom()

from app.ocr.multi_lang_ocr import MultiLangOCR
from app.processing_pipeline_factory import build_processing_pipeline

pipeline = build_processing_pipeline()
ocr = MultiLangOCR()
