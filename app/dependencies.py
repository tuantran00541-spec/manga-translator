from app.runtime_responsiveness import configure_local_cpu_headroom

configure_local_cpu_headroom()

from app.mask_recall_pipeline import MaskRecallOptimizedChapterPipeline
from app.ocr.multi_lang_ocr import MultiLangOCR

pipeline = MaskRecallOptimizedChapterPipeline()
ocr = MultiLangOCR()
