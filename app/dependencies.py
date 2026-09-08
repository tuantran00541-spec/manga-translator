from app.runtime_responsiveness import configure_local_cpu_headroom

configure_local_cpu_headroom()

from app.optimized_pipeline import OptimizedChapterPipeline
from app.ocr.multi_lang_ocr import MultiLangOCR

pipeline = OptimizedChapterPipeline()
ocr = MultiLangOCR()
