from pathlib import Path
import runpy

patch_path = Path("scripts/apply_phase6_patch.py")
source = patch_path.read_text(encoding="utf-8")
old = '''    "    SMART_FILL_CANNY_LOW,\\n",
    "    SMART_FILL_CANNY_LOW,\\n    SMART_FILL_CHROMA_STD_MAX,\\n",
    "lama parameter import",
'''
new = '''    "    SMART_FILL_CANNY_HIGH,\\n    SMART_FILL_CANNY_LOW,\\n    SMART_FILL_CLEAN_RING_MARGIN,\\n",
    "    SMART_FILL_CANNY_HIGH,\\n    SMART_FILL_CANNY_LOW,\\n    SMART_FILL_CHROMA_STD_MAX,\\n    SMART_FILL_CLEAN_RING_MARGIN,\\n",
    "lama parameter import",
'''
if source.count(old) != 1:
    raise RuntimeError(f"phase6 staging import repair expected one match, got {source.count(old)}")
patch_path.write_text(source.replace(old, new, 1), encoding="utf-8")
runpy.run_path(str(patch_path), run_name="__main__")
