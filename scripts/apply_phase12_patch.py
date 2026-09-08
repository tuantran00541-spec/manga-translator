from pathlib import Path


def replace_exact(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one anchor, found {count}")
    path.write_text(source.replace(old, new), encoding="utf-8")


parameters = Path("app/parameters.py")
adaptive = Path("app/detector/adaptive_focus_detector.py")

replace_exact(
    parameters,
    'DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE = _env_int(\n'
    '    "MANGA_DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE",\n'
    '    1344,\n'
    '    minimum=256,\n'
    '    maximum=4096,\n'
    ')\n',
    'DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE = _env_int(\n'
    '    "MANGA_DETECTOR_GRAYSCALE_FALLBACK_MAX_SOURCE_SIDE",\n'
    '    1344,\n'
    '    minimum=256,\n'
    '    maximum=4096,\n'
    ')\n'
    'DETECTOR_FOCUS_MAX_CHIPS = _env_int(\n'
    '    "MANGA_DETECTOR_FOCUS_MAX_CHIPS", 1, minimum=1, maximum=4\n'
    ')\n',
)

replace_exact(
    adaptive,
    'from app.parameters import (\n'
    '    DETECTOR_INPUT_SIZE,\n'
    '    DETECTOR_TALL_IMAGE_FACTOR,\n'
    '    DETECTOR_WINDOW_OVERLAP,\n'
    ')\n',
    'from app.parameters import (\n'
    '    DETECTOR_FOCUS_MAX_CHIPS,\n'
    '    DETECTOR_INPUT_SIZE,\n'
    '    DETECTOR_TALL_IMAGE_FACTOR,\n'
    '    DETECTOR_WINDOW_OVERLAP,\n'
    ')\n',
)

replace_exact(
    adaptive,
    'FOCUS_MAX_CHIPS = 2\n',
    'FOCUS_MAX_CHIPS = DETECTOR_FOCUS_MAX_CHIPS\n',
)

print("phase12 focus refinement budget patch applied")
