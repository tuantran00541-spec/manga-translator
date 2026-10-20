"""Clean every image in a folder: find its text, erase it, and write the result under the same name."""
import sys
from pathlib import Path

import cv2
import numpy as np

from app.detector.combined_detector import CombinedTextDetector
from app.inpaint.lama_inpainter import Inpainter


def main(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    detector, inpainter = CombinedTextDetector(), Inpainter()
    for path in sorted(src.glob("*.png")):
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(dst / path.name), inpainter.inpaint(image, detector.detect(image)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
