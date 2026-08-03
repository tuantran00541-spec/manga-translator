#!/usr/bin/env python3
"""Entry point for Manga Translator (local / Docker)."""

import uvicorn

from app.config import HOST, PORT, RELOAD, WORKERS, ensure_directories, check_models
from app.logging_config import logger
import cv2


def main() -> None:
    ensure_directories()
    cv2.setNumThreads(1)

    missing = check_models()
    if missing:
        logger.warning("=" * 60)
        logger.warning("MISSING MODELS — server will start but processing will fail")
        logger.warning("Missing files in models/:")
        for name in missing:
            logger.warning(f"  • {name}")
        logger.warning("See README.md → section 'Chuẩn bị Mô hình AI'")
        logger.warning("=" * 60)
    else:
        logger.info("All required ONNX models found in models/")

    logger.info(f"Starting Manga Translator on http://{HOST}:{PORT}")
    if HOST == "0.0.0.0":
        logger.warning(
            "Bound to 0.0.0.0 — accessible from network. "
            "Use firewall + auth if public."
        )

    # Single worker recommended: heavy CPU models + Playwright are not multi-process friendly
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        workers=1 if RELOAD else max(1, WORKERS),
        log_level="info",
    )


if __name__ == "__main__":
    main()
