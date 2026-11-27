from __future__ import annotations

import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.inpaint.clustering import compute_manual_crop_region
from app.image_io import read_image, write_image
from app.manifest_utils import atomic_replace
from app.parameters import MANUAL_MASK_THRESHOLD
from app.pipeline import ChapterPipeline
from app.region_policy import subtract_regions_from_mask


class OptimizedChapterPipeline(ChapterPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._repaint_stats_local = threading.local()
        self._last_repaint_detector_stats_fallback: dict | None = None

    def _stats_local(self) -> threading.local:
        local = self.__dict__.get("_repaint_stats_local")
        if local is None:
            local = threading.local()
            self.__dict__["_repaint_stats_local"] = local
        return local

    @property
    def _last_repaint_detector_stats(self) -> dict | None:
        return getattr(
            self._stats_local(),
            "stats",
            self.__dict__.get("_last_repaint_detector_stats_fallback"),
        )

    @_last_repaint_detector_stats.setter
    def _last_repaint_detector_stats(self, value: dict | None) -> None:
        self._stats_local().stats = value
        self.__dict__["_last_repaint_detector_stats_fallback"] = value

    def _exact_manual_repaint_candidate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        force_lama: bool,
    ) -> np.ndarray:
        binary = (mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
        user_pixels = int(np.count_nonzero(binary))
        if user_pixels == 0:
            return image.copy()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        result = image.copy()
        scratch = image.copy()
        h, w = image.shape[:2]
        used_components = 0

        valid_labels: list[tuple[int, int, int, int, int]] = []
        for label in range(1, num_labels):
            x, y, bbox_w, bbox_h, area = (int(v) for v in stats[label])
            if area > 0 and bbox_w > 0 and bbox_h > 0:
                valid_labels.append((label, x, y, bbox_w, bbox_h))

        total_valid = len(valid_labels)
        for idx, (label, x, y, bbox_w, bbox_h) in enumerate(valid_labels):
            crop_box = compute_manual_crop_region(
                x,
                y,
                x + bbox_w,
                y + bbox_h,
                w,
                h,
            )
            cx1, cy1, cx2, cy2 = crop_box
            local_mask = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.uint8)
            component = labels[cy1:cy2, cx1:cx2] == label
            local_mask[component] = 255
            other = (binary[cy1:cy2, cx1:cx2] > 0) & ~component
            hole_mask = (other.astype(np.uint8) * 255) if np.any(other) else None

            paint_kwargs = {"feather": False, "force_lama": force_lama}
            if hole_mask is not None:
                import inspect
                if "hole" in inspect.signature(self.inpainter._smart_paint_region).parameters:
                    paint_kwargs["hole"] = hole_mask

            component_candidate = self.inpainter._smart_paint_region(
                scratch,
                local_mask,
                crop_box,
                **paint_kwargs,
            )
            result[cy1:cy2, cx1:cx2][component] = component_candidate[
                cy1:cy2, cx1:cx2
            ][component]
            if idx + 1 < total_valid:
                scratch[cy1:cy2, cx1:cx2] = image[cy1:cy2, cx1:cx2]
            used_components += 1

        if force_lama:
            self._last_repaint_detector_stats = {
                "mask_mode": "exact_user_mask",
                "detector_source": "not_used",
                "mask_components": int(used_components),
                "user_mask_pixels": user_pixels,
                "inference_mask_pixels": user_pixels,
                "detector_added_pixels": 0,
                "dilation_pixels": 0,
                "feather_outside_mask": False,
            }

        return result

    def _do_reinpaint(
        self,
        processed_dir: Path,
        img_path: Path,
        image: np.ndarray,
        boxes: list[dict],
        manual_mask_posix: str | None = None,
        manual_lama_mask_posix: str | None = None,
        *,
        reuse_auto_clean: bool = False,
        apply_manual_mask: bool = True,
        preserve_regions: list[dict] | None = None,
    ) -> str:
        if not apply_manual_mask:
            clean_path_posix = super()._do_reinpaint(
                processed_dir,
                img_path,
                image,
                boxes,
                manual_mask_posix=manual_mask_posix,
                manual_lama_mask_posix=manual_lama_mask_posix,
                reuse_auto_clean=reuse_auto_clean,
                apply_manual_mask=False,
                preserve_regions=preserve_regions,
            )
            if preserve_regions and reuse_auto_clean:
                clean_image = read_image(Path(clean_path_posix))
                self._restore_preserve_pixels(clean_image, image, preserve_regions)
                self._atomic_write_clean(processed_dir, Path(clean_path_posix), clean_image)
            return clean_path_posix

        standard_mask_path = (
            Path(manual_mask_posix)
            if manual_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=False)
        )
        lama_mask_path = (
            Path(manual_lama_mask_posix)
            if manual_lama_mask_posix
            else self._manual_mask_path(processed_dir, img_path, force_lama=True)
        )

        standard_mask = self._read_manual_mask(
            standard_mask_path,
            image.shape[:2],
        )
        lama_mask = self._read_manual_mask(
            lama_mask_path,
            image.shape[:2],
        )
        standard_mask = subtract_regions_from_mask(
            standard_mask,
            preserve_regions,
        )
        lama_mask = subtract_regions_from_mask(
            lama_mask,
            preserve_regions,
        )

        clean_path_posix = super()._do_reinpaint(
            processed_dir,
            img_path,
            image,
            boxes,
            manual_mask_posix=None,
            manual_lama_mask_posix=None,
            reuse_auto_clean=reuse_auto_clean,
            apply_manual_mask=False,
            preserve_regions=preserve_regions,
        )

        needs_rewrite = bool(preserve_regions)
        clean_image = None

        if (
            standard_mask is not None
            and np.any(standard_mask > MANUAL_MASK_THRESHOLD)
        ):
            clean_image = read_image(Path(clean_path_posix))
            standard_candidate = self._exact_manual_repaint_candidate(
                clean_image,
                standard_mask,
                force_lama=False,
            )
            authority = standard_mask > MANUAL_MASK_THRESHOLD
            clean_image[authority] = standard_candidate[authority]
            needs_rewrite = True

        if lama_mask is not None and np.any(lama_mask > MANUAL_MASK_THRESHOLD):
            if clean_image is None:
                clean_image = read_image(Path(clean_path_posix))
            lama_candidate = self._exact_manual_repaint_candidate(
                image,
                lama_mask,
                force_lama=True,
            )
            authority = lama_mask > MANUAL_MASK_THRESHOLD
            clean_image[authority] = lama_candidate[authority]
            needs_rewrite = True

        if preserve_regions:
            if clean_image is None:
                clean_image = read_image(Path(clean_path_posix))
            self._restore_preserve_pixels(clean_image, image, preserve_regions)

        if self._restore_mask_path(processed_dir, img_path).exists():
            if clean_image is None:
                clean_image = read_image(Path(clean_path_posix))
            needs_rewrite = (
                self._apply_restore_mask(clean_image, image, processed_dir, img_path)
                or needs_rewrite
            )

        if needs_rewrite and clean_image is not None:
            self._atomic_write_clean(processed_dir, Path(clean_path_posix), clean_image)

        return clean_path_posix

    @staticmethod
    def _atomic_write_clean(
        processed_dir: Path,
        clean_path: Path,
        clean_image: np.ndarray,
    ) -> None:
        tmp_clean_path = processed_dir / (
            f"{clean_path.name}.{uuid.uuid4().hex[:12]}.tmp.png"
        )
        try:
            write_image(tmp_clean_path, clean_image)
            atomic_replace(tmp_clean_path, clean_path)
        finally:
            try:
                tmp_clean_path.unlink(missing_ok=True)
            except OSError:
                pass
