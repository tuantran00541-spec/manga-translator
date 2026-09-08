from pathlib import Path
import re

ROOT = Path('.')


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one source match, found {count}')
    return text.replace(old, new, 1)


def replace_count(text: str, old: str, new: str, expected: int, label: str) -> str:
    count = text.count(old)
    if count != expected:
        raise SystemExit(f'{label}: expected {expected} source matches, found {count}')
    return text.replace(old, new)


# B02: artifact transaction identity/generation, not revision magnitude.
path = ROOT / 'app/manifest_utils.py'
text = path.read_text(encoding='utf-8')
anchor = '''    return None


class PageArtifactTransaction:
'''
helper = '''    return None


def _manifest_page_artifact_state(
    chapter_dir: Path,
    page_index: int,
) -> tuple[bool, int, str | None]:
    """Read the persisted artifact generation and owning transaction identity."""
    try:
        raw = json.loads((chapter_dir / "manifest.json").read_text(encoding="utf-8"))
        pages = raw.get("pages") if isinstance(raw, dict) else None
        if isinstance(pages, list) and 0 <= page_index < len(pages):
            page = pages[page_index]
            if isinstance(page, dict):
                generation = int(page.get("artifact_generation") or 0)
                transaction_id = page.get("artifact_transaction_id")
                return True, generation, str(transaction_id) if transaction_id else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return False, 0, None


class PageArtifactTransaction:
'''
text = replace_once(text, anchor, helper, 'manifest artifact state helper')
text = replace_once(
    text,
    '''        self.records: list[dict[str, object]] = []
        self._closed = False
''',
    '''        self.records: list[dict[str, object]] = []
        self.target_artifact_generation: int | None = None
        self._closed = False
''',
    'transaction generation field',
)
text = replace_once(
    text,
    '''        payload = {
            "version": 1,
            "page_index": self.page_index,
            "target_clean_revision": self.target_clean_revision,
            "artifacts": self.records,
        }
''',
    '''        if self.target_artifact_generation is None:
            raise RuntimeError("Artifact transaction generation was not initialized")
        payload = {
            "version": 2,
            "page_index": self.page_index,
            "target_clean_revision": self.target_clean_revision,
            "transaction_id": self.transaction_id,
            "target_artifact_generation": self.target_artifact_generation,
            "artifacts": self.records,
        }
''',
    'transaction journal identity',
)
text = replace_once(
    text,
    '''    def __enter__(self) -> "PageArtifactTransaction":
        self.chapter_dir.mkdir(parents=True, exist_ok=True)
        try:
''',
    '''    def __enter__(self) -> "PageArtifactTransaction":
        self.chapter_dir.mkdir(parents=True, exist_ok=True)
        readable, current_generation, _current_transaction_id = (
            _manifest_page_artifact_state(self.chapter_dir, self.page_index)
        )
        if not readable:
            raise RuntimeError(
                f"Cannot establish artifact generation for page {self.page_index}"
            )
        self.target_artifact_generation = current_generation + 1
        try:
''',
    'transaction enter generation',
)
text = replace_once(
    text,
    '''    def commit(self) -> None:
''',
    '''    def mark_manifest_commit(self, page: dict) -> None:
        """Bind the manifest page to this exact artifact publication."""
        if self.target_artifact_generation is None:
            raise RuntimeError("Artifact transaction generation was not initialized")
        try:
            current_generation = int(page.get("artifact_generation") or 0)
            current_clean_revision = int(page.get("clean_revision") or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid page artifact generation state") from exc
        expected_previous_generation = self.target_artifact_generation - 1
        if current_generation != expected_previous_generation:
            raise RuntimeError("Page artifact generation changed during transaction")
        if current_clean_revision != self.target_clean_revision:
            raise RuntimeError(
                "Page clean revision changed before artifact transaction commit"
            )
        page["artifact_generation"] = self.target_artifact_generation
        page["artifact_transaction_id"] = self.transaction_id

    def commit(self) -> None:
''',
    'transaction manifest marker method',
)
old_exit = '''    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            current_revision = _manifest_page_clean_revision(
                self.chapter_dir, self.page_index
            )
            if current_revision is None:
                return
            if current_revision >= self.target_clean_revision:
                self.commit()
            else:
                self.rollback()
'''
new_exit = '''    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._closed:
            return
        readable, current_generation, current_transaction_id = (
            _manifest_page_artifact_state(self.chapter_dir, self.page_index)
        )
        if not readable or self.target_artifact_generation is None:
            return
        if current_generation > self.target_artifact_generation:
            self.commit()
            return
        if (
            current_generation == self.target_artifact_generation
            and current_transaction_id == self.transaction_id
        ):
            self.commit()
            return
        if (
            current_generation == self.target_artifact_generation
            and current_transaction_id not in {None, self.transaction_id}
        ):
            return
        self.rollback()
'''
text = replace_once(text, old_exit, new_exit, 'transaction exit ownership')
old_recovery = '''            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            page_index = int(payload["page_index"])
            target_revision = int(payload["target_clean_revision"])
            records = payload["artifacts"]
            if page_index < 0 or target_revision < 1 or not isinstance(records, list):
                continue
            current_revision = _manifest_page_clean_revision(
                chapter_dir, page_index
            )
            if current_revision is None:
                continue
            committed = current_revision >= target_revision
'''
new_recovery = '''            payload = json.loads(journal_path.read_text(encoding="utf-8"))
            page_index = int(payload["page_index"])
            target_revision = int(payload["target_clean_revision"])
            transaction_id = str(payload.get("transaction_id") or "")
            target_generation = int(payload.get("target_artifact_generation") or 0)
            records = payload["artifacts"]
            if (
                page_index < 0
                or target_revision < 1
                or not transaction_id
                or target_generation < 1
                or not isinstance(records, list)
            ):
                continue
            readable, current_generation, current_transaction_id = (
                _manifest_page_artifact_state(chapter_dir, page_index)
            )
            if not readable:
                continue
            if current_generation > target_generation:
                committed = True
            elif current_generation < target_generation:
                committed = False
            elif current_transaction_id == transaction_id:
                committed = True
            else:
                continue
'''
text = replace_once(text, old_recovery, new_recovery, 'transaction crash recovery')
old_defaults = '''        for key, default in revision_defaults.items():
            if page.get(key) is None:
                page[key] = default
                changed = True

        source_revision = int(page.get("source_revision") or 0)
'''
new_defaults = '''        for key, default in revision_defaults.items():
            if page.get(key) is None:
                page[key] = default
                changed = True
        if page.get("artifact_generation") is None:
            page["artifact_generation"] = 0
            changed = True

        source_revision = int(page.get("source_revision") or 0)
'''
text = replace_once(text, old_defaults, new_defaults, 'artifact generation schema default')
path.write_text(text, encoding='utf-8')


# B02/B03/B08: serialize skip writes, explicit stale outcomes, pixel protection.
path = ROOT / 'app/pipeline.py'
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    'from dataclasses import replace\n',
    'from dataclasses import replace\nfrom contextlib import ExitStack\n',
    'ExitStack import',
)
text = replace_once(
    text,
    'class ChapterPipeline:\n',
    '''class StaleProcessingStateError(RuntimeError):
    """No requested page committed because user state changed during processing."""


class ChapterPipeline:
''',
    'stale processing exception',
)
marker_pair = '''                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
'''
marker_replacement = '''                    artifact_tx.mark_manifest_commit(target_page)
                    save_manifest_raw(chapter_id, manifest)
                    artifact_tx.commit()
'''
text = replace_count(
    text,
    marker_pair,
    marker_replacement,
    6,
    'artifact manifest ownership call sites',
)
text = replace_once(
    text,
    '''        committed_indices: list[int] = []
        errors: list[tuple[int, Exception]] = []
''',
    '''        committed_indices: list[int] = []
        discarded_stale_indices: list[int] = []
        errors: list[tuple[int, Exception]] = []
''',
    'stale result collection',
)
old_commit_result = '''                    if self._commit_processed_page(
                        chapter_id,
                        processed_dir,
                        page_idx,
                        page_data,
                        snapshot,
                    ):
                        committed_indices.append(page_idx)
'''
new_commit_result = '''                    if self._commit_processed_page(
                        chapter_id,
                        processed_dir,
                        page_idx,
                        page_data,
                        snapshot,
                    ):
                        committed_indices.append(page_idx)
                    else:
                        discarded_stale_indices.append(page_idx)
'''
text = replace_once(text, old_commit_result, new_commit_result, 'stale commit accounting')
text = replace_once(
    text,
    '''        failed_indices = [item[0] for item in errors]
        with get_manifest_lock(chapter_id):
''',
    '''        failed_indices = [item[0] for item in errors]
        discarded_stale_indices = sorted(set(discarded_stale_indices))
        if errors:
            processing_outcome = "partial_failed" if committed_indices else "failed"
        elif discarded_stale_indices:
            processing_outcome = "partial_stale" if committed_indices else "stale_only"
        else:
            processing_outcome = "completed"

        with get_manifest_lock(chapter_id):
''',
    'processing outcome classification',
)
text = replace_once(
    text,
    '''                "committed_page_indices": sorted(committed_indices),
                "failed_page_indices": failed_indices,
                "workers": max_workers,
''',
    '''                "committed_page_indices": sorted(committed_indices),
                "failed_page_indices": failed_indices,
                "discarded_stale_page_indices": discarded_stale_indices,
                "discarded_stale_pages": [
                    {"page_index": page_index, "reason": "processing_state_changed"}
                    for page_index in discarded_stale_indices
                ],
                "outcome": processing_outcome,
                "workers": max_workers,
''',
    'processing outcome manifest',
)
text = replace_once(
    text,
    '''        if errors:
            first_exc = errors[0][1]
            if not committed_indices:
                raise RuntimeError(
                    f"Chapter {chapter_id}: All requested pages failed to process. "
                    f"First error (page {failed_indices[0]}): {first_exc}"
                ) from first_exc
            raise RuntimeError(
                f"Chapter {chapter_id}: Failed to process {len(errors)} page(s) "
                f"(indices: {failed_indices}). First error: {first_exc}"
            ) from first_exc

        return manifest
''',
    '''        if errors:
            first_exc = errors[0][1]
            if not committed_indices:
                raise RuntimeError(
                    f"Chapter {chapter_id}: All requested pages failed to process. "
                    f"First error (page {failed_indices[0]}): {first_exc}"
                ) from first_exc
            raise RuntimeError(
                f"Chapter {chapter_id}: Failed to process {len(errors)} page(s) "
                f"(indices: {failed_indices}). First error: {first_exc}"
            ) from first_exc

        if discarded_stale_indices and not committed_indices:
            raise StaleProcessingStateError(
                f"Chapter {chapter_id}: all requested processing output became stale "
                f"before commit (indices: {discarded_stale_indices})"
            )

        return manifest
''',
    'stale-only conflict',
)
mark_pattern = re.compile(
    r'    def mark_skipped\(self, chapter_id: str, page_indices: list\[int\], skipped: bool\) -> dict:\n.*?\n        return manifest\n\n    @staticmethod\n    def _sync_output_dir',
    re.S,
)
mark_match = mark_pattern.search(text)
if not mark_match:
    raise SystemExit('mark_skipped: source contract changed')
new_mark = '''    def mark_skipped(self, chapter_id: str, page_indices: list[int], skipped: bool) -> dict:
        indices = sorted({int(idx) for idx in page_indices if int(idx) >= 0})
        with ExitStack() as stack:
            for idx in indices:
                stack.enter_context(get_page_lock(chapter_id, idx))
            stack.enter_context(get_manifest_lock(chapter_id))
            manifest = load_manifest_raw(chapter_id)
            for idx in indices:
                if 0 <= idx < len(manifest["pages"]):
                    page = manifest["pages"][idx]
                    changed = bool(page.get("skipped", False)) != bool(skipped)
                    page["skipped"] = skipped
                    if skipped:
                        if page.get("clean") is not None or page.get("boxes"):
                            changed = True
                        page["clean"] = None
                        page["boxes"] = []
                    if changed:
                        bump_page_revision(page, "clean_revision")
                    invalidate_page_render(manifest, idx)
            save_manifest_raw(chapter_id, manifest)
            self._sync_output_dir(chapter_id, manifest, indices)
        return manifest

    @staticmethod
    def _sync_output_dir'''
text = text[:mark_match.start()] + new_mark + text[mark_match.end():]
text = replace_once(
    text,
    '''        if excluded_regions:
            detected = [b for b in detected if not self._box_in_excluded(b, excluded_regions)]
''',
    '''        # Exclusion rectangles protect pixels rather than suppressing
        # detector/OCR evidence. Automatic erase authority is clipped later.
''',
    'excluded-region detector suppression',
)
text = replace_once(
    text,
    '        clean_image = self.inpainter.inpaint(image, effective_boxes)\n',
    '''        clean_image = self.inpainter.inpaint(
            image,
            effective_boxes,
            protected_regions=excluded_regions,
        )
''',
    'auto inpaint protected regions',
)
old_resize = '''            if bin_mask is not None and bin_mask.shape[:2] != (img_h, img_w):
                bin_mask = cv2.resize(bin_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                bin_mask = (
                    (bin_mask > MANUAL_MASK_THRESHOLD).astype(np.uint8) * 255
                )
'''
new_resize = '''            if bin_mask is not None and bin_mask.shape[:2] != (img_h, img_w):
                raise ValueError(
                    "Repaint mask dimensions "
                    f"{bin_mask.shape[:2]} must exactly match page dimensions "
                    f"{(img_h, img_w)}"
                )
'''
text = replace_once(text, old_resize, new_resize, 'strict repaint mask dimensions')
box_method = re.compile(
    r'\n    @staticmethod\n    def _box_in_excluded\(box, excluded_regions: list\[dict\]\) -> bool:\n.*\Z',
    re.S,
)
text, n = box_method.subn('', text)
if n != 1:
    raise SystemExit(f'deprecated exclusion predicate: expected one tail method, found {n}')
path.write_text(text, encoding='utf-8')


# B08: subtract protected rectangles after build_mask dilation.
path = ROOT / 'app/inpaint/lama_inpainter.py'
text = path.read_text(encoding='utf-8')
old_sig = '    def inpaint(self, image: np.ndarray, boxes: list[BubbleBox]) -> np.ndarray:\n'
new_sig = '''    @staticmethod
    def _subtract_protected_regions(
        local_mask: np.ndarray,
        crop_box: tuple[int, int, int, int],
        protected_regions: list[dict] | None,
    ) -> np.ndarray:
        """Remove protected page pixels after all automatic mask dilation."""
        if local_mask is None or not protected_regions:
            return local_mask
        cx1, cy1, cx2, cy2 = (int(value) for value in crop_box)
        clipped = local_mask.copy()
        for region in protected_regions:
            if not isinstance(region, dict):
                continue
            try:
                rx1, rx2 = sorted((int(region.get("x1", 0)), int(region.get("x2", 0))))
                ry1, ry2 = sorted((int(region.get("y1", 0)), int(region.get("y2", 0))))
            except (TypeError, ValueError):
                continue
            ix1, iy1 = max(cx1, rx1), max(cy1, ry1)
            ix2, iy2 = min(cx2, rx2), min(cy2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            clipped[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] = 0
        return clipped

    def inpaint(
        self,
        image: np.ndarray,
        boxes: list[BubbleBox],
        *,
        protected_regions: list[dict] | None = None,
    ) -> np.ndarray:
'''
text = replace_once(text, old_sig, new_sig, 'protected inpaint signature')
text = replace_once(
    text,
    '''            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)

            result = self._smart_paint_region(result, local_mask, crop_box)
''',
    '''            local_mask = build_mask((cy2 - cy1, cx2 - cx1), local_boxes, crop_img)
            local_mask = self._subtract_protected_regions(
                local_mask,
                crop_box,
                protected_regions,
            )

            result = self._smart_paint_region(result, local_mask, crop_box)
''',
    'protected mask subtraction',
)
path.write_text(text, encoding='utf-8')


# B09: signature covers render eligibility.
path = ROOT / 'app/render/identity.py'
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    '''        "translation": str(obj.get("translation") or ""),
        "style": obj.get("style") or {},
''',
    '''        "translation": str(obj.get("translation") or ""),
        "style": obj.get("style") or {},
        "source_missing": bool(obj.get("source_missing", False)),
''',
    'render source_missing identity',
)
path.write_text(text, encoding='utf-8')


# B10: bounded async upload read and off-event-loop decode.
path = ROOT / 'app/routers/editor.py'
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    'from app.security import validate_chapter_id\n',
    'from app.security import MAX_IMAGE_PIXELS, MAX_REQUEST_BYTES, validate_chapter_id\nfrom app.upload_utils import read_upload_limited\n',
    'editor upload security imports',
)
text = replace_once(
    text,
    '''router = APIRouter(prefix="/api", tags=["editor"])


''',
    '''router = APIRouter(prefix="/api", tags=["editor"])


def _decode_repaint_mask_payload(mask_bytes: bytes) -> np.ndarray:
    """Decode one bounded 8-bit mask without running OpenCV on the event loop."""
    if not mask_bytes:
        raise ValueError("Empty mask payload")
    encoded = np.frombuffer(mask_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Invalid repaint mask image: failed to decode")
    if decoded.dtype != np.uint8:
        raise ValueError("Repaint mask must be an 8-bit image")
    height, width = decoded.shape[:2]
    if height <= 0 or width <= 0 or height * width > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Repaint mask dimensions {width}x{height} exceed the decoded pixel limit"
        )
    if decoded.ndim == 2:
        mask_array = decoded
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        mask_array = decoded[:, :, 3]
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        mask_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Repaint mask must be 8-bit grayscale, BGR, or BGRA")
    if not np.any(mask_array > 0):
        raise ValueError("Repaint mask is empty")
    return np.ascontiguousarray(mask_array)


''',
    'repaint decode helper',
)
route_old = '''    try:
        image = read_image(img_path)
        img_h, img_w = image.shape[:2]
    except Exception as exc:
        logger.opt(exception=True).error("Chapter {} page {} operation 'repaint_mask' cannot read base image: {}", chapter_id, page_index, exc)
        raise HTTPException(500, f"Cannot read base page image: {exc}") from exc

    mask_bytes = await mask.read()
    if not mask_bytes:
        raise HTTPException(400, "Empty mask payload")
    logger.info(
        "Chapter {} page {}: repaint mask ({} bytes, mode={})",
        chapter_id,
        page_index,
        len(mask_bytes),
        mode,
    )

    encoded = np.frombuffer(mask_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(400, "Invalid repaint mask image: failed to decode")

    if decoded.ndim == 3 and decoded.shape[2] == 4:
        mask_array = decoded[:, :, 3]
    elif decoded.ndim == 3:
        mask_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    else:
        mask_array = decoded

    if mask_array.shape[:2] != (img_h, img_w):
        try:
            mask_array = cv2.resize(mask_array, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        except Exception as exc:
            raise HTTPException(400, f"Mask dimensions {mask_array.shape[:2]} cannot be matched to page dimensions {(img_h, img_w)}") from exc

    if not np.any(mask_array > 0):
        raise HTTPException(400, "Repaint mask is empty")

'''
route_new = '''    try:
        mask_bytes = await read_upload_limited(mask, MAX_REQUEST_BYTES)
    except HTTPException:
        raise
    if not mask_bytes:
        raise HTTPException(400, "Empty mask payload")
    logger.info(
        "Chapter {} page {}: repaint mask ({} bytes, mode={})",
        chapter_id,
        page_index,
        len(mask_bytes),
        mode,
    )

    try:
        mask_array = await run_in_threadpool(_decode_repaint_mask_payload, mask_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

'''
text = replace_once(text, route_old, route_new, 'repaint event-loop preparation')
text = replace_once(
    text,
    '''        manifest = await run_in_threadpool(
            pipeline.repaint_mask,
            chapter_id,
            page_index,
            mask_array,
            force_lama=mode == "lama",
        )
        return urlify_manifest(manifest)
    except Exception as exc:
''',
    '''        manifest = await run_in_threadpool(
            pipeline.repaint_mask,
            chapter_id,
            page_index,
            mask_array,
            force_lama=mode == "lama",
        )
        return urlify_manifest(manifest)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
''',
    'repaint validation error mapping',
)
path.write_text(text, encoding='utf-8')


# B03 route semantics: stale-only is an HTTP 409 conflict.
path = ROOT / 'app/routers/chapters.py'
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    'from app.parameters import PIPELINE_DEFAULT_WORKERS\n',
    'from app.parameters import PIPELINE_DEFAULT_WORKERS\nfrom app.pipeline import StaleProcessingStateError\n',
    'chapters stale exception import',
)
text = replace_once(
    text,
    '''    try:
        manifest = pipeline.process_pages(req.chapter_id, req.page_indices, workers=workers)
        return urlify_manifest(manifest)
    except RuntimeError as exc:
''',
    '''    try:
        manifest = pipeline.process_pages(req.chapter_id, req.page_indices, workers=workers)
        return urlify_manifest(manifest)
    except StaleProcessingStateError as exc:
        logger.warning(
            "Chapter {} pages {} operation 'process_pages' discarded stale output: {}",
            req.chapter_id,
            req.page_indices,
            exc,
        )
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
''',
    'stale route response',
)
path.write_text(text, encoding='utf-8')


# Regression coverage for B02/B03/B07/B08/B09/B10.
sanity = r'''from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def manifest_bytes(clean_revision: int, artifact_generation: int = 0, transaction_id=None) -> bytes:
    page = {"clean_revision": clean_revision, "artifact_generation": artifact_generation}
    if transaction_id:
        page["artifact_transaction_id"] = transaction_id
    return json.dumps(
        {"schema_version": 3, "chapter_id": "a1b2c3d4", "pages": [page]},
        separators=(",", ":"),
    ).encode("utf-8")


def transaction_identity_checks() -> None:
    import app.manifest_utils as mu

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live = root / "clean.png"
        live.write_bytes(b"OLD")
        (root / "manifest.json").write_bytes(manifest_bytes(1))
        tx = mu.PageArtifactTransaction(root, 0, [live], 2)
        tx.__enter__()
        live.write_bytes(b"NEW")
        (root / "manifest.json").write_bytes(manifest_bytes(2))
        tx.__exit__(RuntimeError, RuntimeError("commit rejected"), None)
        check(live.read_bytes() == b"OLD", "unrelated revision accepted as artifact commit")
        check(not tx.journal_path.exists(), "rolled-back transaction journal survived")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live = root / "clean.png"
        live.write_bytes(b"OLD")
        (root / "manifest.json").write_bytes(manifest_bytes(1))
        first = mu.PageArtifactTransaction(root, 0, [live], 2)
        first.__enter__()
        live.write_bytes(b"FIRST")
        manifest = json.loads((root / "manifest.json").read_text())
        page = manifest["pages"][0]
        page["clean_revision"] = 2
        first.mark_manifest_commit(page)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        second = mu.PageArtifactTransaction(root, 0, [live], 3)
        second.__enter__()
        live.write_bytes(b"SECOND")
        manifest = json.loads((root / "manifest.json").read_text())
        page = manifest["pages"][0]
        page["clean_revision"] = 3
        second.mark_manifest_commit(page)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        second.commit()

        check(mu.recover_page_artifact_transactions(root) == 1, "superseded journal not resolved")
        check(live.read_bytes() == b"SECOND", "older recovery overwrote newer artifact")
        check(not first.journal_path.exists(), "superseded journal survived")


def stale_outcome_checks() -> None:
    import app.manifest_utils as mu
    import app.pipeline as pipeline_module
    from app.pipeline import ChapterPipeline, StaleProcessingStateError

    class DummyInpainter:
        session_loaded = False
        _prefer_dynamic = False
        serialized_inference = True
        lama_model_path = ""

    class FakePipeline(ChapterPipeline):
        def __init__(self):
            super().__init__()
            self._detector = object()
            self._inpainter = DummyInpainter()

        def _shared_seam_detections(self, chapter_id, work_items):
            return {}, set(), {
                "attempts": 0,
                "failures": 0,
                "wall_ms": 0.0,
                "bubble_model_ms": 0.0,
                "text_model_ms": 0.0,
                "mser_ms": 0.0,
                "text_grayscale_fallback_runs": 0,
                "text_grayscale_fallback_ms": 0.0,
                "detector_total_ms": 0.0,
            }

        def _process_page(self, *args, **kwargs):
            return {"boxes": []}

        def _commit_processed_page(self, *args, **kwargs):
            return False

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chapter = root / "a1b2c3d4"
        chapter.mkdir()
        original = chapter / "original.png"
        original.write_bytes(b"fixture")
        manifest = {
            "schema_version": 3,
            "chapter_id": "a1b2c3d4",
            "pages": [{
                "original": original.as_posix(),
                "clean": None,
                "boxes": [],
                "skipped": False,
                "excluded_regions": [],
                "clean_revision": 0,
                "source_revision": 1,
                "process_revision": 0,
                "render_revision": 0,
                "artifact_generation": 0,
            }],
        }
        (chapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        old_mu_dir, old_pipeline_dir = mu.PROCESSED_DIR, pipeline_module.PROCESSED_DIR
        mu.PROCESSED_DIR = root
        pipeline_module.PROCESSED_DIR = root
        try:
            try:
                FakePipeline().process_pages("a1b2c3d4", [0], workers=1)
            except StaleProcessingStateError:
                pass
            else:
                raise AssertionError("stale-only process returned success")
            saved = json.loads((chapter / "manifest.json").read_text())
            run = saved["last_processing_run"]
            check(run["outcome"] == "stale_only", "stale-only outcome missing")
            check(run["discarded_stale_page_indices"] == [0], "stale page index missing")
            check(
                run["discarded_stale_pages"] == [{"page_index": 0, "reason": "processing_state_changed"}],
                "stale reason missing",
            )
        finally:
            mu.PROCESSED_DIR = old_mu_dir
            pipeline_module.PROCESSED_DIR = old_pipeline_dir


def protected_region_checks() -> None:
    from app.inpaint.lama_inpainter import Inpainter

    mask = np.full((100, 100), 255, np.uint8)
    clipped = Inpainter._subtract_protected_regions(
        mask,
        (0, 0, 100, 100),
        [{"x1": 0, "y1": 0, "x2": 20, "y2": 100}],
    )
    check(not np.any(clipped[:, :20] > 0), "protected pixels retained destructive authority")
    check(np.all(clipped[:, 20:] > 0), "protection removed unrelated mask pixels")
    source = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check("_box_in_excluded" not in source, "center-based exclusion predicate remains")
    check("protected_regions=excluded_regions" in source, "auto inpaint does not receive protection")


def recovery_cache_checks() -> None:
    from app.detector.recovery import SecondaryTextRecovery

    class FakeMser:
        def __init__(self):
            self.calls = 0

        def detectRegions(self, gray):
            self.calls += 1
            return [], np.array([[2, 2, 4, 4]], dtype=np.int32)

    recovery = SecondaryTextRecovery()
    fake = FakeMser()
    recovery._mser = fake
    image = np.zeros((32, 32, 3), np.uint8)
    gray = np.zeros((32, 32), np.uint8)
    first = recovery._extract_primitives(image, gray)
    second = recovery._extract_primitives(image, gray)
    check(fake.calls == 1, "same-image MSER extraction repeated")
    check(first is second, "same-image primitive cache was not reused")


def render_identity_checks() -> None:
    from app.render.identity import render_input_signature

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.png"
        base.write_bytes(b"base")
        manifest = {
            "pages": [{
                "original": base.as_posix(),
                "clean": None,
                "source_revision": 1,
                "process_revision": 1,
                "clean_revision": 1,
                "render_revision": 0,
                "skipped": False,
                "text_objects": [{
                    "id": "obj",
                    "shape": "rectangle",
                    "region": {"x1": 1, "y1": 1, "x2": 5, "y2": 5},
                    "translation": "hello",
                    "style": {},
                    "source_missing": False,
                }],
            }],
        }
        before = render_input_signature(manifest, 0)
        manifest["pages"][0]["text_objects"][0]["source_missing"] = True
        after = render_input_signature(manifest, 0)
        check(before != after, "source_missing does not invalidate render identity")


def repaint_input_checks() -> None:
    from app.routers.editor import _decode_repaint_mask_payload

    mask = np.zeros((16, 20), np.uint8)
    mask[3:7, 4:9] = 255
    ok, encoded = cv2.imencode(".png", mask)
    check(ok, "could not encode repaint fixture")
    decoded = _decode_repaint_mask_payload(encoded.tobytes())
    check(decoded.shape == mask.shape and np.array_equal(decoded, mask), "repaint decode changed pixels")
    editor_source = (ROOT / "app/routers/editor.py").read_text(encoding="utf-8")
    pipeline_source = (ROOT / "app/pipeline.py").read_text(encoding="utf-8")
    check("await mask.read()" not in editor_source, "repaint route still performs unbounded read")
    check("read_upload_limited(mask, MAX_REQUEST_BYTES)" in editor_source, "repaint upload is not bounded")
    check("_decode_repaint_mask_payload, mask_bytes" in editor_source, "repaint decode is not offloaded")
    check("must exactly match page dimensions" in pipeline_source, "mismatched repaint masks are still stretched")


def main() -> None:
    transaction_identity_checks()
    stale_outcome_checks()
    protected_region_checks()
    recovery_cache_checks()
    render_identity_checks()
    repaint_input_checks()
    print("backend second-pass sanity: PASS")


if __name__ == "__main__":
    main()
'''
(ROOT / 'scripts/backend_second_pass_sanity.py').write_text(sanity, encoding='utf-8')


# Keep the normal cross-platform foundation gate authoritative.
path = ROOT / '.github/workflows/backend-foundation-gate.yml'
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    '''      - app/routers/render_commit.py
      - app/routers/export.py
''',
    '''      - app/routers/render_commit.py
      - app/routers/editor.py
      - app/render/identity.py
      - app/routers/export.py
''',
    'foundation paths second-pass source',
)
text = replace_once(
    text,
    '''      - scripts/backend_foundation_sanity.py
      - scripts/backend_phase4_sanity.py
''',
    '''      - scripts/backend_foundation_sanity.py
      - scripts/backend_second_pass_sanity.py
      - scripts/backend_phase4_sanity.py
''',
    'foundation path second-pass sanity',
)
text = replace_once(
    text,
    '''          app/detector/mask_builder.py app/inpaint/lama_inpainter.py app/pipeline.py
          app/routers/render_commit.py app/routers/export.py
          scripts/backend_foundation_sanity.py scripts/backend_phase4_sanity.py
''',
    '''          app/detector/mask_builder.py app/inpaint/lama_inpainter.py app/pipeline.py
          app/routers/render_commit.py app/routers/editor.py app/render/identity.py app/routers/export.py
          scripts/backend_foundation_sanity.py scripts/backend_second_pass_sanity.py scripts/backend_phase4_sanity.py
''',
    'foundation compile second-pass',
)
text = replace_once(
    text,
    '''          python scripts/backend_foundation_sanity.py
          python scripts/backend_phase4_sanity.py
''',
    '''          python scripts/backend_foundation_sanity.py
          python scripts/backend_second_pass_sanity.py
          python scripts/backend_phase4_sanity.py
''',
    'foundation run second-pass',
)
path.write_text(text, encoding='utf-8')

print('backend second-pass apply patch prepared')
