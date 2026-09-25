#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import zipfile

import cv2
import numpy as np


def _basename(value: object) -> str:
    text = str(value or "")
    return PureWindowsPath(text).name or PurePosixPath(text).name


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class BundleReader:
    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path) if path.is_file() else None
        if self._zip:
            self._entries = {
                PurePosixPath(name).name: name
                for name in self._zip.namelist()
                if not name.endswith("/")
            }
        else:
            self._entries = {item.name: item for item in path.rglob("*") if item.is_file()}

    def close(self) -> None:
        if self._zip:
            self._zip.close()

    def read(self, name: str) -> bytes | None:
        target = self._entries.get(_basename(name))
        if target is None:
            return None
        return self._zip.read(target) if self._zip else Path(target).read_bytes()

    def manifest(self) -> dict:
        raw = self.read("manifest.json")
        if raw is None:
            raise ValueError("Bundle does not contain manifest.json")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("pages"), list):
            raise ValueError("Invalid chapter manifest")
        return value


def _decode(raw: bytes | None, flags: int) -> np.ndarray | None:
    if raw is None:
        return None
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    return image if image is not None and image.size else None


def audit_bundle(path: Path, *, strict_manual_authority: bool = False) -> dict:
    reader = BundleReader(path)
    try:
        manifest = reader.manifest()
        pages = manifest["pages"]
        counters = {
            "slices": len(pages), "processed_slices": 0, "skipped_slices": 0,
            "review_slices": 0, "residue_slices": 0, "manual_correction_slices": 0,
            "boxes": 0, "safe_boxes": 0, "review_boxes": 0,
            "deferred_boxes": 0, "residue_regions": 0,
        }
        evidence = {
            "paired_auto_final_slices": 0, "manual_mask_pixels": 0,
            "changed_pixels": 0, "outside_manual_changed_pixels": 0,
            "missing_clean_slices": [], "invalid_mask_slices": [],
        }

        for index, page in enumerate(pages):
            skipped = bool(page.get("skipped"))
            counters["skipped_slices" if skipped else "processed_slices"] += 1
            counters["review_slices"] += int(bool(page.get("needs_review")))
            residue = page.get("residue_regions") or []
            counters["residue_regions"] += len(residue)
            counters["residue_slices"] += int(bool(residue))
            boxes = [box for box in (page.get("boxes") or []) if isinstance(box, dict)]
            counters["boxes"] += len(boxes)
            counters["safe_boxes"] += sum(bool(box.get("safe_to_inpaint")) for box in boxes)
            counters["review_boxes"] += sum(bool(box.get("needs_review")) for box in boxes)
            counters["deferred_boxes"] += sum(bool(box.get("deferred_reason")) for box in boxes)
            if skipped:
                continue
            final = _decode(reader.read(_basename(page.get("clean"))), cv2.IMREAD_COLOR)
            if final is None:
                evidence["missing_clean_slices"].append(index)
                continue
            manual_names = [_basename(page.get("manual_mask")), _basename(page.get("manual_lama_mask"))]
            manual_masks = [_decode(reader.read(name), cv2.IMREAD_GRAYSCALE) for name in manual_names if name]
            manual_masks = [mask for mask in manual_masks if mask is not None]
            if not manual_masks:
                continue
            if any(mask.shape != final.shape[:2] for mask in manual_masks):
                evidence["invalid_mask_slices"].append(index)
                continue
            counters["manual_correction_slices"] += 1
            authority = np.maximum.reduce(manual_masks) > 10
            evidence["manual_mask_pixels"] += int(np.count_nonzero(authority))
            automatic = _decode(
                reader.read(f"auto_clean_{_basename(page.get('original'))}"),
                cv2.IMREAD_COLOR,
            )
            if automatic is None or automatic.shape != final.shape:
                continue
            evidence["paired_auto_final_slices"] += 1
            changed = np.any(automatic != final, axis=2)
            evidence["changed_pixels"] += int(np.count_nonzero(changed))
            evidence["outside_manual_changed_pixels"] += int(np.count_nonzero(changed & ~authority))

        processed = max(1, counters["processed_slices"])
        metrics = {
            "automatic_completion_proxy": 1.0 - counters["manual_correction_slices"] / processed,
            "manual_correction_slice_rate": counters["manual_correction_slices"] / processed,
            "review_slice_rate": counters["review_slices"] / processed,
            "residue_slice_rate": counters["residue_slices"] / processed,
        }
        failures = []
        warnings = []
        if evidence["missing_clean_slices"]:
            failures.append("processed slices are missing final clean images")
        if evidence["invalid_mask_slices"]:
            failures.append("manual masks do not match final image dimensions")
        if evidence["outside_manual_changed_pixels"] and strict_manual_authority:
            failures.append("manual repaint changed pixels outside its persisted authority mask")
        elif evidence["outside_manual_changed_pixels"]:
            warnings.append(
                "auto_clean and final images differ outside manual masks; the bundle does not "
                "persist an automatic-baseline identity, so this is review evidence rather than proof of overreach"
            )
        return {
            "chapter_id": manifest.get("chapter_id"),
            "source": str(path),
            "source_sha256": _file_sha256(path),
            "semantics": {
                "automatic_completion_proxy": "1 - slices_with_human_repaint / processed_slices",
                "recall_claim": "none; source images do not carry exhaustive labelled text boxes",
            },
            "counts": counters,
            "manual_evidence": evidence,
            "metrics": metrics,
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "warnings": warnings,
        }
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-manual-authority", action="store_true",
        help="Fail on auto_clean/final differences outside manual masks; use only when baseline identity is known.",
    )
    args = parser.parse_args()
    report = audit_bundle(args.bundle.resolve(), strict_manual_authority=args.strict_manual_authority)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
