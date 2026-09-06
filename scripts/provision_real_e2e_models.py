from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORT_DIR = ROOT / "benchmark-results"
PROVENANCE_PATH = MODELS / ".e2e-provenance.json"

EXACT_ONNX_SHA256 = {
    "bubble_yolo.onnx": "9298fba141d8f7293007cbef72e2f20fdd12273a6e06b4967375fdb296e6c4db",
    "text_segmenter.onnx": "8e6c5d1a8d8ffd62bf563b91ac8562246abb31ac0f4786e6d8daf158214f6d9e",
    "lama-manga-dynamic.onnx": "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9",
}

LAMA_URL = "https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx"
LAMA_SHA256 = EXACT_ONNX_SHA256["lama-manga-dynamic.onnx"]
TEXT_PT_URL = "https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m/resolve/main/comic-text-segmenter.pt"
TEXT_PT_SHA256 = "f2dded0d2f5aaa25eed49f1c34a4720f1c1cd40da8bc3138fde1abb202de625e"
BUBBLE_PT_REVISION = "a081a21a12d3e1bbf31536ef00e7f5bf0a9b72a3"
BUBBLE_PT_URL = (
    "https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m/"
    f"resolve/{BUBBLE_PT_REVISION}/comic-speech-bubble-detector.pt"
)
BUBBLE_PT_SHA256 = "10bc9f702698148e079fb4462a6b910fcd69753e04838b54087ef91d5633097b"
TYPER_COMMIT = "35dd0c3046a5a6f9d2ea7c6c6a5498da05f4529b"
TYPER_ZIP_URL = (
    "https://raw.githubusercontent.com/darkmax159159357/TypeR/"
    f"{TYPER_COMMIT}/releases/TypeR-v2.9.9.zip"
)

_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_SHA_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, *, max_bytes: int = 1_500_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "manga-translator-real-e2e/1.0",
            "Accept": "*/*",
        },
    )
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as output:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise RuntimeError(f"download too large: {length} bytes from {url}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError(f"download exceeded {max_bytes} bytes: {url}")
                output.write(chunk)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def ensure_sha(url: str, path: Path, expected: str) -> None:
    if path.is_file() and sha256(path) == expected:
        return
    download(url, path)
    actual = sha256(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def safe_extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe zip member: {member.filename!r}")
        zf.extractall(target)


def exact_bundle(bundle_url: str) -> dict | None:
    if not bundle_url.strip():
        return None
    with tempfile.TemporaryDirectory(prefix="e2e-model-bundle-") as tmp_dir:
        tmp = Path(tmp_dir)
        archive = tmp / "models.zip"
        unpacked = tmp / "unpacked"
        download(bundle_url.strip(), archive)
        safe_extract_zip(archive, unpacked)
        for name, expected in EXACT_ONNX_SHA256.items():
            matches = list(unpacked.rglob(name))
            if len(matches) != 1:
                raise RuntimeError(
                    f"exact bundle must contain exactly one {name}; found {len(matches)}"
                )
            actual = sha256(matches[0])
            if actual != expected:
                raise RuntimeError(f"exact bundle {name} SHA mismatch: {actual}")
            shutil.copy2(matches[0], MODELS / name)
    return {
        "mode": "exact_bundle",
        "bundle_url_configured": True,
        "onnx_sha256": {
            name: sha256(MODELS / name) for name in EXACT_ONNX_SHA256
        },
    }


def _text_files(root: Path):
    suffixes = {
        ".js", ".jsx", ".ts", ".tsx", ".py", ".json", ".html",
        ".txt", ".md", ".sh", ".cmd",
    }
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
            and path.stat().st_size <= 8 * 1024 * 1024
        ):
            yield path


def discover_typer_detector(
    typer_root: Path,
) -> tuple[str, str | None, list[str]]:
    """Audit what the pinned TypeR bundle says, without trusting URL proximity.

    TypeR places multiple model URLs close together in minified/generated files.
    The previous bootstrap incorrectly paired the bubble SHA with a neighboring
    .pt URL. The actual provisioning source is now the immutable Hugging Face
    revision above; this discovery remains evidence that can be inspected in the
    provenance report when TypeR changes its packaging.
    """
    candidates: list[tuple[int, str, str | None, str]] = []
    debug: list[str] = []
    for path in _text_files(typer_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            lower = line.lower()
            if not any(
                token in lower
                for token in ("public.pt", "detection", "detector", "yolo")
            ):
                continue
            start = max(0, index - 4)
            end = min(len(lines), index + 5)
            block = "\n".join(lines[start:end])
            urls = [u.rstrip("),;]") for u in _URL_RE.findall(block)]
            shas = _SHA_RE.findall(block)
            expected = shas[0].lower() if shas else None
            for url in urls:
                score = 0
                url_lower = url.lower()
                if "public.pt" in url_lower:
                    score += 100
                if "comic-speech-bubble-detector" in url_lower:
                    score += 100
                if url_lower.endswith(".pt") or ".pt?" in url_lower:
                    score += 60
                if any(
                    token in url_lower for token in ("yolo", "detection", "detector")
                ):
                    score += 20
                if any(
                    token in lower
                    for token in ("public.pt", "detection_model", "detector_model")
                ):
                    score += 20
                if score:
                    candidates.append(
                        (score, url, expected, f"{path}:{index + 1}")
                    )
        for url in _URL_RE.findall(text):
            url = url.rstrip("),;]")
            if url.lower().endswith(".pt") or ".pt?" in url.lower():
                score = 140 if "comic-speech-bubble-detector" in url.lower() else 40
                candidates.append((score, url, None, str(path)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    seen: set[str] = set()
    deduped: list[tuple[int, str, str | None, str]] = []
    for row in candidates:
        if row[1] in seen:
            continue
        seen.add(row[1])
        deduped.append(row)
        debug.append(
            f"{row[0]}\t{row[1]}\t{row[2] or ''}\t{row[3]}"
        )
    if not deduped:
        raise RuntimeError("TypeR bundle did not expose a detector model URL")
    top = deduped[0]
    return top[1], top[2], debug[:20]


def export_yolo(
    pt_path: Path,
    output_path: Path,
    *,
    expected_names: set[str],
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    names_raw = (
        getattr(model.model, "names", None)
        or getattr(model, "names", None)
        or {}
    )
    if isinstance(names_raw, dict):
        names = {str(value) for value in names_raw.values()}
    else:
        names = {str(value) for value in names_raw}
    if names != expected_names:
        raise RuntimeError(
            f"{pt_path.name} classes mismatch: expected {sorted(expected_names)}, "
            f"got {sorted(names)}"
        )
    exported = model.export(
        format="onnx",
        imgsz=1024,
        opset=12,
        simplify=True,
        dynamic=False,
        nms=False,
        device="cpu",
        verbose=False,
    )
    exported_path = Path(str(exported))
    if not exported_path.is_file():
        fallback = pt_path.with_suffix(".onnx")
        if fallback.is_file():
            exported_path = fallback
        else:
            raise RuntimeError(f"Ultralytics did not produce ONNX for {pt_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        shutil.move(str(exported_path), output_path)
    return {
        "pt_sha256": sha256(pt_path),
        "onnx_sha256": sha256(output_path),
        "classes": sorted(names),
    }


def validate_contracts() -> dict:
    import onnxruntime as ort

    report: dict[str, dict] = {}
    checks = {
        "bubble_yolo.onnx": {"outputs": 1, "features": 6},
        "text_segmenter.onnx": {"outputs": 2, "features": 37},
        "lama-manga-dynamic.onnx": {"outputs": 1, "dynamic": True},
    }
    for name, spec in checks.items():
        path = MODELS / name
        if not path.is_file():
            raise RuntimeError(f"missing model after provisioning: {path}")
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        row = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "inputs": [item.shape for item in inputs],
            "outputs": [item.shape for item in outputs],
        }
        if len(outputs) != spec["outputs"]:
            raise RuntimeError(
                f"{name} output count mismatch: {len(outputs)} != {spec['outputs']}"
            )
        if name in {"bubble_yolo.onnx", "text_segmenter.onnx"}:
            shape = list(inputs[0].shape)
            if shape != [1, 3, 1024, 1024]:
                raise RuntimeError(f"{name} input contract mismatch: {shape}")
            first = list(outputs[0].shape)
            if len(first) != 3 or first[1] != spec["features"]:
                raise RuntimeError(f"{name} output contract mismatch: {first}")
        else:
            image_shape = list(inputs[0].shape)
            dynamic = any(
                dim is None or isinstance(dim, str) for dim in image_shape[2:4]
            )
            if not dynamic:
                raise RuntimeError(f"{name} is not dynamic: {image_shape}")
        report[name] = row
    return report


def public_equivalent() -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e2e-public-models-") as tmp_dir:
        tmp = Path(tmp_dir)
        ensure_sha(
            LAMA_URL,
            MODELS / "lama-manga-dynamic.onnx",
            LAMA_SHA256,
        )

        text_pt = tmp / "comic-text-segmenter.pt"
        ensure_sha(TEXT_PT_URL, text_pt, TEXT_PT_SHA256)
        text_info = export_yolo(
            text_pt,
            MODELS / "text_segmenter.onnx",
            expected_names={"text_comic"},
        )

        # Keep TypeR only as immutable provenance evidence. Its generated bundle
        # contains neighboring model URLs, so it must not choose the bytes that
        # enter the E2E detector. The canonical detector is pinned directly to
        # the upstream Hugging Face revision and verified before deserialization.
        typer_zip = tmp / "TypeR-v2.9.9.zip"
        typer_root = tmp / "TypeR"
        download(TYPER_ZIP_URL, typer_zip, max_bytes=50 * 1024 * 1024)
        safe_extract_zip(typer_zip, typer_root)
        typer_url, typer_sha, candidates = discover_typer_detector(typer_root)

        bubble_pt = tmp / "comic-speech-bubble-detector.pt"
        ensure_sha(BUBBLE_PT_URL, bubble_pt, BUBBLE_PT_SHA256)
        bubble_info = export_yolo(
            bubble_pt,
            MODELS / "bubble_yolo.onnx",
            expected_names={"text_bubble", "text_free"},
        )

    return {
        "mode": "public_equivalent",
        "bundle_url_configured": False,
        "lama": {
            "url": LAMA_URL,
            "sha256": sha256(MODELS / "lama-manga-dynamic.onnx"),
        },
        "text_segmenter": {"url": TEXT_PT_URL, **text_info},
        "bubble_detector": {
            "source_url": BUBBLE_PT_URL,
            "source_revision": BUBBLE_PT_REVISION,
            "expected_pt_sha256": BUBBLE_PT_SHA256,
            "typer_reference": {
                "bundle": TYPER_ZIP_URL,
                "discovered_url": typer_url,
                "discovered_sha256": typer_sha,
                "discovery_candidates": candidates,
            },
            **bubble_info,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision real E2E detector/inpaint models on GitHub Actions."
    )
    parser.add_argument(
        "--bundle-url", default=os.getenv("E2E_MODEL_BUNDLE_URL", "")
    )
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = exact_bundle(args.bundle_url)
    if report is None:
        report = public_equivalent()
    report["contracts"] = validate_contracts()
    report["production_exact"] = all(
        (MODELS / name).is_file()
        and sha256(MODELS / name) == expected
        for name, expected in EXACT_ONNX_SHA256.items()
    )
    PROVENANCE_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "e2e-model-provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
