## Part 1: bugs

### 1. `app/downloader/registry.py` â `download_chapter` deletes every page when the width filter removes all pages (certain)

Quoted code:

```python
pages = [path for path in paths if _image_width(path) >= DOWNLOAD_JS_MIN_IMAGE_WIDTH]
for path in set(paths) - set(pages):
    path.unlink(missing_ok=True)
return pages or paths
```

Trigger: a chapter whose every downloaded image is narrower than `DOWNLOAD_JS_MIN_IMAGE_WIDTH` (300 px), e.g. thumbnails or low-resolution pages.

What goes wrong: all files are unlinked, then `pages or paths` returns the now-deleted `paths`, so callers receive paths to missing files. This is data loss and a downstream crash when the files are opened.

Smallest fix: only unlink when there is at least one retained page; otherwise return the original paths unchanged:

```python
if pages:
    for path in set(paths) - set(pages):
        path.unlink(missing_ok=True)
    return pages
return paths
```

### 2. `app/pipeline.py` â `_shared_seam_detections` can report seam context as unavailable even when detection succeeds (suspected)

Quoted code:

```python
if source_page < 0 or source_y2 <= source_y1 or core_y2 <= core_y1:
    if stitch_core.get("unsafe_before") or stitch_core.get("unsafe_after"):
        unavailable_pages.add(idx)
    continue
```

Trigger: a slice whose `stitch_core` lacks one of the expected integer fields but still has `unsafe_before`/`unsafe_after` set.

What goes wrong: the page is marked `seam_context_unavailable` even though the seam detector may later run successfully for another consumer of the same cut. The impact depends on whether the missing field is genuinely fatal or merely optional metadata.

Smallest fix: distinguish âno seam consumer possibleâ from âmetadata incompleteâ; only add to `unavailable_pages` when the corresponding seam consumer cannot be constructed.

### 3. `app/detector/kiuyha_detector.py` â `page_boxes` drops boxes near an inner window edge even when they are not actually cut (suspected)

Quoted code:

```python
if (by1 <= EDGE_TOUCH / scale and y0 > 0) or (by2 >= part_h - EDGE_TOUCH / scale and y0 + part_h < h):
    continue
```

Trigger: a text box that lies close to, but does not cross, an internal page-scale window boundary.

What goes wrong: the box is skipped on the assumption that the next window will see it whole. If the box is merely near the edge, it can be missed entirely.

Smallest fix: skip only when the box actually extends beyond the window edge, not merely when it is within `EDGE_TOUCH` of it.

---

## Part 2: tests

```python
# tests/test_beam_geometry.py
import numpy as np

from app.detector.boxes import BubbleBox
from app.inpaint.mask_geometry import remap_local_mask_page_space
from app.page_processing import _contained, _fold_nested
from app.pipeline import ChapterPipeline


def test_remap_local_mask_page_space_copies_intersection():
    source_box = {"x1": 0, "y1": 0, "x2": 4, "y2": 4}
    target_box = {"x1": 2, "y1": 2, "x2": 6, "y2": 6}
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 255

    result = remap_local_mask_page_space(mask, source_box, target_box)

    assert result is not None
    assert result.shape == (4, 4)
    assert result[0, 0] == 255


def test_map_seam_detection_to_page_clips_mask_and_coordinates():
    mask = np.arange(1, 13, dtype=np.uint8).reshape(4, 3)
    box = BubbleBox(1, 0, 4, 4, 1.0, mask.copy())

    mapped = ChapterPipeline._map_seam_detection_to_page(
        box,
        seam_source_y1=10,
        page_source_y1=12,
        page_height=3,
    )

    assert mapped is not None
    assert (mapped.y1, mapped.y2) == (0, 2)
    assert mapped.mask is not None
    assert mapped.mask.shape == (2, 3)


def test_contained_uses_half_area_threshold():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    large = BubbleBox(0, 0, 100, 100, 0.9, mask)
    small = BubbleBox(10, 10, 20, 20, 0.8, mask[:10, :10].copy())

    assert _contained(small, large) is True


def test_fold_nested_merges_a_small_box_into_its_containing_box():
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    large = BubbleBox(0, 0, 100, 100, 0.9, mask)
    small = BubbleBox(10, 10, 20, 20, 0.8, mask.copy())

    folded = _fold_nested([large, small])

    assert len(folded) == 1
```

```python
# tests/test_beam_clustering.py
from app.detector.boxes import BubbleBox
from app.inpaint.clustering import cluster_boxes, split_oversized_cluster_area


def _mask(h=10, w=10):
    return np.ones((h, w), dtype=np.uint8) * 255


def test_cluster_boxes_merges_touching_boxes():
    import numpy as np

    boxes = [
        BubbleBox(0, 0, 10, 10, 0.9, _mask()),
        BubbleBox(12, 0, 22, 10, 0.8, _mask()),
    ]

    clusters = cluster_boxes(boxes)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_split_oversized_cluster_area_splits_by_bounding_box_area():
    import numpy as np

    boxes = [
        BubbleBox(0, 0, 10, 10, 0.9, _mask()),
        BubbleBox(100, 100, 110, 110, 0.8, _mask()),
    ]

    result = split_oversized_cluster_area(boxes, img_w=120, img_h=120)

    assert len(result) == 2
    assert {min(b.x1 for b in group) for group in result} == {0, 100}
```

```python
# tests/test_beam_jobs.py
import asyncio

from app.page_processing_jobs import ChapterProcessingJobManager


def test_job_manager_reuses_an_active_job():
    manager = ChapterProcessingJobManager(lambda *args, **kwargs: None)

    async def run():
        first = manager.start("chapter-a", page_indices=[0, 1], workers=1)
        second = manager.start("chapter-a", page_indices=[2], workers=1)
        return first, second

    first, second = asyncio.run(run())

    assert first["job_id"] == second["job_id"]
    assert second["reused"] is True


def test_job_manager_marks_completed_after_plan_finishes():
    def plan(chapter_id, page_indices, workers, on_page_done, **kwargs):
        for page_index in page_indices:
            on_page_done(page_index)

    manager = ChapterProcessingJobManager(plan)

    async def run():
        started = manager.start("chapter-b", page_indices=[0, 1], workers=1)
        job_id = started["job_id"]
        while manager.snapshot(job_id)["status"] in {"pending", "running"}:
            await asyncio.sleep(0.01)
        return manager.snapshot(job_id)

    finished = asyncio.run(run())

    assert finished["status"] == "completed"
    assert finished["completed"] == 2
    assert finished["done_indices"] == [0, 1]
```

```python
# tests/test_beam_registry.py
from pathlib import Path

from PIL import Image

import app.downloader.registry as registry


def _write_small_images(output_dir: Path, count: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = output_dir / f"{index:03d}.png"
        Image.new("RGB", (10, 10), "white").save(path)
        paths.append(path)
    return paths


def test_download_chapter_empty_filtered_pages_returns_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.security.validate_url", lambda url: url)
    monkeypatch.setattr(registry, "is_asura_chapter_page", lambda url: False)

    class StaticAdapter:
        def extract_image_urls(self, chapter_url):
            return ["http://example.com/a.png", "http://example.com/b.png"]

        def download_urls(self, urls, output_dir, referer):
            return _write_small_images(Path(output_dir), len(urls))

    class JsAdapter:
        def extract_image_urls(self, chapter_url):
            return []

    monkeypatch.setattr(registry, "STATIC_ADAPTER", StaticAdapter())
    monkeypatch.setattr(registry, "JS_ADAPTER", JsAdapter())

    result = registry.download_chapter("http://example.com/chapter", tmp_path / "out")

    # fails today: registry returns paths to unlinked images when every page is filtered out
    assert all(Path(path).exists() for path in result)
```