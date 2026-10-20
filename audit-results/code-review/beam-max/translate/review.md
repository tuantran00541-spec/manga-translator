## Part 1: bugs

### 1. `app/ai_mode/seams.py` â `sync_seam_mirrors` revives overlap-dropped objects
**Certain.**  
Quoted lines:
```python
for obj in page.get("text_objects") or []:
    if not isinstance(obj, dict) or obj.get("joined_into"):
        continue
    found = _owner(pages, page_index, obj)
    if found is None or not str(found[1].get("translation") or "").strip():
        continue
```
Trigger: a page has two seam-mirror objects that mostly overlap; `drop_overlapping_letters` clears the smaller one and sets `overlap_dropped=True`, then `_sync_seams` calls `sync_seam_mirrors`, which copies the ownerâs translation back into the dropped object.  
What goes wrong: the overlap drop is undone, so both overlapping translations render and garble each other.  
Smallest fix: skip dropped objects:
```python
if not isinstance(obj, dict) or obj.get("joined_into") or obj.get("overlap_dropped"):
    continue
```

### 2. `app/ai_mode/glossary.py` â `read_glossary` labels image batches by batch position
**Certain.**  
Quoted line:
```python
images = [(f"SLICE {index + 1}", image) for index, image in slices]
```
Trigger: a glossary image batch starts at source slice 5.  
What goes wrong: slices are labelled `SLICE 1`, `SLICE 2`, â¦ instead of their chapter positions, so the modelâs reading-order and slice references are wrong.  
Smallest fix:
```python
images = [(f"SLICE {page_index + 1}", image) for page_index, image in slices]
```

### 3. `app/ai_mode/checkpoints.py` â `pack` sizes the sheet to image width, not tag width
**Certain.**  
Quoted line:
```python
width = max(left + c.image.shape[1] for left, _, c in rows)
```
Trigger: a crop image is narrower than `70` px.  
What goes wrong: the tag rectangle is drawn with width `max(cw, 70)`, but the sheet is only `image width` wide, so the tag is clipped/truncated and the sheet is too narrow.  
Smallest fix:
```python
width = max(left + max(c.image.shape[1], 70) for left, _, c in rows)
```

### 4. `app/ai_mode/job.py` â `scan` does not guard the decider task
**Suspected.**  
Quoted lines:
```python
judge = asyncio.ensure_future(asyncio.to_thread(decider.storyless_slices, self.provider, self.api_key, active, load))
...
storyless = await judge if judge is not None else None
```
Trigger: `decider.storyless_slices` raises, e.g. an image read fails inside `load`.  
What goes wrong: the exception propagates out of `scan`, failing the whole job instead of degrading to âno decider answerâ.  
Smallest fix: wrap `await judge` in `try/except`, set `storyless = None`, and record the error.

### 5. `app/ai_mode/job.py` â `_keep_art` and `_render_one` do not catch `HTTPException` from pipeline calls
**Suspected.**  
Quoted lines:
```python
except (ValueError, RuntimeError, OSError) as exc:
```
and
```python
except (ValueError, RuntimeError, OSError) as exc:
```
Trigger: `pipeline.preserve_and_reinpaint` raises an `HTTPException`.  
What goes wrong: the exception escapes the stage and fails the chapter instead of being recorded as a region-level failure.  
Smallest fix: add `HTTPException` to both caught exception tuples.

### 6. `app/render/identity.py` â render signature omits `letter_bounds`
**Suspected.**  
Quoted lines:
```python
"translation": str(obj.get("translation") or ""),
"style": obj.get("style") or {},
"source_missing": bool(obj.get("source_missing", False)),
```
Trigger: `letter_bounds` changes while translation, style, region and source state stay the same.  
What goes wrong: `render_artifact_is_current` can report a render as current even though the lettering position changed.  
Smallest fix: include `letter_bounds` in `_normalized_text_object`.

---

## Part 2: tests

```python
# tests/test_beam_checkpoints.py
import numpy as np

from app.ai_mode.checkpoints import Crop, pack


def test_pack_sheet_is_wide_enough_for_its_tag():
    # fails today: checkpoints.pack ignores tag width when sizing the sheet
    crop = Crop("1.1", 0, (0, 0, 5, 5), np.zeros((5, 5, 3), dtype=np.uint8))
    sheet, placed = pack([crop])[0]
    assert len(placed) == 1
    assert sheet.shape[1] >= 70
```

```python
# tests/test_beam_glossary.py
from types import SimpleNamespace

import numpy as np

from app.ai_mode import glossary


def test_read_glossary_labels_slices_by_page_index(monkeypatch):
    # fails today: read_glossary labels image batches by batch position
    captured = {}

    def fake(provider, model, api_key, prompt, images, **kwargs):
        captured["images"] = images
        return type("Result", (), {"data": {}, "estimated_cost_usd": 0.0})()

    monkeypatch.setattr(glossary, "request_vision_json", fake)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    glossary.read_glossary(
        SimpleNamespace(), "model", "key", "Vietnamese", "vi", [(4, image), (5, image)]
    )

    assert [label for label, _ in captured["images"]] == ["SLICE 5", "SLICE 6"]
```

```python
# tests/test_beam_seams.py
import numpy as np

from app.ai_mode.seams import _shared, join_stacked_lines, sync_seam_mirrors


def test_shared_overlap_is_share_of_smaller_box():
    assert _shared((0, 0, 10, 10), (5, 5, 15, 15)) == 0.25


def test_join_stacked_lines_merges_a_vertical_block():
    manifest = {
        "pages": [
            {
                "text_objects": [
                    {"id": "a", "region": {"x1": 10, "y1": 10, "x2": 50, "y2": 30}},
                    {"id": "b", "region": {"x1": 12, "y1": 32, "x2": 48, "y2": 50}},
                ]
            }
        ]
    }

    assert join_stacked_lines(manifest, [0]) == 1
    objects = manifest["pages"][0]["text_objects"]
    assert objects[0]["joined_lines"] == ["b"]
    assert objects[1]["joined_into"] == "a"


def test_sync_seam_mirrors_does_not_revive_dropped_text():
    # fails today: sync_seam_mirrors copies an owner translation over an overlap-dropped mirror
    manifest = {
        "pages": [
            {
                "skipped": False,
                "source_page": 0,
                "stitch_core": {
                    "source_y1": 0,
                    "core_source_y1": 0,
                    "core_source_y2": 50,
                },
                "text_objects": [
                    {
                        "id": "mirror",
                        "region": {"x1": 0, "y1": 40, "x2": 10, "y2": 60},
                        "translation": "",
                        "overlap_dropped": True,
                    }
                ],
            },
            {
                "skipped": False,
                "source_page": 0,
                "stitch_core": {
                    "source_y1": 50,
                    "core_source_y1": 50,
                    "core_source_y2": 100,
                },
                "text_objects": [
                    {
                        "id": "owner",
                        "region": {"x1": 0, "y1": -10, "x2": 10, "y2": 10},
                        "translation": "owner text",
                    }
                ],
            },
        ]
    }

    changed = sync_seam_mirrors(manifest)
    assert manifest["pages"][0]["text_objects"][0]["translation"] == ""
    assert manifest["pages"][0]["text_objects"][0].get("overlap_dropped") is True
```

```python
# tests/test_beam_parsing.py
from app.ai_mode.page_scan import parse_scan
from app.translation.vision import parse_vision_translation


def test_parse_scan_rejects_a_logo_covering_most_of_the_slice():
    data = {
        "slices": [
            {
                "slice": 0,
                "is_credit": False,
                "credit_confidence": 0.0,
                "logos": [{"x1": 0, "y1": 0, "x2": 100, "y2": 100, "confidence": 0.9}],
            }
        ]
    }
    scans = parse_scan(data, {0: (100, 100)})
    assert scans[0].logos == ()


def test_parse_vision_translation_fills_omitted_ids_when_allowed():
    content = '{"translations":[{"id":"1","translated_text":"xin chao"}]}'
    result = parse_vision_translation(content, {"1", "2"}, allow_missing=True)
    assert result == {"1": "xin chao", "2": ""}
```

```python
# tests/test_beam_polish.py
from app.ai_mode.polish import Line, better, quality, settled


def test_better_rejects_an_equal_rewrite():
    scores = {"natural": 2.0, "clear": 2.0, "faithful": 2.0}
    assert not better(scores, scores)


def test_quality_stays_on_one_scale():
    scores = {"natural": 3.0, "clear": 2.0, "faithful": 2.0}
    assert 0.0 <= quality(scores) <= 1.0


def test_settled_english_line_needs_a_first_person_pronoun():
    line = Line(page_index=0, id="1", source="I cannot stop", text="...", speaker="A")
    memory = {"address": [{"from": "A", "to": "B", "self": "ta", "other": "ngÆ°Æ¡i"}], "glossary": {}}
    address, _terms = settled(line, memory)
    assert address == ['self "ta"']
```

```python
# tests/test_beam_glossary_merge.py
from app.ai_mode.glossary import merge_glossaries


def test_merge_glossaries_drops_a_rare_close_spelling():
    parts = [
        {"names": [{"source": "Hero", "target": "Anh HÃ¹ng"}]},
        {"names": [{"source": "Hero", "target": "Anh HÃ¹ng"}]},
        {"names": [{"source": "Hreo", "target": "Anh HÃ¹ng"}]},
    ]
    merged = merge_glossaries(parts)
    assert [entry["source"] for entry in merged["names"]] == ["Hero"]
```