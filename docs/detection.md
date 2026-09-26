# Text detection

Slices are about 800 × 2400 px (up to 4096). Both detectors see a whole
slice in **one** forward pass by laying its top and bottom halves side by
side in the model's square input, so text reaches the model 1.5–2× bigger
than when the slice is shrunk whole into the middle of the square (0.43× for
2400 px, 0.25× at 4096), where two thirds of the input was grey padding.

## Kiuyha + Otsu (used when `models/kiuyha_text_1280.onnx` is present)

`app/detector/kiuyha_detector.py`

1. [Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO)
   (YOLO26n, 2.4 M parameters, trained at 1280 on manga incl. English and
   Vietnamese fan translations) finds text boxes on the two halves in its
   1280 input (0.79× for an 800 px wide slice). Boxes are padded 16 px and
   merged across the overlap.
2. Each box becomes a letter mask: pixels far from the box's border colour
   (Otsu), minus anything touching the border (bubble outline, art), closed
   into word blobs and grown 6 px past the letter outline.

## Text segmenter (fallback)

`app/one_shot_cleanup.py` — `text_segmenter.onnx` (YOLOv8m-seg, 1024 input)
on the two halves; `MANGA_TEXT_DETECTOR=segmenter` forces it.

## Evidence

Same 16 slices of a real chapter, cleaned with LaMa after each detector
(output on the `audit-evidence` branch; full chapters: Chapter run workflow):

| Detector | Text blocks left of 24 | Detection | Over-erase |
| --- | --- | --- | --- |
| Segmenter, whole slice in one pass (old) | 5 | 1× | 28% |
| Segmenter, two halves | 4 | 1× | 8% |
| Kiuyha + Otsu, two halves | 3 | 0.22× | 19% |

The three blocks left by every detector are stylised sound effects ("HUFF",
large Korean SFX). Checked by eye as well: the counts cannot see letter
outlines left as ghosts, which the first Otsu version did and the current
one does not.

Tried and dropped: detecting in overlapping windows (2.7–3.5× the detector
time), feeding LaMa a slice's text regions top to bottom with pending text
hidden, adding back grain after LaMa (visible speckles), YOLO26 models
trained only on Manga109 (black-and-white Japanese pages; they missed most
webtoon text), and running the segmenter on Kiuyha's box crops (it misses
text without surrounding context).
