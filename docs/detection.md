# Text detection

`app/detector/kiuyha_detector.py`

1. [Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO)
   (YOLO26n, 2.4 M parameters, trained at 1280 on manga incl. English and
   Vietnamese fan translations) finds text boxes. A slice (about 800 × 2400
   px) is laid in the 1280 input as its top and bottom halves side by side,
   so text reaches the model at 0.79× instead of 0.53× for the whole slice.
   Boxes are padded 16 px and merged across the overlap.
2. Each box becomes a letter mask: pixels far from the box's border colour
   (Otsu), minus anything touching the border (bubble outline, art), closed
   into word blobs and grown past the letter outline. Letters cut by the
   slice edge may touch the border on that side.
3. After inpainting, Kiuyha looks again; text it still sees inside a
   first-pass box (gradient or two-colour lettering the colour split missed)
   is erased as the whole box, and that box is saved so re-inpainting keeps it.

## Evidence

Chapter run workflow, real chapters, results on the `audit-evidence` branch:

| Chapter | Slices | Clean time | Text blocks left |
| --- | --- | --- | --- |
| The Academy's Weapon Replicator 1 | 87 | 116 s | 13 of 234 |
| War of Extinction 1 | 139 | 141 s | 10 of 281, 5 of them art the counter misread |

What is left is watermarks, Korean logos and credits, and motion-blurred title
lettering.

Tried and dropped: the previous YOLOv8m text segmenter (1.7× slower, more
text left), detecting in overlapping windows (2.7–3.5× the detector time),
adding back grain after LaMa (visible speckles), and YOLO26 models trained
only on Manga109 (they missed most webtoon text).
