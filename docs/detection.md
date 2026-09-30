# Text detection

`app/detector/kiuyha_detector.py` finds where text is; `app/detector/ctd_mask.py` decides which pixels to erase.

1. [Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO)
   (YOLO26n, 2.4 M parameters, trained at 1280 on manga incl. English and
   Vietnamese fan translations) finds text boxes. A slice (about 800 × 2400
   px) is laid in the 1280 input as its top and bottom halves side by side,
   so text reaches the model at 0.79× instead of 0.53× for the whole slice.
   Boxes are padded 16 px and merged across the overlap.
2. Each padded box goes through the mask head of
   [comic-text-detector](https://github.com/dmMaze/comic-text-detector) at
   full and half size (the half pass catches very large lettering). The
   letters it reads are grown by a thin outline (12 % of the letter height,
   so a white stroke round the letters goes too) and then, pixel by pixel,
   into smooth pixels unlike the colours round the box: glow and shadow.
   Growth that runs off the box or keeps going is art touching the letters
   and is dropped. A run of dots such as "..." stays with the text it follows,
   and a box holding two separate captions becomes two texts.
3. After inpainting, the same model looks inside every erased block; a block
   where it still reads letters gets one more pass with a fresh mask.
   Kiuyha does not run again.

LaMa fills each crop in one pass at a 512 px long side: on labelled art the
fill error is the same as at full size and the model runs 2–3× faster. A hole
ringed by one colour, of any shade, is filled flat instead of by LaMa.

One image always gives one mask: the background colours are clustered with
k-means++, whose random draws are seeded, so a page cleaned twice gives the same
pixels (checked on two runner CPUs).

## The CTD model

`scripts/export_ctd_onnx.py` turns the released `comictextdetector.pt.onnx`
into `models/ctd_seg.onnx`:

- 43 % of its weights are denormal floats, which made every CPU convolution
  crawl (35 s per 1024² tile); they are zeroed (1.2 s, masks unchanged).
- Its stride-2 transposed convolutions are rewritten exactly as a 3×3
  convolution into four phases and a pixel shuffle.
- Only the mask output is kept, and the input size is free (multiples of 64).

The largest difference from the original is 1e-6; the masks agree on every
pixel.

## Evidence

Whole real chapters through the app on a 4-core GitHub runner, two page
workers, results on the `audit-evidence` branch:

| Chapter | Slices | Text blocks | Time | Text left |
| --- | --- | --- | --- | --- |
| The World After the End 254 (webtoon) | 68 | 167 | 418 s | none |
| Youngest Scion of the Mages 137 (webtoon) | 52 | 145 | 332 s | none |

The first chapter took 418 s and 446 s on two different runner CPUs, and the
sampled slices matched pixel for pixel. A.I mode on Shadow Slave ch.1 (135
slices) leaves no text blocking the export.

When the masks replaced the hand-tuned rules, both cleaners ran on the same
runner (flow bench):

| Chapter | Pages | Old rules | CTD masks | Story text left |
| --- | --- | --- | --- | --- |
| The Hero Cannot Rest 1 (webtoon) | 36 | 763 s | 562 s | none; the old rules missed a glowing title |
| Oneshot on MangaDex (manga, coloured characters) | 39 | 301 s | 382 s | none in either |

On labelled synthetic tiles, the CTD mask inside Kiuyha boxes erases 73.7 %
of what has to go at 73.1 % precision, against 65.4 % at 43.1 % for the old
rules, which painted over 2.3× the text. What is left is watermarks and site
banners. Stylised sound effects are mostly kept as art. On large manga pages with heavy black
art the new path can still take a stroke of art that Kiuyha boxed as text;
the restore brush puts it back.

Tried and dropped: training a small UNet from scratch on synthetic lettering
(too slow on CPU to beat a pretrained model), letting the mask spread through
white top-hat strokes (it followed art as readily as glow), a CRF refinement
(no gain), rescaling each block to a 64 px letter height (worse than the two
fixed scales), and the previous YOLOv8m text segmenter.
