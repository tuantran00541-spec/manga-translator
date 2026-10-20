This folder holds the first version of a manga and manhwa translator (`app/`, see `README.md`). Its weakest part is
cleaning: finding the lettering on a comic slice and erasing it so the art underneath looks untouched. Today it leaves
text behind, leaves faint ghosts of letters, smears or blots the balloons and art, and misses text outside balloons
(captions, narration boxes, sound effects, signs). Your job is to make cleaning as good as you can.

- `python3 clean.py IN_DIR OUT_DIR` cleans every `*.png` in `IN_DIR` and writes the result to `OUT_DIR` under the same
  name and size. It runs the first version's pipeline now; change it and anything in `app/` as you see fit.
- `models/` holds the three ONNX models the first version uses: `bubble_yolo.onnx` (speech balloons),
  `text_segmenter.onnx` (text; note the first version may not read its output the way the model was built) and
  `lama.onnx` (LaMa inpainting). You may use them in any way, or not at all.
- `dev/` holds the slices of one real chapter to work on. No answers are given: judge results yourself, by measuring
  and, if your tools can show you images, by looking.
- Python 3.12 with numpy, opencv-python-headless, onnxruntime, Pillow, scipy and scikit-image is installed. This
  machine has no internet access, so nothing else can be installed or downloaded.

A grader will later run `python3 clean.py` on slices of other chapters you have not seen, with a 40 minute limit for
about 80 slices on this machine's 4 CPU cores. It counts text and faint letter traces left where the original had
lettering, and art changed away from any lettering. Erasing art, balloons or panels to hide text costs as much as
leaving text. Work until the cleaning is as good as you can make it, and leave `clean.py` working.
