---
license: mit
github: https://github.com/sadowb/CuratorML
datasets:
- MS92/MangaSegmentation
- ShadowB/Manga109_RegionLevelTextSegmentation
language:
- ja
pipeline_tag: image-segmentation
library_name: ultralytics
tags:
- manga
- manga-segmentation
- image-segmentation
- instance-segmentation
- yolo
- yolo26
- yolo26s
- comics
- document-understanding
- layout-analysis
- ocr-preprocessing
- translation-pipeline
model-index:
- name: yolo26s-manga-panel-bubble-text-segmentation
  results:
  - task:
      type: image-segmentation
      name: Manga region instance segmentation
    dataset:
      type: custom
      name: Book-level validation split from Manga109-derived segmentation data
    metrics:
    - type: precision
      name: Box Precision
      value: 0.96521
    - type: recall
      name: Box Recall
      value: 0.95165
    - type: map
      name: Box mAP@0.5
      value: 0.97494
    - type: map
      name: Box mAP@0.5:0.95
      value: 0.89988
    - type: precision
      name: Mask Precision
      value: 0.96564
    - type: recall
      name: Mask Recall
      value: 0.95026
    - type: map
      name: Mask mAP@0.5
      value: 0.97013
    - type: map
      name: Mask mAP@0.5:0.95
      value: 0.84573
---

# YOLO26s Manga Panel, Text, and Balloon Segmentation

This is a YOLO26s segmentation model for manga page layout analysis. It detects and segments three region types needed by manga OCR and translation pipelines:

- panels / frames,
- text regions,
- speech or narration balloons.

The model is intended for manga document-understanding workflows where page regions must be located before OCR, reading-order reconstruction, translation, inpainting, or human review.

## Model Details

### Model Description

This model is an Ultralytics-compatible YOLO26s instance segmentation model trained on Manga109-derived segmentation data. It predicts bounding boxes, class IDs, confidence scores, and pixel masks for manga page regions.

- **Developed by:** ShadowB / Abdelhadi Marjane
- **Model type:** Image segmentation / instance segmentation
- **Architecture:** YOLO26s segmentation model (`yolo26s-seg.yaml`)
- **Base checkpoint:** `yolo26s-seg.pt`
- **Library:** Ultralytics
- **Task:** Manga region instance segmentation
- **Primary domain:** Manga/comic page images
- **Languages:** Japanese manga pages. The model detects page regions visually; it does not read or translate text.
- **License:** MIT for this model repository. Dataset licenses and access rules may differ.
- **Number of classes:** 3
- **Parameters:** 11,436,269
- **Stride:** 8, 16, 32
- **Checkpoint size:** about 23.4 MB for `best.pt`
- **Training date recorded in checkpoint:** 2026-04-29
- **Ultralytics version recorded in checkpoint:** 8.4.43
- **This model is part of github:sadowb/CuratorML** a translation workspace that uses it for region detection.

## Label Schema

The labels are stored in the model checkpoint and match the expected YOLO dataset `names` mapping:

| Class ID | Label | Description |
|---:|---|---|
| 0 | `frame` | Manga page panel/frame regions, including bordered or visually separated panels |
| 1 | `text` | Visible text regions, usually the regions passed to OCR or translation post-processing |
| 2 | `balloon` | Speech balloons, thought bubbles, narration bubbles, or similar text containers |

Notes:

- Background is not an explicit class.
- `text` is the visual text region, not the OCR transcription.
- `balloon` is the container region around dialogue or narration text.
- `frame` is the panel/layout region, not necessarily a semantic scene label.
- Keep this class order unchanged in `data.yaml`, inference code, and downstream post-processing.

Recommended dataset config:

```yaml
names:
  0: frame
  1: text
  2: balloon
```

## Uses

### Direct Use

Use this model to segment manga page regions from a page image. Direct outputs can be used to locate:

- panels/frames,
- text regions,
- speech or narration balloons.

### Downstream Use

This model is designed to be one component in a larger manga translation or document-understanding system. A typical downstream flow is:

1. segment panels, balloons, and text regions,
2. associate text regions with balloons and panels,
3. run OCR on text regions,
4. reconstruct reading order,
5. translate text with surrounding visual/layout context,
6. inpaint or clean original text,
7. render translated text back into the page.

Example structured output expected by a downstream pipeline:

```json
{
  "page": "example_page.jpg",
  "regions": [
    {
      "id": 1,
      "class_id": 0,
      "label": "frame",
      "confidence": 0.94,
      "bbox": [x1, y1, x2, y2],
      "mask": "..."
    },
    {
      "id": 2,
      "class_id": 2,
      "label": "balloon",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2],
      "mask": "..."
    },
    {
      "id": 3,
      "class_id": 1,
      "label": "text",
      "confidence": 0.89,
      "bbox": [x1, y1, x2, y2],
      "mask": "..."
    }
  ]
}
```

### Out-of-Scope Use

This model should not be treated as:

- an OCR model,
- a translation model,
- a reading-order model by itself,
- a general natural-image segmentation model,
- a legal/copyright analysis tool,
- a safety-critical segmentation system,
- a perfect layout parser for every comic style.

It only segments visible page regions. It does not understand text content, speaker identity, story context, or translation quality.

## How to Get Started with the Model

Install dependencies:

```bash
pip install ultralytics pillow opencv-python
```

Run inference:

```python
from ultralytics import YOLO

# Replace with your local path or the Hugging Face model ID after upload.
model = YOLO("best.pt")

results = model.predict(
    source="example_manga_page.jpg",
    imgsz=1280,
    conf=0.25,
    iou=0.7,
    retina_masks=True,
)

class_names = {
    0: "frame",
    1: "text",
    2: "balloon",
}

for result in results:
    if result.boxes is None:
        print("No regions detected.")
        continue

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        label = class_names.get(class_id, str(class_id))

        print({
            "index": i,
            "class_id": class_id,
            "label": label,
            "confidence": confidence,
            "bbox": bbox,
        })

    # Saves an annotated image with boxes/masks.
    result.save(filename="segmented_output.jpg")
```

For high-quality mask extraction in a manga translation pipeline, use `retina_masks=True` during inference so masks are returned at higher resolution.

## Training Details

### Training Data

This model uses a merged Manga109-derived segmentation dataset with three region classes: `frame`, `text`, and `balloon`.

| Dataset | Hugging Face ID | Use | Notes |
|---|---|---|---|
| MangaSegmentation | `MS92/MangaSegmentation` | Segmentation annotations for manga regions | Dataset card references “Advancing Manga Analysis: Comprehensive Segmentation Annotations for the Manga109 Dataset.” |
| Manga109 Region-Level Text Segmentation | `ShadowB/Manga109_RegionLevelTextSegmentation` | Region-level text masks | Used to support the `text` class and downstream OCR/translation needs. |

### Dataset Composition

The provided split audit records a book-level split across 109 manga groups:

| Split | Books / Groups | Images |
|---|---:|---:|
| Train | 83 | 7,174 |
| Validation | 12 | 1,468 |
| Test | 14 | 1,488 |
| Total | 109 | 10,130 |

The book-level split is important because random page-level splits can overestimate performance by leaking manga-specific style, art, and layout patterns between train and validation data.

### Preprocessing

The training data was normalized into a YOLO-compatible segmentation layout with the following class mapping:

```text
0: frame
1: text
2: balloon
```

Known preprocessing goals:

- merge Manga109-derived annotations into a common three-class schema,
- preserve separate panel/frame, text-region, and balloon masks,
- use a book-level split to better evaluate generalization across manga titles,
- train in an Ultralytics segmentation format compatible with `yolo segment train`.

### Training Procedure

The model was trained on Kaggle with Ultralytics YOLO26s segmentation. The training script builds a book-level train/validation/test split, maps the labels to three classes (`frame`, `text`, `balloon`), and keeps `overlap_mask=False` because manga regions can sit inside each other.

Training used `yolo26s-seg.pt` as the starting checkpoint, image size 1280, batch size 8 across two GPUs, and MuSGD. The run completed 41 epochs and took 11h 13m overall. The checkpoint stores an Ultralytics training time value of 11.0054 hours, which reflects the active training budget rather than the full notebook runtime.

#### Training Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Architecture | YOLO26s segmentation |
| Base checkpoint | `yolo26s-seg.pt` |
| Image size | 1280 |
| Batch size | 8 |
| Epochs completed | 41 |
| Overall run time | 11h 13m |
| Optimizer | MuSGD |
| Learning rate | 0.01 initial, 0.01 final factor |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Cosine LR | True |
| AMP | True |
| Device | `0,1` |
| Overlap mask | False |
| Main augmentations | mosaic 0.3, copy-paste 0.1, HSV-V 0.04, no flips/rotation |

#### Model Size

| Item | Value |
|---|---:|
| Checkpoint size, `best.pt` | 23,439,133 bytes |
| Parameters | 11,436,269 |

## Evaluation

### Testing Data, Factors & Metrics

The available metrics are from the validation run recorded in `best.pt`, `results.csv`, and the validation artifacts in this repository.

- **Validation split:** book-level validation split
- **Validation groups/books:** 12
- **Validation images:** 1,468
- **Test split available in audit:** 14 groups/books, 1,488 images
- **Metrics reported:** box precision, box recall, box mAP, mask precision, mask recall, mask mAP
- **Artifact source:** `/validationResultsOfMangaModel/results.csv` and checkpoint `train_metrics`

Recommended factors for further evaluation:

- unseen manga titles/books,
- dense vs sparse pages,
- bordered vs borderless panels,
- large vs small balloons,
- small or dense text,
- heavy screentone regions,
- low-resolution or compressed pages,
- overlapping text/balloon/frame regions.

### Results

The following values come from the `best.pt` checkpoint `train_metrics` field:

| Metric | Value |
|---|---:|
| Box Precision | 0.96521 |
| Box Recall | 0.95165 |
| Box mAP@0.5 | 0.97494 |
| Box mAP@0.5:0.95 | 0.89988 |
| Mask Precision | 0.96564 |
| Mask Recall | 0.95026 |
| Mask mAP@0.5 | 0.97013 |
| Mask mAP@0.5:0.95 | 0.84573 |
| Validation box loss | 0.43638 |
| Validation segmentation loss | 0.59429 |
| Validation classification loss | 0.26392 |
| Validation DFL loss | 0.00241 |
| Fitness | 1.74561 |

The final row in `results.csv`, epoch 41, records very similar overall metrics:

| Metric | Epoch 41 Value |
|---|---:|
| Box Precision | 0.96489 |
| Box Recall | 0.95021 |
| Box mAP@0.5 | 0.97432 |
| Box mAP@0.5:0.95 | 0.89907 |
| Mask Precision | 0.96627 |
| Mask Recall | 0.94811 |
| Mask mAP@0.5 | 0.96986 |
| Mask mAP@0.5:0.95 | 0.84459 |

### Per-Class Results

The local artifacts provided here include overall metrics, PR/F1/P/R curves, labels visualization, and confusion matrices. A per-class numeric mAP table was not present in `results.csv`.

To add per-class metrics, run a validation command that prints or exports per-class results, then update this table:

| Class | Box mAP@0.5 | Box mAP@0.5:0.95 | Mask mAP@0.5 | Mask mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| `frame` | TODO | TODO | TODO | TODO |
| `text` | TODO | TODO | TODO | TODO |
| `balloon` | TODO | TODO | TODO | TODO |

Suggested command when the dataset is available:

```bash
yolo segment val \
  model=best.pt \
  data=/path/to/data.yaml \
  imgsz=1280 \
  split=val \
  plots=True
```

### Evaluation Artifacts

If the files are uploaded with this model repository, the following artifacts document the run:

| Artifact | Purpose |
|---|---|
| `results.csv` | epoch-by-epoch training and validation metrics |
| `results.png` | metric curves over training |
| `labels.jpg` | label distribution visualization |
| `confusion_matrix.png` | confusion matrix |
| `confusion_matrix_normalized.png` | normalized confusion matrix |
| `BoxPR_curve.png` | box precision-recall curve |
| `MaskPR_curve.png` | mask precision-recall curve |
| `BoxF1_curve.png`, `MaskF1_curve.png` | F1 curves |


#### Curves and Visual Results

Training curves:

![Training and validation results](https://cdn-uploads.huggingface.co/production/uploads/631a6bcfc9f8cd19a73722d8/4TMLSLUEzjctZfgt3nbej.png)


Labels:

![label distribution](https://cdn-uploads.huggingface.co/production/uploads/631a6bcfc9f8cd19a73722d8/fc_dGUJD-ZlPLrxoCTdjl.jpeg)

Confusion matrix:

![confusion_matrix](https://cdn-uploads.huggingface.co/production/uploads/631a6bcfc9f8cd19a73722d8/Mx7l4oWRkG1394tLRBIN-.png)

Normalized confusion matrix:

![confusion_matrix_normalized](https://cdn-uploads.huggingface.co/production/uploads/631a6bcfc9f8cd19a73722d8/O_V0DfUzGLX9Zz6kZj0OR.png)


Mask PR curve:

![MaskPR_curve](https://cdn-uploads.huggingface.co/production/uploads/631a6bcfc9f8cd19a73722d8/vDe2NB438KgocPVt7Yp-m.png)


Box PR curve:

![BoxPR_curve](https://cdn-uploads.huggingface.co/production/uploads/631a6bcfc9f8cd19a73722d8/VKqIXZUpqGNc5Gvwe8Baq.png)


### Summary

The model achieves strong validation performance on the book-level validation split, with mask mAP@0.5 of 0.97013 and mask mAP@0.5:0.95 of 0.84573. The model is suitable as a practical manga layout segmentation component, especially for pipelines that need panel, text, and balloon masks before OCR or translation.

For production use, visual inspection is still recommended because manga segmentation quality depends heavily on small text, dense screentones, borderless panels, overlapping regions, and unusual page layouts.

## Bias, Risks, and Limitations

This model is specialized for Manga109-style manga pages. It may not generalize well to:

- Western comics,
- colored comics,
- vertical webtoons,
- very low-resolution scans,
- pages with unusual layouts,
- handwritten or highly stylized text,
- heavily compressed images,
- non-manga documents,
- very small text or very thin panel borders.

Known technical limitations:

- `text` masks can be sensitive to small font size, dense screentones, and low contrast.
- `balloon` masks may be imperfect for irregular balloons, overlapping balloons, or narration boxes.
- `frame` predictions can confuse panel borders with artwork lines on complex pages.
- Validation metrics may not fully capture mask-boundary quality needed for inpainting or redrawing.
- Even book-level splits may not cover every real-world manga style.

### Recommendations

Users should:

- visually inspect masks before using them in a production translation pipeline,
- evaluate on their own manga pages before deployment,
- prefer book/title-level splits for new evaluations,
- tune confidence and IoU thresholds for their use case,
- use `retina_masks=True` when precise masks are needed,
- combine this model with OCR, reading-order logic, and human review.

## Environmental Impact

Carbon emissions were not measured for this run.

- **Hardware Type:** multi-GPU Kaggle environment recorded as `device=0,1`; exact GPU type not recorded in the checkpoint
- **Hours used:** about 11.0 hours
- **Cloud Provider:** Kaggle
- **Compute Region:** Not recorded
- **Carbon Emitted:** Not measured

Carbon emissions can be estimated using the Machine Learning Impact calculator:
https://mlco2.github.io/impact

## Technical Specifications

### Model Architecture and Objective

This is a YOLO26s segmentation model with a YOLO-style detection backbone/head and segmentation mask output. The training objective optimizes object detection and instance segmentation losses to predict:

- bounding boxes,
- class probabilities,
- instance segmentation masks.

The checkpoint records:

```text
nc: 3
scale: s
yaml_file: yolo26s-seg.yaml
head: Segment26
stride: 8, 16, 32
```

### Compute Infrastructure

#### Hardware

- Training environment: Kaggle
- Device setting: `0,1`
- Exact GPU model: TPU

#### Software

- Python: TODO: add exact version if known
- PyTorch: TODO: add exact training version if known
- Ultralytics: 8.4.43 recorded in checkpoint

## Data and License Notes

The model repository is licensed under MIT. This does not override the licenses, access restrictions, attribution requirements, or redistribution rules of the datasets used to train/evaluate the model.

Users are responsible for checking and following the terms for:

- `MS92/MangaSegmentation`
- `ShadowB/Manga109_RegionLevelTextSegmentation`

Important notes:

- Dataset redistribution may be restricted.
- MangaSegmentation has its own license/citation requirements.
- The MIT license applies to this model card/model repository content, not necessarily to the original manga images or dataset annotations.

## Citation

If you use this model, cite the relevant datasets and papers.

### MangaSegmentation

```bibtex
@inproceedings{xie2025advancing,
  title={Advancing Manga Analysis: Comprehensive Segmentation Annotations for the Manga109 Dataset},
  author={Minshan Xie and Jian Lin and Hanyuan Liu and Chengze Li and Tien-Tsin Wong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

### Manga109

```bibtex
@article{aizawa2020building,
  title={Building a Manga Dataset "Manga109" with Annotations for Multimedia Applications},
  author={Aizawa, Kiyoharu and Fujimoto, Azuma and Otsubo, Atsushi and Ogawa, Toru and Matsui, Yusuke and Tsubota, Koki and Ikuta, Hikaru},
  journal={IEEE MultiMedia},
  year={2020}
}
```

### This Model

```bibtex
@misc{shadowb_yolo26s_manga_region_segmentation,
  title={YOLO26s Manga Panel, Text, and Balloon Segmentation},
  author={Abdelhadi marjane},
  year={2026},
  publisher={Hugging Face},
  howpublished={ShadowB/Manga109-panel-Balloon-text-yoloV26-segmentation}}
    }
```
## Glossary

- **Frame / Panel:** A visual manga page region containing a scene or layout unit.
- **Text region:** The visible text area, usually passed to OCR.
- **Balloon:** A speech bubble, thought bubble, narration bubble, or similar text container.
- **Instance segmentation:** A task that detects individual objects and predicts a separate mask for each object instance.
- **mAP:** Mean Average Precision, a standard detection/segmentation metric.
- **Book-level split:** A split where entire manga titles/books are held out, reducing leakage between train and validation data.

## More Information

- Hugging Face model card documentation: https://huggingface.co/docs/hub/model-cards
- Hugging Face annotated model card guide: https://huggingface.co/docs/hub/model-card-annotated
- Manga109 project: https://manga109.github.io/manga109-project-website

## Model Card Authors

Abdelhadi Marjane

## Model Card Contact
Abdelhadi Marjane
CuratorML is the translation workspace this model was trained for. Issues and PRs are open.