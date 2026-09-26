---
license: agpl-3.0
base_model: Ultralytics/YOLO26
tags:
  - manga
  - panel-detection
  - text-detection
  - yolo
  - yolo26
  - ultralytics
  - tflite
  - android
  - on-device
  - mobile
  - comic
  - manga109
datasets:
  - hal-utokyo/Manga109-s
pipeline_tag: object-detection
---

# Manga panel & text detector (YOLO26-nano)

A lightweight YOLO26-nano model fine-tuned on [Manga109-s](https://huggingface.co/datasets/hal-utokyo/Manga109-s) for detecting **panels** and **text bubbles** in manga pages. Designed for on-device Android inference via TFLite/LiteRT.

> **License change (September 2026):** this model was previously listed as Apache 2.0. That was a mistake on my part. It is fine-tuned from Ultralytics' YOLO26n pretrained weights, which are distributed under AGPL-3.0, so the fine-tuned weights are AGPL-3.0 as well. I was never able to grant Apache 2.0 for them. If you use this model in your project, please check the [License](#license) section below. Thanks to the community member who pointed this out.

## Performance

### INT8 TFLite (2.71 MB)

| Metric | All | Panel | Text |
|--------|-----|-------|------|
| mAP50 | 0.956 | 0.985 | 0.928 |
| mAP50-95 | 0.846 | 0.953 | 0.740 |
| Precision | 0.954 | 0.966 | 0.935 |
| Recall | 0.912 | 0.956 | 0.877 |

### Quantization impact (FP32 vs INT8)

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| mAP50 | 0.9569 | 0.9561 | -0.0008 |
| mAP50-95 | 0.8464 | 0.8458 | -0.0006 |
| Precision | 0.9507 | 0.9535 | +0.0028 |
| Recall | 0.9167 | 0.9124 | -0.0043 |

### Training curves

![Validation Metrics](training_curves/validation_metrics.png)

![Training Loss](training_curves/training_loss.png)

## Model details

- **Architecture:** Ultralytics YOLO26-nano (2.57M parameters)
- **Input size:** 640x640
- **Classes:** `0: panel`, `1: text`
- **INT8 TFLite size:** 2.71 MB
- **Inference speed:** ~100-180ms CPU

## Files

| File | Format | Size | Use case |
|------|--------|------|----------|
| `manga_panel_detector_fp32.pt` | PyTorch FP32 | ~15 MB | Fine-tuning, FP16 export, further training |
| `manga_panel_detector_int8.tflite` | TFLite INT8 | 2.71 MB | Android/mobile deployment |

## Usage

### Python (ultralytics)
```python
from ultralytics import YOLO

model = YOLO("manga_panel_detector_fp32.pt")
results = model.predict("manga_page.jpg", conf=0.25)

for box in results[0].boxes:
    cls = int(box.cls)  # 0=panel, 1=text
    conf = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    label = "panel" if cls == 0 else "text"
    print(f"{label} ({conf:.2f}): [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### Android (TFLite / LiteRT)

Use `manga_panel_detector_int8.tflite` with the TensorFlow Lite interpreter or Google's LiteRT runtime.

- **Input:** 640x640 RGB image normalized to [0, 1]
- **Output:** Up to 300 detections, each with [x1, y1, x2, y2, confidence, class_id]
- **Recommended confidence threshold:** 0.25

## Training

| Parameter | Value |
|-----------|-------|
| Base model | Ultralytics `yolo26n.pt` (pretrained on COCO, AGPL-3.0) |
| Training framework | Ultralytics (AGPL-3.0) |
| Dataset | Manga109-s (87 manga titles, ~18k pages, ~32k annotations) |
| Classes | panel (frame), text (speech/dialog) |
| Epochs | 66 (early stopping, patience=20) |
| Best epoch | 48 (mAP50 = 0.957) |
| Image size | 640x640 |
| Batch size | 16 |
| GPU | NVIDIA T4 (Google Colab) |
| Training time | ~4 hours |
| Augmentation | No hue/saturation shift (grayscale manga), no horizontal flip (RTL reading order), reduced mosaic (0.5), no rotation (axis-aligned panels) |

## License

### Model weights: AGPL-3.0

These weights are a fine-tune of [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics), which Ultralytics distributes under the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html). Ultralytics' position is that models trained or fine-tuned from their weights and code fall under AGPL-3.0 as well, so this model is released under AGPL-3.0.

In practice, this means:

- You can use, modify and redistribute the model, including in open-source projects, as long as the project that distributes it is released under AGPL-3.0 (or under GPL-3.0, which AGPL-3.0 allows combining with).
- If you run the model as part of a network service (a hosted API, a web app that runs it server-side, etc.), AGPL-3.0 requires that you make the complete source code of that service available to its users.
- Using it in a closed-source or proprietary product (including a paid app that does not publish its source) is not covered by AGPL-3.0. For that, you need an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

Earlier versions of this page listed the model as Apache 2.0. That label was incorrect, because I did not have the rights to grant it. If you adopted the model under that assumption, please review your project against the terms above.

I'm not a lawyer, and whether copyleft licenses extend to trained weights has not been settled in court. The above reflects the license of the base model and Ultralytics' published position, and is the safest reading for anyone redistributing the model.

### Training data: Manga109-s

The training data has its own [license terms](https://huggingface.co/datasets/hal-utokyo/Manga109-s), which apply in addition to AGPL-3.0:

- Results obtained from machine learning experiments on Manga109-s, including pretrained models, may be used commercially.
- When publishing or distributing results, the use of the Manga109-s dataset must be clearly indicated.
- Selling manga images from the dataset together with the results is forbidden, and the dataset itself may not be redistributed.

This repository contains only model weights, with no images or annotations from Manga109-s.

## Citation

If you use this model, please cite:
```bibtex
@misc{leoxs22_manga_panel_detector_2026,
    author={Leandro Narosky},
    title={{Manga Panel and Text Detector (YOLO26-nano)}},
    year={2026},
    publisher={Hugging Face},
    url={https://huggingface.co/leoxs22/manga-panel-detector-yolo26n}
}
```

### Base model

This model builds on Ultralytics YOLO26 (Copyright © Ultralytics, AGPL-3.0):
```bibtex
@software{ultralytics_yolo26,
    author={{Ultralytics}},
    title={{Ultralytics YOLO26}},
    year={2026},
    url={https://github.com/ultralytics/ultralytics},
    license={AGPL-3.0}
}
```

### Dataset

This model was trained on Manga109-s. Please cite:
```bibtex
@article{multimedia_aizawa_2020,
    author={Kiyoharu Aizawa and Azuma Fujimoto and Atsushi Otsubo and Toru Ogawa and Yusuke Matsui and Koki Tsubota and Hikaru Ikuta},
    title={Building a Manga Dataset ``Manga109'' with Annotations for Multimedia Applications},
    journal={IEEE MultiMedia},
    volume={27},
    number={2},
    pages={8--18},
    doi={10.1109/mmul.2020.2987895},
    year={2020}
}

@article{mtap_matsui_2017,
    author={Yusuke Matsui and Kota Ito and Yuji Aramaki and Azuma Fujimoto and Toru Ogawa and Toshihiko Yamasaki and Kiyoharu Aizawa},
    title={Sketch-based Manga Retrieval using Manga109 Dataset},
    journal={Multimedia Tools and Applications},
    volume={76},
    number={20},
    pages={21811--21838},
    doi={10.1007/s11042-016-4020-z},
    year={2017}
}
```