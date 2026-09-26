---
license: apache-2.0
pipeline_tag: object-detection
tags:
- manga
- text-detection
- yolo
- ocr
- onnx
datasets:
- hal-utokyo/Manga109-s
base_model:
- Ultralytics/YOLO26
---

# Manga Text Bubble Detector (YOLO-Nano)

This repository contains a lightweight object detection model designed to detect speech bubbles and text regions in Manga pages. It is useing **YOLO26** architecture that utilizes an **End-to-End (Head-to-Head)** prediction head, eliminating the need for Non-Maximum Suppression (NMS) during inference. It being trained on a diverse dataset of English, Vietnamese, and Japanese manga.

## Dataset Details

The model was trained on a composite dataset containing **5,595 images**, utilizing a mix of high-quality scans and fan translations. The data was split into **80% Train, 10% Validation, and 10% Test**.

### Folder Structure & Counts
```text
/Manga_Project
├── images/
│   ├── train/ (4,416 images)
│   ├── val/   (579 images)
│   └── test/  (600 images)
└── labels/ (Corresponding YOLO .txt files)
```

### Data Sources
* **Manga109-s:** ~3,000 images (High-quality official scans)
* **Mangadex-EN:** ~2,000 images (English fan translations)
* **Mangadex-VI:** ~1,000 images (Vietnamese fan translations)

*Note: Some Mangadex images were filtered out during processing due to download errors or API limits, resulting in the final count of ~5.6k images.*

## Model Performance

The model was trained for 100 epochs (with early stopping) on an Nvidia Tesla T4 at **1280x1280** resolution.

### Metrics Comparison (Test Set - 600 Images)
| Model | Class | Precision | Recall | mAP@50 | mAP@50-95 | Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | Text | 0.929 | 0.863 | 0.947 | 0.765 | 2.4M |
| **YOLO26s** | Text | **0.937** | **0.893** | **0.961** | **0.802** | 9.5M |

* **Inference Speed (T4 GPU):**
  * **Nano:** ~11.0ms per image
  * **Small:** ~27.5ms per image

## Usage

### 1. Using PyTorch (Python)
Requires `ultralytics` package.

```python
from ultralytics import YOLO

# Load the model
model = YOLO('model.pt')

# Run inference
# Note: imgsz=1280 is recommended for small text bubbles
results = model.predict('path/to/manga_page.jpg', imgsz=1280, conf=0.25)

# Display result
results[0].show()
```

### 2. Using ONNX (Python)
Useful for deployment without PyTorch dependencies.

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load model
session = ort.InferenceSession('model.onnx')

# Preprocess Image
img = cv2.imread('test.jpg')
img = cv2.resize(img, (1280, 1280))
img = img.transpose((2, 0, 1)) # HWC -> CHW
img = np.expand_dims(img, axis=0) # Add batch dimension
img = img.astype(np.float32) / 255.0 # Normalize

# Run Inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img})

print("Output Shape:", outputs[0].shape)
# Returns (1, 300, 6)
```

## Example Result in YOLO26-n
<details open>
<summary>
  <strong>Test 1<strong>
</summary>
<img src="https://huggingface.co/Kiuyha/Manga-Bubble-YOLO/resolve/main/Test1.png" alt="Test 1" />
</details>

<details>
<summary><strong>Test 2<strong></summary>
<img src="https://huggingface.co/Kiuyha/Manga-Bubble-YOLO/resolve/main/Test2.png" alt="Test 2" />
</details>

<details>
<summary><strong>Test 3<strong></summary>
<img src="https://huggingface.co/Kiuyha/Manga-Bubble-YOLO/resolve/main/Test3.png" alt="Test 3" />
</details>


## Training Configuration

The model was trained using the following hyperparameters:
```python
model.train(
  data='dataset/data.yaml',
  epochs=100,
  patience=10,
  batch=8,
  lr0=0.0001,
  imgsz=1280,
  device='cuda'
)
```

## Credits & Citations

We gratefully acknowledge the following datasets and tools used to build this project:

**Manga109-s Dataset**
Used for high-quality Japanese manga panel data.
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
```

**Magi (Annotation Tool)**
Used to auto-annotate the Mangadex portion of the dataset.
```bibtex
@misc{magiv2,
  title={Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names}, 
  author={Ragav Sachdeva and Gyungin Shin and Andrew Zisserman},
  year={2024},
  eprint={2408.00298},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={[https://arxiv.org/abs/2408.00298](https://arxiv.org/abs/2408.00298)}, 
}
```

Note: The dataset cannot be released due to copyright concerns.