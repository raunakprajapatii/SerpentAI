# 🐍 Indian Snake Species Classification — DINOv2 ViT-L/14

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-latest-blueviolet)](https://github.com/huggingface/pytorch-image-models)
[![Dataset](https://img.shields.io/badge/Dataset-SnakeCLEF%202022-green)](https://www.imageclef.org/node/288)
[![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fine-tuning **DINOv2 ViT-Large/14** to recognise **48 Indian snake species plus a `no_snake` rejection class** (49 classes total) from the SnakeCLEF 2022 dataset. Training uses a freeze-then-unfreeze schedule, mixed precision, gradient clipping and resumable checkpoints, and reaches **94.7% validation accuracy** (Top-5: 98.4%).

---

## 📋 Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset & Filtering](#dataset--filtering)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Authors](#author)

---

## Overview

Automated snake identification matters for biodiversity research and for snakebite treatment, where knowing the species guides antivenom choice. This project fine-tunes a **DINOv2 ViT-Large/14** backbone (via `timm`) on a filtered subset of SnakeCLEF 2022 containing **Indian species** with enough images to learn from.

Key highlights:
- Filtered **48 Indian species** from a global dataset of 270,251 images and 1,572 classes
- Added a dedicated **`no_snake` class** (1,982 images of snake-like objects and background scenes) so the model can reject non-snake inputs instead of forcing a species guess
- **Two-phase training** over 30 epochs: head-only for 7 epochs, then full fine-tuning
- Mixed precision (FP16), gradient clipping, label smoothing, strong augmentation (RandAugment), and batch-level checkpoint resumption
- Multi-GPU training with `DataParallel` (2 GPUs on Kaggle)
- Detailed evaluation: classification report, confusion matrix, ROC / PR curves, Top-K accuracy, calibration, error analysis, and embedding visualizations (PCA, t-SNE)

---

## Model Architecture

![Model Architecture](assets/model_architecture.png)

The model is a standard Vision Transformer with a linear classification head. There is no extra CNN branch or fusion module.

1. **Patchify**: the 518 × 518 × 3 input is split into non-overlapping 14 × 14 patches, giving 37 × 37 = **1,369 patches**.
2. **Linear projection**: each flattened patch (14 × 14 × 3 = **588** values) is projected to a **1024-dim** embedding.
3. **[CLS] token + position embeddings**: a learnable [CLS] token is prepended and position embeddings are added.
4. **Transformer encoder × 24**: each block applies `Norm → Multi-head Attention → residual add → Norm → MLP → residual add`.
5. **Classification head**: the final [CLS] representation (1024-d) goes through `Linear(1024 → 49)`, and a softmax converts the logits into class probabilities.

### Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backbone | DINOv2 ViT-L/14 (`vit_large_patch14_dinov2`) | Strong self-supervised visual features that transfer well to fine-grained recognition |
| Input size | 518 × 518 | DINOv2's native resolution (37 × 37 patches of 14 px) |
| Head | `Linear(1024 → 49)` (timm default) | Lightweight head on top of the CLS token |
| Phase 1 (epochs 1–7) | Backbone frozen, head only | Warm-starts the head before disturbing pretrained features |
| Phase 2 (epochs 8–30) | Full model unfrozen | Lets the backbone adapt to fine-grained scale and texture cues |
| Optimizer | AdamW (lr = 2e-5, weight decay = 1e-4) | Standard for transformer fine-tuning |
| Scheduler | CosineAnnealingLR (`T_max = 30`) | Smooth decay across the whole run |
| Loss | CrossEntropy (label smoothing = 0.1) | Reduces overconfidence on look-alike species |
| Precision | FP16 via `torch.amp.autocast` + `GradScaler` | Lower memory use and faster training |
| Grad clipping | `clip_grad_norm_` (max = 1.0) | Stabilises training, especially right after unfreezing |

---

## Dataset & Filtering

- **Source**: [SnakeCLEF 2022](https://www.imageclef.org/node/288), 270,251 images across 1,572 classes
- **Metadata files used**:
  - `SnakeCLEF2022-TrainMetadata.csv`: image paths and labels
  - `SnakeCLEF2022-ISOxSpeciesMapping.csv`: country-level species presence flags

**Filtering pipeline**

| Step | Result |
|------|--------|
| Species with `india == 1` in the ISO mapping | 159 species, 27,084 images |
| Keep species with more than 100 and fewer than 600 images | **48 species, 13,085 images** (106–599 images per class, mean 272.6) |
| Add `no_snake` negative class | +1,982 images → **49 classes, 15,067 images** |

- **Train / validation split**: 70 / 30, stratified by `class_id` (seed 42), giving **4,521 validation images**
- **Labels**: class IDs are remapped to a contiguous 0–48 range. Class `0` is `no_snake` (it was assigned `-1` before sorting).

### Augmentation Pipeline

| Split | Transforms |
|-------|-----------|
| Train | Resize(560) → RandomCrop(518) → RandomHorizontalFlip → RandomVerticalFlip → ColorJitter(0.2, 0.2, 0.2, 0.1) → RandAugment(num_ops=2, magnitude=9) → Normalize(ImageNet) |
| Val | Resize(518) → Normalize(ImageNet) |

Corrupted or unreadable images are skipped by falling back to the next sample, and `Image.LOAD_TRUNCATED_IMAGES` is enabled.

---

## Training Strategy

```
Epochs 1–7   │ Backbone FROZEN   │ Only head.parameters() trainable
─────────────┼───────────────────┼──────────────────────────────────
Epochs 8–30  │ Backbone UNFROZEN │ All parameters trainable
```

- **Batch size**: 8 (1,319 training batches per epoch)
- **Checkpointing**: every 500 batches and at the end of every epoch (`checkpoint.pth`, containing model, optimizer, scheduler, epoch, batch index and best accuracy)
- **Resumption**: training resumes from the saved epoch and batch index, re-applies the correct freeze/unfreeze state, and restores optimizer and scheduler state. This was used to continue across Kaggle session limits.
- **Best model**: `best_model.pth` is saved whenever validation accuracy improves; the final weights are saved as `snake_model.pth`
- **Monitoring**: per-batch gradient norm, per-epoch learning rate, train/val loss and validation accuracy are logged for plotting

---

## Results

| Metric | Value |
|--------|-------|
| Final validation accuracy (epoch 30) | **94.71%** |
| Best validation accuracy (epoch 25, from the accuracy curve) | 94.80% |
| Top-3 accuracy | 97.99% |
| Top-5 accuracy | 98.43% |
| Final train loss / val loss | 0.7042 / 0.8843 |
| Macro avg precision / recall / F1 | 0.94 / 0.93 / 0.94 |
| Weighted avg precision / recall / F1 | 0.95 / 0.95 / 0.95 |
| Validation images | 4,521 |
| Classes | 49 (48 species + `no_snake`) |

> Note: train loss plateaus near 0.70 because of label smoothing (0.1), which sets a loss floor and caps the model's confidence at about 0.90.

### Observations

- **Most classes are strong.** Most classes score F1 ≥ 0.93. The `no_snake` class (class 0, 595 validation images) reaches precision 0.98 and recall 0.99.
- **A few classes are harder**, and they are all visually similar species:

  | Class | F1 |
  |-------|----|
  | 3 | 0.75 |
  | 33 | 0.77 |
  | 2 | 0.86 |
  | 27 | 0.87 |
  | 37 | 0.88 |
  | 39 | 0.89 |

  The most common single error is class 3 → 2 (7 cases). The high-confidence mistakes include similar-looking green snakes.
- **Confident errors are rare**: only 9 wrong predictions have confidence above 0.9. 77 correct predictions have confidence below 0.5, mostly camouflaged snakes in cluttered or dark scenes.
- **Calibration**: the reliability curve sits slightly above the diagonal, so the model is mildly under-confident, which is expected with label smoothing.
- **Embeddings**: t-SNE of the CLS features shows well-separated species clusters, with overlap mainly among the confusable classes.

---

## Visualizations

### Training curves

| Train vs Val Loss | Validation Accuracy |
|:-:|:-:|
| ![Loss Curve](assets/loss_curve.png) | ![Accuracy Curve](assets/accuracy_curve.png) |

| Learning Rate Schedule | Gradient Norm per Batch |
|:-:|:-:|
| ![LR Curve](assets/lr_curve.png) | ![Gradient Norm](assets/grad_norm.png) |

### Evaluation

| Confusion Matrix | Per-Class Accuracy |
|:-:|:-:|
| ![Confusion Matrix](assets/confusion_matrix.png) | ![Per-Class Accuracy](assets/per_class_accuracy.png) |

| ROC Curve (per class) | Precision–Recall Curve (per class) |
|:-:|:-:|
| ![ROC Curve](assets/roc_curve.png) | ![PR Curve](assets/pr_curve.png) |

![Classification Report Heatmap](assets/classification_report_heatmap.png)

| Top-K Accuracy | Calibration Curve |
|:-:|:-:|
| ![Top-K](assets/topk_accuracy.png) | ![Calibration](assets/calibration_curve.png) |

### Confidence & error analysis

| Confidence: Correct vs Incorrect | Per-Class Confidence |
|:-:|:-:|
| ![Confidence Distribution](assets/confidence_dist.png) | ![Per-Class Confidence](assets/per_class_conf_boxplot.png) |

![Most Confused Pairs](assets/confused_pairs.png)

![High-Confidence Wrong Predictions](assets/high_conf_wrong.png)

![Low-Confidence Correct Predictions](assets/low_conf_correct.png)

### Dataset & embeddings

| Snake vs No-Snake | Train vs Val Distribution |
|:-:|:-:|
| ![Class Balance](assets/class_balance_pie.png) | ![Train Val Distribution](assets/train_val_dist.png) |

| PCA of CLS Embeddings | t-SNE of CLS Embeddings |
|:-:|:-:|
| ![PCA](assets/pca_embeddings.png) | ![t-SNE](assets/tsne_embeddings.png) |

![PCA Explained Variance](assets/pca_variance.png)

---

## How to Run

### On Kaggle (recommended)
1. Open the notebook: *[your Kaggle notebook link here]*
2. Add the SnakeCLEF 2022 competition data: **Data → Add Data → Competitions → SnakeCLEF2022**
3. Add the `no_snake` dataset (snake-like objects and background images)
4. Enable GPUs: **Settings → Accelerator → GPU T4 x2**
5. Run all cells. If the session is interrupted, attach the saved `checkpoint.pth` as a dataset and rerun; training resumes from the stored epoch and batch.

### Locally
```bash
git clone https://github.com/raunakprajapatii/snake-species-classifier
cd snake-species-classifier
pip install torch torchvision timm scikit-learn matplotlib seaborn pandas pillow
jupyter notebook SnakeSpeciesClassifier.ipynb
```

Update these paths in the notebook before running:
```python
BASE_PATH          = "/path/to/snakeclef2022"
TRAIN_METADATA     = BASE_PATH + "/SnakeCLEF2022-TrainMetadata.csv"
ISO_MAPPING        = BASE_PATH + "/SnakeCLEF2022-ISOxSpeciesMapping.csv"
TRAIN_IMG_DIR      = BASE_PATH + "/SnakeCLEF2022-medium_size/SnakeCLEF2022-medium_size"
NO_SNAKE_TRAIN_DIR = "/path/to/no_snake_dataset/no_snake_training"
```

### Inference sketch
```python
import torch, timm
from torchvision import transforms
from PIL import Image

model = timm.create_model("vit_large_patch14_dinov2", pretrained=False, num_classes=49)
state = torch.load("snake_model.pth", map_location="cpu")
# the checkpoint was saved from DataParallel, so strip the "module." prefix
state = {k.replace("module.", "", 1): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

tfm = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

x = tfm(Image.open("snake.jpg").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    probs = model(x).softmax(dim=1)
print(probs.topk(5))   # class 0 = no_snake
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `PyTorch` | Training loop, loss, optimizer, mixed precision (AMP), `DataParallel` |
| `timm` | DINOv2 ViT-L/14 model and pretrained weights |
| `torchvision` | Transforms and augmentation (incl. RandAugment) |
| `scikit-learn` | Stratified split, classification report, confusion matrix, ROC/PR, calibration, PCA, t-SNE |
| `Matplotlib / Seaborn` | All training and evaluation plots |
| `Pandas / NumPy` | Metadata loading, filtering and array operations |
| `Pillow` | Image loading with truncation tolerance |

---

## Project Structure

```
snake-species-classifier/
│
├── SnakeSpeciesClassifier.ipynb
├── README.md
└── assets/
    ├── model_architecture.png
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── lr_curve.png
    ├── grad_norm.png
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    ├── classification_report_heatmap.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── topk_accuracy.png
    ├── calibration_curve.png
    ├── confidence_dist.png
    ├── per_class_conf_boxplot.png
    ├── confused_pairs.png
    ├── high_conf_wrong.png
    ├── low_conf_correct.png
    ├── class_balance_pie.png
    ├── train_val_dist.png
    ├── pca_embeddings.png
    ├── tsne_embeddings.png
    └── pca_variance.png
```

---

## Author

**Rounak** — [GitHub](https://github.com/raunakprajapatii) · [LinkedIn](http://www.linkedin.com/in/rounak-prajapati-3896jee)

## Co-Authors

- **Pratik Prajapati** — [GitHub](https://github.com/USERNAME_HERE)
- **Vinita Soni** — [GitHub](https://github.com/USERNAME_HERE)
- **Shraddha Singh** — [GitHub](https://github.com/USERNAME_HERE)
- **Priya Singh** — [GitHub](https://github.com/USERNAME_HERE)

*Built as an academic group course project on deep learning and computer vision.*

---

## License

This project is licensed under the [MIT License](LICENSE).
