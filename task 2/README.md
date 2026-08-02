# 🛰️ Satellite Image Registration for Deforestation Detection using LoFTR

This project was developed as part of an AI/CV technical assignment focused on satellite image registration and change detection.

The objective is to accurately match **Before / After Sentinel-2 satellite images** in order to support deforestation analysis. The project compares traditional feature-based approaches with modern transformer-based local feature matching (LoFTR) and includes an experimental fine-tuning pipeline.

---

# Project Overview

The pipeline consists of several stages:

- loading Sentinel-2 imagery
- parsing GeoJSON deforestation annotations
- ROI extraction around labeled regions
- tile generation
- feature matching
- model evaluation
- visualization
- experimental LoFTR fine-tuning

The implementation is fully modular and designed for experimentation.

---

# Features

- Sentinel-2 JP2 dataset support
- GeoJSON annotation parser
- ROI extraction from deforestation polygons
- Tile generation with overlap
- Classical feature matching
  - ORB
  - SIFT
- Deep feature matching
  - LoFTR (pretrained)
  - LoFTR (fine-tuned)
- Visualization utilities
- Evaluation metrics
- Training pipeline
- Inference pipeline

---

# Engineering Challenges

## 1. Fine-tuning LoFTR

During fine-tuning several issues appeared that required modifications to the training pipeline.

### Problem

A naive Binary Cross Entropy objective over the entire coarse correlation matrix introduced an extreme class imbalance.

Most cells correspond to negative matches while only a very small subset represents valid correspondences.

This caused unstable optimization and model collapse.

### Solution

The training pipeline was redesigned to optimize only valid correspondences using the supervision strategy inspired by the original LoFTR implementation.

This stabilized optimization and allowed the model to successfully adapt to satellite imagery.

---

## 2. Ground Truth Quality

### Problem

Pseudo Ground Truth correspondences were generated using SIFT.

However, SIFT produces very sparse matches over low-texture agricultural regions, which negatively affects supervision.

### Solution

Invalid crops containing too few correspondences are skipped during training.

```python
if pos_mask.sum() < 5:
    continue
```

This prevents the network from learning from unreliable supervision.

---

## 3. GPU Utilization

### Problem

GPU utilization remained close to 1% while CPU usage reached 100%.

The bottleneck was located in online data preparation:

- ROI extraction
- tile generation
- SIFT correspondence computation

### Solution

The data loading pipeline was optimized and gradient clipping was introduced:

- AdamW optimizer
- gradient clipping
- weight decay

Further optimization could include offline caching of generated supervision.

---

# Model Comparison

The project evaluates four levels of feature matching:

| Level | Model |
|--------|----------------|
| Level 1 | ORB |
| Level 2 | SIFT |
| Level 3 | LoFTR (Pretrained) |
| Level 4 | LoFTR (Fine-Tuned) |

The comparison notebook visualizes

- number of matches
- confidence distribution
- inference time
- qualitative registration quality

---

# Project Structure

```text
task2/
│
├── data/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_demo_inference.ipynb
│   ├── 03_finetuning_analysis.ipynb
│   └── 04_comparison.ipynb
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── dataset/
│   ├── matching/
│   ├── finetune/
│   ├── visualization/
│   ├── metrics/
│   └── tiling/
│
├── weights/
│
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone <repository>

cd task2

pip install -r requirements.txt
```

---

# Running Inference

Launch the comparison notebook:

```text
notebooks/04_comparison.ipynb
```

The notebook automatically loads the fine-tuned LoFTR checkpoint from:

```text
weights/
```

and visualizes

- matched keypoints
- confidence
- qualitative registration results

---

# Fine-Tuning

To reproduce fine-tuning:

```bash
python src/finetune/train.py
```

Training uses

- Sentinel-2 image pairs
- ROI extraction
- automatically generated correspondence supervision

---

# Technologies

- Python
- PyTorch
- Kornia
- OpenCV
- Rasterio
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Future Improvements

Possible future extensions include

- ROI detector before matching
- self-supervised correspondence generation
- SuperPoint + LightGlue comparison
- multi-scale image registration
- larger Sentinel-2 training dataset
- quantitative registration benchmarks

