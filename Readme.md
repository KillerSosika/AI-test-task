# AI Test Task

This repository contains solutions for two independent Computer Vision tasks implemented in Python.

The project demonstrates practical experience with:

- Deep Learning
- Computer Vision
- Satellite imagery
- Feature matching
- Image registration
- Model fine-tuning
- Dataset preparation
- Visualization pipelines

---

# Repository Structure

```
AI-test-task
│
├── task1/
│   ├── notebooks/
│   ├── src/
│   └── ...
│
├── task2/
│   ├── data/
│   ├── notebooks/
│   ├── scripts/
│   ├── src/
│   ├── weights/
│   └── README.md
│
├── EuroSAT_RGB/
├── articles/
└── requirements.txt
```

---

# Task 1

Task 1 contains the solution for the first computer vision assignment.

Main components include:

- data exploration
- preprocessing
- model implementation
- evaluation
- visualization

(See `task1/README.md` for details.)

---

# Task 2 — Satellite Image Registration

Task 2 focuses on matching **Sentinel-2 satellite images** acquired at different times for change detection and deforestation analysis.

The implementation includes a complete image registration pipeline:

- Sentinel-2 dataset loader
- GeoJSON parser
- ROI extraction
- Tile generation
- Classical feature matching
- Transformer-based feature matching
- Fine-tuning pipeline
- Evaluation
- Visualization

---

# Implemented Matchers

The repository contains several matching approaches for comparison.

## Classical

- ORB
- SIFT

## Deep Learning

- LoFTR (Pretrained)
- LoFTR (Fine-Tuned)

Several experiments with LightGlue + ALIKED were also performed during development.

---

# Fine-Tuning

The repository includes an experimental fine-tuning pipeline for LoFTR on satellite imagery.

Main improvements include:

- custom supervision generation
- stable loss computation
- dataset filtering
- gradient clipping
- optimizer tuning
- checkpoint saving

---

# Technologies

- Python 3.11
- PyTorch
- Kornia
- OpenCV
- Rasterio
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Installation

Clone repository

```bash
git clone <repository>

cd AI-test-task
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running

## Task 1

See

```
task1/README.md
```

---

## Task 2

Open

```
task2/notebooks/04_comparison.ipynb
```

or run

```bash
python task2/scripts/run_deforestation_pipeline.py
```

To reproduce fine-tuning:

```bash
python task2/src/finetune/train.py
```

---

# Results

The project compares multiple feature matching methods using:

- number of matches
- confidence scores
- inference time
- qualitative visualization
- registration robustness

Example outputs include:

- ORB
- SIFT
- LoFTR (Pretrained)
- LoFTR (Fine-Tuned)

---

# Future Work

Potential improvements include:

- SuperPoint + LightGlue
- ROI proposal network
- self-supervised correspondence generation
- larger Sentinel-2 dataset
- quantitative benchmark metrics
- homography estimation evaluation
- change detection after registration

---

# Project Goals

The primary objective of this repository was to explore modern image matching techniques for remote sensing applications while comparing classical Computer Vision methods with Transformer-based approaches.

---

# Author

Developed as part of an AI / Computer Vision technical assignment.