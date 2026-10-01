# Boundary-Aware Mamba (B-Mamba) for Endoscopic Polyp Segmentation

This repository contains the official PyTorch implementation of **B-Mamba**, a novel State Space Model (SSM) architecture designed specifically for medical image segmentation, with a focus on detecting camouflaged endoscopic polyps.

## 📌 Overview
State Space Models (like Mamba) have emerged as powerful alternatives to Transformers, offering linear complexity $O(N)$ for long-sequence modeling. However, standard Mamba architectures often over-smooth high-frequency details, causing them to struggle with the fuzzy, ambiguous boundaries typical of medical lesions (e.g., polyps).

**B-Mamba solves this** by fundamentally altering the Mamba Selective Scan mechanism. Instead of relying solely on input features, the SSM dynamic parameters ($B, C, \Delta$) are explicitly conditioned on boundary gradient maps (extracted via Sobel operators). This forces the sequence model's memory and transition dynamics to mathematically adapt when crossing a lesion boundary.

## ✨ Key Features
- **Boundary-Conditioned SSM:** The first Mamba architecture to inject edge gradients directly into the state space matrices.
- **Medical Segmentation Focus:** Optimized for high-resolution medical scans where lesions lack contrast with surrounding mucosa.
- **Robust Evaluation:** Includes gold-standard medical evaluation metrics (Dice Similarity Coefficient, Intersection over Union) and differentiable Dice Loss.

## 🚀 Getting Started

### 1. Dataset
This project uses the **Kvasir-SEG** dataset for training and evaluation.
- The data should be placed in `dataset/Kvasir-SEG/`.
- Structure should be:
  ```
  dataset/Kvasir-SEG/
  ├── images/
  └── masks/
  ```

### 2. Project Structure
- `polyp_segmentation/b_mamba/b_mamba_model.py`: Core architecture of the Boundary-Aware Mamba.
- `polyp_segmentation/shared/dataset_polyp.py`: PyTorch DataLoader for medical images.
- `polyp_segmentation/shared/metrics_polyp.py`: Dice and IoU metrics.
- `train_polyp.py`: Main training loop.
- `job_train_polyp.pbs`: HPC submission script.

### 3. Training
To train the model on a GPU cluster using PBS:
```bash
qsub job_train_polyp.pbs
```
Or run locally:
```bash
python train_polyp.py
```

## 📚 References
If this codebase helps your research, consider exploring the foundational literature:
1. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (Gu & Dao, 2023)
2. *PraNet: Parallel Reverse Attention Network for Polyp Segmentation* (Fan et al., 2020)
3. *U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation* (Ma et al., 2024)
