# Liver Tumor Segmentation from CT Scans

Automated medical image segmentation system for detecting and classifying liver tumors from CT scans using deep learning. Developed as part of a Master's thesis at Politechnika Krakowska.

## Results

Evaluated on a held-out test set (20% of 131 patients, 3,833 slices) from the full LiTS dataset.

| Metric | Baseline (Kaggle subset) | Final model (full LiTS) | Change |
|---|---|---|---|
| Pixel Accuracy | 98.84% | **99.39%** | +0.55pp |
| Liver Dice | 0.7977 | **0.8831** | +8.5pp |
| Liver Recall | 0.8143 | **0.8892** | +7.5pp |
| Liver Hausdorff | 40.96 | **26.64** | -35% |
| Tumor Dice | 0.7017 | **0.7546** | +5.3pp |
| Tumor Recall | 0.7069 | **0.7577** | +5.1pp |
| Tumor Precision | 0.7283 | **0.7765** | +4.8pp |
| Tumor Hausdorff | 13.06 | **11.46** | -12% |
| Mean Dice | 0.7750 | **0.8483** | +7.3pp |

Results are competitive with published 2D U-Net benchmarks on the LiTS dataset.

## Dataset

- **Full LiTS** (Medical Segmentation Decathlon Task03_Liver): 131 patients, 19,163 slices
- 3 classes: background (93%), liver (6.6%), tumor (0.37%)
- Data prepared as 2D slices in NIfTI format with HU windowing

## Architecture

**2D U-Net** (PyTorch) for multi-class semantic segmentation.

### Key design decisions

| Component | Choice | Reason |
|---|---|---|
| Loss | Hybrid Focal-Dice (40%/60%) | Handles extreme class imbalance |
| Class weights | 1 : 10 : 100 (BG/Liver/Tumor) | Forces focus on rare tumor class |
| Sampling | Stratified tumor oversampling | 37.5% of batches contain tumor slices |
| Inference | Test-Time Augmentation (TTA) | +1.7pp Tumor Dice, +2.8pp Liver Dice |
| Optimizer | Adam, LR=1e-3 | ReduceLROnPlateau scheduler |
| Training | 100 epochs, batch size 8 | RTX A2000 8GB |

## Tech Stack

- **Deep Learning:** PyTorch, U-Net
- **Medical Imaging:** nibabel, NIfTI processing, HU windowing
- **Augmentation:** Albumentations (ShiftScaleRotate, flip, brightness)
- **Data Science:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Grad-CAM
- **Tools:** Jupyter Notebooks, Git

## Project Structure

```
├── src/
│   ├── unet.py                  # U-Net architecture
│   ├── unet_parts.py            # U-Net building blocks
│   ├── Dataset.py               # PyTorch Dataset class
│   ├── train.py                 # Training (baseline, Kaggle subset)
│   ├── train_finetune.py        # Training (full LiTS dataset)
│   ├── prepare_msd_data.py      # MSD → prepared_fine_tuning pipeline
│   ├── optimal_segmentation.py  # Post-processing optimization
│   └── img_transformations.py   # Augmentation pipeline
├── notebooks/
│   ├── 00-dataset-analysis.ipynb
│   ├── 01-explore-nii.ipynb
│   ├── 02-exploratory-data-analysis.ipynb
│   └── 03-model-evaluation.ipynb
└── data/
    ├── prepared/                # Kaggle LiTS subset (~47 patients)
    └── prepared_fine_tuning/    # Full LiTS (131 patients)
```

## Evaluation Methodology

- Each model evaluated on its own held-out test set (80/20 split)
- Metrics computed per-slice (2D), not per-patient volume (3D)
- Hausdorff distance computed on slices containing the relevant class
- Direct comparison between baseline and final model is indicative due to different test sets
