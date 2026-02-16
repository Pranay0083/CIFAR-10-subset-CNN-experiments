# CIFAR-10 Subset CNN Experiments

A hands-on task involving training and experimenting with a small CNN (or comparable baseline model) on a CIFAR-10 subset.

## Project Structure

```
├── notebooks/
│   ├── 01_baseline_cnn.ipynb              # Phase 1 & 2 — Lab setup + Baseline
│   ├── 02_tweak1_regularization.ipynb     # Tweak 1 — Baseline vs BatchNorm
│   ├── 03_tweak2_dropout.ipynb            # Tweak 2 — Baseline vs Dropout
│   └── 04_tweak3_augmentation.ipynb       # Tweak 3 — No Augmentation vs Bag of Tricks
├── data/                        # CIFAR-10 (auto-downloaded)
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -p venv python==3.12
conda activate venv
pip install -r requirments.txt
```

## Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training subset | 5,000 images (first 5k) |
| Validation subset | 1,000 images (first 1k) |
| Image size | 32×32×3 |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Normalization | mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) |

## Baseline — Model 0 (SimpleCNN)

**Architecture:**
```
Conv2d(3→16, 3×3, pad=1) → ReLU → MaxPool(2)
Conv2d(16→32, 3×3, pad=1) → ReLU → MaxPool(2)
Flatten → Linear(2048→128) → ReLU → Linear(128→10)
```

**Training config:** Adam (lr=0.001) · CrossEntropyLoss · 15 epochs · batch size 64

## Model Comparison Table

| # | Model | Params | Train Acc | Val Acc | Peak Val Acc | Train–Val Gap | Diagnosis | Notebook |
|---|---|---|---|---|---|---|---|---|
| 0 | SimpleCNN (baseline) | 268,650 | 92.82% | 55.10% | 58.70% | +37.72% | Overfitting | [01_baseline_cnn.ipynb](notebooks/01_baseline_cnn.ipynb) |
| 1 | SimpleCNN + BatchNorm | 269,002 | 99.86% | 55.10% | 55.10% | +47.36% | Overfitting | [02_tweak1_regularization.ipynb](notebooks/02_tweak1_regularization.ipynb) |
| 2 | SimpleCNN + Dropout | 268,650 | 55.68% | 57.80% | 57.80% | -2.12% | UNDERFITTING | [03_tweak2_dropout.ipynb](notebooks/03_tweak2_dropout.ipynb) |
| 3 | SimpleCNN + Augmentation (Flip/Rot/Crop) | 268,650 | 56.70% | 54.10% | 55.90% | +2.60% | UNDERFITTING | [04_tweak3_augmentation.ipynb](notebooks/04_tweak3_augmentation.ipynb) |
