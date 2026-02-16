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
| 1.1 | Increased Capacity | 620,362 | 95.42% | 57.50% | 58.90% | +37.92% | Overfitting | [1.1_increasing_model_capacity.ipynb](notebooks/1.1_increasing_model_capacity.ipynb) |
| 1.2 | GAP (Global Avg Pool) | 94,538 | 54.08% | 50.60% | 50.80% | +3.48% | Underfitting | [1.2_gap_replacing_dense_layers.ipynb](notebooks/1.2_gap_replacing_dense_layers.ipynb) |
| 1.3 | Stacked Conv Blocks | 289,194 | 94.86% | 65.20% | 65.20% | +29.66% | Overfitting (Best) | [1.3_stacked_conv_blocks.ipynb](notebooks/1.3_stacked_conv_blocks.ipynb) |
| 1.4 | Wide Filters | 1,149,770 | 92.70% | 56.60% | 68.00% | +36.10% | Overfitting | [1.4_widening_information_bottleneck.ipynb](notebooks/1.4_widening_information_bottleneck.ipynb) |
| 1.5 | Skip Connections | 1,190,922 | 95.54% | 56.00% | 57.10% | +39.54% | Overfitting | [1.5_skip_connections.ipynb](notebooks/1.5_skip_connections.ipynb) |
| 1.6 | One Cycle LR | 1,190,922 | 99.12% | 57.30% | 57.70% | +41.82% | Overfitting | [1.6_one_cycle_lr.ipynb](notebooks/1.6_one_cycle_lr.ipynb) |

## Conclusion

**Best Model to proceed with:** Model 1.3 (Stacked Conv Blocks).
It achieved the high stable validation accuracy (65.20%) while maintaining a reasonable parameter count. Although it clearly overfits (94% train vs 65% val), this indicates it has the capacity to learn the dataset well, making it the ideal candidate for applying regularization techniques (Phase 2).

