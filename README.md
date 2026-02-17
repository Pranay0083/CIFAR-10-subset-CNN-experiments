# CIFAR-10 Subset CNN Experiments

A hands-on task involving training and experimenting with a small CNN on a CIFAR-10 subset.  
Approach: **Architecture first → Regularization → Dropout → Augmentation**.

## Project Structure

```
├── notebooks/
│   ├── Phase 0 — Preliminary (simple 2-layer CNN)
│   │   ├── 0.0_baseline_simple_cnn.ipynb
│   │   ├── 0.1_simple_batchnorm.ipynb
│   │   ├── 0.2_simple_dropout.ipynb
│   │   └── 0.3_simple_augmentation.ipynb
│   ├── Phase 1 — Architecture exploration
│   │   ├── 1.1_increasing_model_capacity.ipynb
│   │   ├── 1.2_gap_replacing_dense_layers.ipynb
│   │   ├── 1.3_stacked_conv_blocks.ipynb
│   │   ├── 1.4_widening_information_bottleneck.ipynb
│   │   ├── 1.5_skip_connections.ipynb
│   │   └── 1.6_one_cycle_lr.ipynb
│   ├── Phase 2 — Regularization (on best architecture)
│   │   ├── 2.1_weight_decay.ipynb
│   │   └── 2.2_label_smoothing.ipynb
│   ├── Phase 3 — Dropout
│   │   ├── 3.1_stacked_blocks_dropout.ipynb
│   │   └── 3.2_wider_filters_dropout.ipynb
│   ├── Phase 4 — Data Augmentation
│   │   ├── 4.1_wider_filters_augmentation.ipynb
│   │   └── 4.2_augmentation_longer_training.ipynb
│   └── Phase 5 — Final
│       └── 5.0_final_comparison.ipynb
├── data/                        # CIFAR-10 (auto-downloaded)
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -p venv python==3.12
conda activate venv
pip install -r requirements.txt
```

## Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training subset | 5,000 images (first 5k) |
| Validation subset | 1,000 images (first 1k) |
| Image size | 32×32×3 |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Normalization | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |

## Baseline — Model 0 (SimpleCNN)

**Architecture:**
```
Conv2d(3→16, 3×3, pad=1) → ReLU → MaxPool(2)
Conv2d(16→32, 3×3, pad=1) → ReLU → MaxPool(2)
Flatten → Linear(2048→128) → ReLU → Linear(128→10)
```

**Training config:** Adam (lr=0.001) · CrossEntropyLoss · 15 epochs · batch size 64

## Model Comparison Table

### Phase 0 — Preliminary (Simple 2-Layer CNN)

| # | Model | Params | Peak Val Acc | Train–Val Gap | Diagnosis | Notebook |
|---|---|---|---|---|---|---|
| 0.0 | SimpleCNN (baseline) | 268,650 | 58.70% | +37.72% | Overfitting | [0.0_baseline_simple_cnn.ipynb](notebooks/0.0_baseline_simple_cnn.ipynb) |
| 0.1 | SimpleCNN + BatchNorm | 269,002 | 55.10% | +47.36% | Overfitting | [0.1_simple_batchnorm.ipynb](notebooks/0.1_simple_batchnorm.ipynb) |
| 0.2 | SimpleCNN + Dropout | 268,650 | 57.80% | -2.12% | Underfitting | [0.2_simple_dropout.ipynb](notebooks/0.2_simple_dropout.ipynb) |
| 0.3 | SimpleCNN + Augmentation | 268,650 | 55.90% | +2.60% | Underfitting | [0.3_simple_augmentation.ipynb](notebooks/0.3_simple_augmentation.ipynb) |

### Phase 1 — Architecture Exploration

| # | Model | Params | Peak Val Acc | Train–Val Gap | Diagnosis | Notebook |
|---|---|---|---|---|---|---|
| 1.1 | Increased Capacity (3-layer) | 620,362 | 58.90% | +37.92% | Overfitting | [1.1_increasing_model_capacity.ipynb](notebooks/1.1_increasing_model_capacity.ipynb) |
| 1.2 | GAP (Global Avg Pool) | 94,538 | 50.80% | +3.48% | Underfitting | [1.2_gap_replacing_dense_layers.ipynb](notebooks/1.2_gap_replacing_dense_layers.ipynb) |
| 1.3 | Stacked Conv Blocks + BN + GAP | 289,194 | **65.20%** | +29.66% | Overfitting | [1.3_stacked_conv_blocks.ipynb](notebooks/1.3_stacked_conv_blocks.ipynb) |
| 1.4 | Wide Filters (64→128→256) + BN + GAP | 1,149,770 | **68.00%** | +36.10% | Overfitting | [1.4_widening_information_bottleneck.ipynb](notebooks/1.4_widening_information_bottleneck.ipynb) |
| 1.5 | Skip Connections (ResBlocks) | 1,190,922 | 57.10% | +39.54% | Overfitting | [1.5_skip_connections.ipynb](notebooks/1.5_skip_connections.ipynb) |
| 1.6 | One Cycle LR (on 1.5 arch) | 1,190,922 | 57.70% | +41.82% | Overfitting | [1.6_one_cycle_lr.ipynb](notebooks/1.6_one_cycle_lr.ipynb) |

### Phase 2 — Regularization (on best architecture from Phase 1)

| # | Model | Params | Peak Val Acc | Train–Val Gap | Diagnosis | Notebook |
|---|---|---|---|---|---|---|
| 2.1 | Stacked Blocks + Weight Decay (L2) | 289,194 | 61.60% | +35.76% | Severe overfitting | [2.1_weight_decay.ipynb](notebooks/2.1_weight_decay.ipynb) |
| 2.2 | Stacked Blocks + Label Smoothing | 289,194 | 61.60% | +35.76% | Still overfitting | [2.2_label_smoothing.ipynb](notebooks/2.2_label_smoothing.ipynb) |

### Phase 3 — Dropout

| # | Model | Params | Peak Val Acc | Train–Val Gap | Diagnosis | Notebook |
|---|---|---|---|---|---|---|
| 3.1 | Stacked Blocks + Dropout | 289,194 | 63.60% | -0.50% | Well-regularized | [3.1_stacked_blocks_dropout.ipynb](notebooks/3.1_stacked_blocks_dropout.ipynb) |
| 3.2 | Wider Filters + Dropout | 1,149,770 | 63.80% | +5.06% | Mild overfitting | [3.2_wider_filters_dropout.ipynb](notebooks/3.2_wider_filters_dropout.ipynb) |

### Phase 4 — Data Augmentation

| # | Model | Params | Peak Val Acc | Train–Val Gap | Diagnosis | Notebook |
|---|---|---|---|---|---|---|
| 4.1 | Wider + Dropout + Augmentation (15 ep) | 1,149,770 | 60.80% | -2.12% | Good generalization | [4.1_wider_filters_augmentation.ipynb](notebooks/4.1_wider_filters_augmentation.ipynb) |
| 4.2 | Wider + Dropout + Aug + OneCycleLR (50 ep) | 1,149,770 | **75.00%** | +11.84% | Healthy | [4.2_augmentation_longer_training.ipynb](notebooks/4.2_augmentation_longer_training.ipynb) |

## Conclusion

**Best model: 4.2 — Wider Filters + Dropout + Augmentation + OneCycleLR (50 epochs) → 75.00% val accuracy.**

Starting from a simple 2-layer CNN at 58.70%, systematic experimentation across five phases revealed several key insights for training on a small (5k sample) CIFAR-10 subset:

1. **Architecture matters most.** Stacked conv blocks with BatchNorm and GAP (Model 1.3) jumped to 65.20%, and widening filters to 64→128→256 (Model 1.4) pushed it to 68.00% — the two biggest accuracy gains in the entire study. Skip connections (1.5) and OneCycleLR alone (1.6) did not help on this small dataset.

2. **Traditional regularization alone is insufficient.** Weight decay (2.1) and label smoothing (2.2) both stalled at 61.60% with ~36% train–val gaps. On a data-starved regime, penalizing weights or softening labels cannot compensate for the lack of data diversity.

3. **Dropout is the single most effective regularizer.** Adding dropout to the stacked blocks (3.1) collapsed the train–val gap from +29.66% to **−0.50%** while maintaining 63.60% accuracy — the best-regularized model before augmentation.

4. **Data augmentation + longer training unlocks the best performance.** Augmentation with only 15 epochs (4.1) actually hurt accuracy (60.80%) because the model didn't have enough time to learn from augmented samples. Extending to 50 epochs with OneCycleLR (4.2) achieved **75.00%** — a **+16.30 pp** improvement over the baseline and **+7.00 pp** over the best architecture-only model.

5. **Order of operations matters.** The progression *architecture → regularization → dropout → augmentation → training schedule* proved effective. Each phase built on the previous one's gains rather than trying to fix fundamental capacity or generalization issues with the wrong tool.
