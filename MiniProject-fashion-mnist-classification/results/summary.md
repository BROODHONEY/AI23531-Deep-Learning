# Results summary

Test metrics are for the checkpoint with the best validation macro-F1 (mean ± sample std over seeds; std omitted when n=1). 'Gap' = training accuracy minus validation accuracy at the selected epoch (for augmented models, training accuracy is measured on augmented batches, so it is not comparable). Train time is wall-clock per run on 1 CPU thread (two jobs shared a 2-core machine).

| Model | Params | Epochs | Seeds | Test acc (%) | Test macro-F1 (%) | Best epoch | Gap (pts) | Train time (min) | Inference (ms/img) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression (baseline) | 7,850 | 20 | 3 | 84.06 ± 0.13 | 84.05 ± 0.17 | 14.7 | -0.5 | 0.1 | 0.001 |
| MLP | 535,818 | 20 | 3 | 89.20 ± 0.05 | 89.19 ± 0.11 | 17.3 | +0.8 | 1.1 | 0.010 |
| Simple CNN | 421,642 | 10 | 3 | 92.42 ± 0.12 | 92.41 ± 0.12 | 9.7 | +1.8 | 7.9 | 0.631 |
| ResNet-lite | 77,754 | 12 | 3 | 91.79 ± 0.14 | 91.78 ± 0.15 | 11.7 | -1.3 | 21.0 | 0.840 |
| Tiny ViT | 139,018 | 15 | 1 | 87.49 | 87.38 | 15.0 | -2.0 | 42.7 | 0.911 |
| ResNet-lite, no augmentation | 77,754 | 12 | 2 | 92.69 ± 0.02 | 92.67 ± 0.01 | 11.5 | +1.7 | 20.4 | 0.853 |

Best model by mean validation macro-F1: **ResNet-lite, no augmentation**.

## Paired significance (exact McNemar test, seed 0, 10,000 test images)

| A | B | only A correct | only B correct | p-value |
|---|---|---:|---:|---:|
| ResNet-lite, no augmentation | Simple CNN | 260 | 248 | 0.63 |
| Logistic regression (baseline) | ResNet-lite, no augmentation | 188 | 1038 | 9.1e-143 |
| MLP | Simple CNN | 195 | 526 | 5.7e-36 |
| Simple CNN | ResNet-lite | 349 | 258 | 0.00025 |
| ResNet-lite | Tiny ViT | 634 | 218 | 8.5e-48 |
| ResNet-lite, no augmentation | ResNet-lite | 300 | 197 | 4.4e-06 |

## Most frequent confusions — ResNet-lite, no augmentation (summed over seeds)

| True class | Predicted as | Count | % of true class |
|---|---|---:|---:|
| Shirt | T-shirt/top | 185 | 9.2 |
| T-shirt/top | Shirt | 153 | 7.6 |
| Shirt | Coat | 118 | 5.9 |
| Coat | Shirt | 99 | 5.0 |
| Shirt | Pullover | 95 | 4.8 |
| Coat | Pullover | 80 | 4.0 |

## Per-class F1 — ResNet-lite, no augmentation (mean over seeds)

| Class | F1 (%) |
|---|---:|
| Shirt | 78.33 |
| T-shirt/top | 88.04 |
| Coat | 89.00 |
| Pullover | 89.65 |
| Dress | 93.00 |
| Sneaker | 96.61 |
| Ankle boot | 96.80 |
| Bag | 98.14 |
| Sandal | 98.34 |
| Trouser | 98.82 |
