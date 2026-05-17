# Results

Numbers reported here are tied to a specific commit so anyone can reproduce them exactly. Update this file whenever the model architecture or training schedule changes.

---

## Reference Run

| Field | Value |
|-------|-------|
| Commit | _(fill in after training)_ |
| Date | 2026-05-12 |
| Hardware | _(e.g. NVIDIA RTX 3060 12 GB / AMD Ryzen 7 5800X)_ |
| PyTorch | 2.x, mixed precision (AMP) on CUDA |
| Training samples | 8,000 |
| Validation samples | 1,600 |
| Epochs | 30 (full model) / 30 (baseline) |
| Batch size | 64 |
| Optimiser | AdamW — lr=5e-4, weight\_decay=1e-3 |
| LR schedule | CosineAnnealingLR |
| Random seed | _(not pinned; see note below)_ |

---

## Benchmark

| Model | Parameters | Best val accuracy (16-class) |
|-------|-----------|------------------------------|
| Baseline (1 conv + linear) | ~17 K | _(fill in)_ |
| FunctionCNN (ResNet + SE + multi-task) | ~10 M | _(fill in)_ |
| Gain | — | _(fill in pp)_ |

Reproduce with:

```bash
python -m src.train --compare
```

The script prints a per-epoch log followed by a `=== Baseline vs Full ===` summary table. Paste the last few lines of that output in the **Training Log** section below.

---

## Training Log

```
(paste the tail of `python -m src.train --compare` here)
```

---

## Notes

- The dataset is fully synthetic and regenerated on every run. Accuracy will drift by ~1 pp between runs unless a fixed seed is set in `src/train.py`.
- Mixed precision is enabled automatically on CUDA and disabled on CPU. CPU runs are roughly 10× slower but produce comparable final accuracy.
- The reconstruction loss (`MSE`) serves as a regulariser only and is not reported as a standalone metric. To evaluate it directly, add an MSE metric to the `evaluate()` function in `src/train.py`.
