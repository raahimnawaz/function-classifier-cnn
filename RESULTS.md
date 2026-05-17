# Results

A reproducible benchmark template. Run the comparison locally and paste your numbers into the **Reference Run** section below to pin them to a specific commit and hardware configuration.

The dataset is regenerated from scratch each run with no fixed seed, so accuracy will drift by ~1 pp between runs. Pin a seed in `src/train.py` if you need bit-reproducibility.

---

## How to reproduce

```bash
pip install -e ".[dev]"
python -m src.train --compare
```

The script writes a per-epoch log, the `=== Baseline vs Full ===` summary, `training_curves.png`, and `predictions.png`. Paste the tail of the log into the **Training Log** section below.

---

## Reference Run

| Field | Value |
|-------|-------|
| Commit | _(fill in after training)_ |
| Date | _(fill in)_ |
| Hardware | _(e.g. NVIDIA RTX 3060 12 GB / AMD Ryzen 7 5800X)_ |
| PyTorch | 2.x, mixed precision (AMP) on CUDA |
| Training samples | 8,000 |
| Validation samples | 1,600 |
| Epochs | 30 (full model) / 30 (baseline) |
| Batch size | 64 |
| Optimiser | AdamW — lr=5e-4, weight_decay=1e-3 |
| LR schedule | CosineAnnealingLR |
| Random seed | _(not pinned by default; see note above)_ |

---

## Benchmark

| Model | Parameters | Best val accuracy (16-class) |
|-------|-----------|------------------------------|
| Baseline (1 conv + linear) | ~17 K | _(fill in)_ |
| FunctionCNN (ResNet + SE + multi-task) | ~10 M | _(fill in)_ |
| Gain | — | _(fill in pp)_ |

---

## Training Log

```
(paste the tail of `python -m src.train --compare` here)
```

---

## Notes

- Mixed precision is enabled automatically on CUDA and disabled on CPU. CPU runs are roughly 10× slower but produce comparable final accuracy.
- The reconstruction loss (`MSE`) serves as a regulariser only and is not reported as a standalone metric. To evaluate it directly, add an MSE term to `evaluate()` in `src/train.py`.
- `training_curves.png` and `predictions.png` at the repo root are committed sample outputs from a prior run; running `--compare` will overwrite them with your run's results.
