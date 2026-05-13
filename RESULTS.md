# Results

Numbers reported here are tied to a specific run so anyone can reproduce
them. Update this file when the model or training schedule changes.

## Reference run

| Field                | Value                                       |
| -------------------- | ------------------------------------------- |
| Commit               | _fill in once committed_                    |
| Date                 | 2026-05-12                                  |
| Hardware             | _e.g. NVIDIA RTX 3060 12 GB, AMD Ryzen 7_   |
| PyTorch              | 2.x, mixed precision (AMP) on CUDA          |
| Train / val samples  | 8000 / 1600 (`NUM_TRAIN` / `NUM_VAL`)       |
| Epochs               | 30 (full) / 10 (baseline)                   |
| Batch size           | 64                                          |
| Optimizer            | AdamW, lr=5e-4, weight_decay=1e-3           |
| LR schedule          | CosineAnnealingLR                           |
| Seed                 | _not pinned in current code_                |

## Headline numbers

| Model                                       | Params | Best val acc |
| ------------------------------------------- | ------ | ------------ |
| Baseline (1 conv + linear)                  | ~17 K  | _fill in_    |
| FunctionCNN (ResNet + SE + multi-task)      | ~10 M  | _fill in_    |
| Gain                                        | —      | _fill in pp_ |

Reproduce with:

```bash
python -m src.train --compare
```

The script prints the per-epoch log and final `Baseline vs Full` table to
stdout. Paste the last ~5 lines of that log under "Training log" below.

## Training log

```
(paste the tail of `python -m src.train --compare` output here)
```

## Notes

- The dataset is fully synthetic and regenerated on every run; numbers will
  drift by ~1 pp between runs unless a seed is pinned.
- Mixed precision is automatic on CUDA, off on CPU. CPU runs are ~10x slower
  but produce comparable accuracy.
- Reconstruction loss is used as a regulariser, not evaluated directly. If
  you want to report it, add an MSE metric in `evaluate()`.
