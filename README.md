# Function CNN

A multi-task convolutional network that looks at a rendered plot of a function
and tells you **what function it is**, **what structural properties it has**
(periodic, monotone, bounded, symmetric, ...), and **reconstructs the plot**
as an auxiliary self-supervised task.

The model handles both **2D curves** (sine, polynomial, exponential, log, ...)
and **3D surfaces** (paraboloid, saddle, ripple, gaussian, ...) — 16 function
families in total — from a single 128x128 grayscale image.

---

## Problem framing

Mathematicians and engineers spend a lot of time staring at plots and asking
*"what kind of function is this?"* The shape of the curve usually gives it
away — but that intuition is hard to write down as rules. This project trains
a CNN to learn that intuition from synthetic examples.

Given a plot, the network simultaneously answers three questions:

1. **Classification** — which of 16 function families does this belong to?
2. **Property detection** — does it look periodic? monotone? symmetric? does
   it have a saddle point?
3. **Reconstruction** — can we re-draw the plot from the network's internal
   representation? (Acts as a regulariser; forces features to retain shape
   information, not just class identity.)

All three heads share a single ResNet-style backbone with squeeze-and-excitation
attention.

---

## Architecture

```
                input  (1 x 128 x 128)
                       │
                       ▼
          ┌─────────────────────────────┐
          │   Backbone (5 residual      │  128 -> 64 -> 32 -> 16 -> 8 -> 4
          │   blocks, BN + SE attention)│
          └─────────────┬───────────────┘
                        │
                AdaptiveAvgPool2d(1)
                        │
                        ▼
          ┌─────────────────────────────┐
          │     Shared FC trunk         │  512 → 512 → 256
          └─┬───────────┬───────────┬───┘
            │           │           │
            ▼           ▼           ▼
       ┌────────┐  ┌──────────┐ ┌──────────┐
       │Classify│  │ Detector │ │ Decoder  │
       │ 16-way │  │ 9 sigm.  │ │ recon img│
       └────────┘  └──────────┘ └──────────┘
```

Each `ResConvBlock`:

```
  x ──► Conv3x3 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─┐
   │                                        ▼
   └──► 1x1 skip ──────────────────────► (+) ─ SE attention ─ ReLU ─ MaxPool ─ Dropout
```

Loss = `CE(class) + 0.5 · BCE(properties) + 0.3 · MSE(reconstruction)`.

Training uses AdamW, cosine LR schedule, and mixed precision (`torch.amp`) when
a GPU is available.

---

## Training curves

![training curves](training_curves.png)

Loss falls smoothly and validation accuracy tracks training accuracy — the
multi-task losses and dropout keep the network from overfitting on
synthetically generated data.

---

## Sample predictions

![predictions](predictions.png)

Three rows per column: original plot, predicted class (green = correct,
red = wrong), and the decoder's reconstruction from the shared latent.

---

## Baseline comparison

The notebook (`function_cnn.ipynb`, "Baseline comparison" section) trains a
deliberately weak **1-conv-layer CNN** under the *same* data pipeline and
training schedule, so you can see what the architecture is actually buying:

| Model                                 | Params      | Val accuracy (16-class) |
| ------------------------------------- | ----------- | ----------------------- |
| Baseline (1 conv + linear)            | ~17 K       | see [RESULTS.md](RESULTS.md) |
| FunctionCNN (ResNet + SE + multi-task)| ~10 M       | see [RESULTS.md](RESULTS.md) |

Exact numbers, hardware, and seed are pinned in [RESULTS.md](RESULTS.md).
Reproduce with `python -m src.train --compare`.

---

## Why these techniques matter

- **Residual blocks** — without skip connections, a 10-layer-deep CNN over
  small (128 px) plot images is hard to optimise; the residual path keeps
  gradients flowing and lets the deeper blocks specialise without harming the
  early features.
- **Squeeze-and-Excitation attention** — different function families care
  about different channel activations (a wave's repeating ridges vs. a
  parabola's smooth bowl). SE blocks let the network reweight channels per
  example *for free*, with negligible parameter cost.
- **Multi-task learning (classifier + detector + decoder)** — the property
  detector forces the latent to encode *interpretable* shape information
  (monotone, periodic, ...). The decoder forces it to retain *geometric*
  information. Both reduce overfitting and make a single class label go
  further as a supervision signal.
- **Mixed precision (AMP)** — roughly 2× training throughput on modern NVIDIA
  GPUs with no accuracy loss, because BatchNorm + residual paths absorb the
  fp16 numeric noise well.
- **Cosine LR schedule + AdamW + weight decay** — robust defaults that don't
  need per-experiment tuning; cosine annealing prevents the high-LR
  overshoot that hurts the last few percent of accuracy.
- **Synthetic data generation** — every sample is a fresh randomised
  instance, so the dataset is effectively infinite and the model never sees
  the same plot twice. The cost is paid up-front once.

---

## Repo layout

```
.
├── README.md
├── RESULTS.md                # pinned numbers, hardware, training log
├── LICENSE                   # MIT
├── pyproject.toml            # `pip install -e ".[dev,demo]"`
├── requirements.txt          # runtime deps (torch, numpy, matplotlib, PIL, gradio)
├── requirements-dev.txt      # + jupyter, ruff, black, pytest, mypy
├── .github/workflows/ci.yml  # ruff + pytest on push/PR
├── function_cnn.py           # back-compat shim → imports from src/
├── function_cnn.ipynb        # notebook / demo
├── training_curves.png       # generated by `python -m src.train`
├── predictions.png           # generated by `python -m src.train`
├── gradio_demo.py            # draw-a-curve web demo
├── src/
│   ├── __init__.py
│   ├── data.py               # function generators + FunctionDataset
│   ├── model.py              # FunctionCNN, BaselineCNN, SE/Res blocks
│   └── train.py              # training loop, eval, CLI (--baseline / --compare)
└── tests/
    ├── test_dataset.py       # generator / dataset shapes and ranges
    └── test_model.py         # forward-pass shapes, param-count sanity

Trained weights (`function_cnn.pth`) are not checked in — regenerate via
`python -m src.train`.
```

---

## Quick start

```bash
# install (editable, with dev + demo extras)
pip install -e ".[dev,demo]"
# or, plain runtime only:
#   pip install -r requirements.txt

# run tests
pytest

# train the full model (writes function_cnn.pth, training_curves.png, predictions.png)
python -m src.train

# train just the baseline
python -m src.train --baseline

# train both and print the comparison
python -m src.train --compare

# spin up the draw-a-curve demo
python gradio_demo.py
```

The original `function_cnn.py` and `function_cnn.ipynb` are kept as a
single-file reference and an interactive demo, respectively. New work should
go in `src/`.
