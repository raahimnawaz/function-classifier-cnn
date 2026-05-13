"""Synthetic function dataset: 2D curves and 3D surfaces rendered as 128x128 grayscale images.

Each sample carries:
  - the rasterised image (1x128x128)
  - a class label (one of 16 function types)
  - a 9-dim vector of binary structural properties (periodic, monotone, ...).
"""
from __future__ import annotations

import random
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ── Taxonomy ─────────────────────────────────────────────────────────────────

FUNCTION_TYPES_2D = [
    "linear", "quadratic", "cubic", "sine", "cosine",
    "exponential", "logarithmic", "absolute",
]

FUNCTION_TYPES_3D = [
    "paraboloid", "saddle", "sine_surface", "gaussian",
    "ripple", "cone", "hyperboloid", "spiral_surface",
]

FUNCTION_TYPES = FUNCTION_TYPES_2D + FUNCTION_TYPES_3D
NUM_CLASSES = len(FUNCTION_TYPES)

FEATURE_NAMES = [
    "is_periodic",
    "is_monotone_increasing",
    "is_monotone_decreasing",
    "has_multiple_peaks",
    "is_bounded_above",
    "is_bounded_below",
    "is_3d",
    "is_symmetric",
    "has_saddle_point",
]
NUM_FEATURES = len(FEATURE_NAMES)

IMG_SIZE = 128


# ── 2D generation ────────────────────────────────────────────────────────────

def generate_function_2d(func_type: str, x: np.ndarray):
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    if func_type == "linear":
        a = random.uniform(-3, 3)
        b = random.uniform(-5, 5)
        y = a * x + b
        if a > 0.1:   features[1] = 1
        elif a < -0.1: features[2] = 1

    elif func_type == "quadratic":
        a = random.choice([-1, 1]) * random.uniform(0.5, 3)
        b = random.uniform(-3, 3)
        c = random.uniform(-5, 5)
        y = a * x ** 2 + b * x + c
        features[5 if a > 0 else 4] = 1
        features[7] = 1

    elif func_type == "cubic":
        a = random.choice([-1, 1]) * random.uniform(0.1, 1)
        b = random.uniform(-2, 2)
        c = random.uniform(-3, 3)
        d = random.uniform(-5, 5)
        y = a * x ** 3 + b * x ** 2 + c * x + d
        features[3] = 1

    elif func_type == "sine":
        a = random.uniform(1, 4); b = random.uniform(0.5, 3); c = random.uniform(0, 2 * np.pi)
        y = a * np.sin(b * x + c)
        features[[0, 3, 4, 5]] = 1

    elif func_type == "cosine":
        a = random.uniform(1, 4); b = random.uniform(0.5, 3); c = random.uniform(0, 2 * np.pi)
        y = a * np.cos(b * x + c)
        features[[0, 3, 4, 5]] = 1

    elif func_type == "exponential":
        a = random.uniform(0.5, 2)
        b = random.choice([-1, 1]) * random.uniform(0.2, 1)
        y = a * np.exp(b * x)
        features[1 if b > 0 else 2] = 1
        features[5] = 1

    elif func_type == "logarithmic":
        a = random.choice([-1, 1]) * random.uniform(0.5, 3)
        b = random.uniform(-3, 3)
        y = a * np.log(np.abs(x) + 1) + b
        features[1 if a > 0 else 2] = 1

    elif func_type == "absolute":
        a = random.choice([-1, 1]) * random.uniform(0.5, 3)
        b = random.uniform(-3, 3)
        c = random.uniform(-5, 5)
        y = a * np.abs(x + b) + c
        features[5 if a > 0 else 4] = 1
        features[7] = 1
    else:
        raise ValueError(f"Unknown 2D function type: {func_type}")

    y = np.clip(y, -15, 15)
    return y.astype(np.float32), features


# ── 3D generation ────────────────────────────────────────────────────────────

def generate_function_3d(func_type: str, X: np.ndarray, Y: np.ndarray):
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    features[6] = 1  # is_3d

    if func_type == "paraboloid":
        a = random.choice([-1, 1]) * random.uniform(0.3, 2)
        b = random.choice([-1, 1]) * random.uniform(0.3, 2)
        cx = random.uniform(-1, 1); cy = random.uniform(-1, 1)
        Z = a * (X - cx) ** 2 + b * (Y - cy) ** 2
        if a > 0 and b > 0: features[[5, 7]] = 1
        elif a < 0 and b < 0: features[[4, 7]] = 1

    elif func_type == "saddle":
        a = random.uniform(0.3, 2); b = random.uniform(0.3, 2)
        Z = a * X ** 2 - b * Y ** 2
        features[[7, 8]] = 1

    elif func_type == "sine_surface":
        a = random.uniform(1, 3); bx = random.uniform(0.5, 2); by = random.uniform(0.5, 2)
        Z = a * np.sin(bx * X) * np.cos(by * Y)
        features[[0, 3, 4, 5, 7]] = 1

    elif func_type == "gaussian":
        a = random.uniform(1, 4); sx = random.uniform(0.5, 2); sy = random.uniform(0.5, 2)
        cx = random.uniform(-1, 1); cy = random.uniform(-1, 1)
        Z = a * np.exp(-((X - cx) ** 2 / (2 * sx ** 2) + (Y - cy) ** 2 / (2 * sy ** 2)))
        features[[4, 5, 7]] = 1

    elif func_type == "ripple":
        a = random.uniform(1, 3); freq = random.uniform(1, 3)
        R = np.sqrt(X ** 2 + Y ** 2) + 1e-6
        Z = a * np.sin(freq * R) / R
        features[[0, 3, 4, 5, 7]] = 1

    elif func_type == "cone":
        a = random.choice([-1, 1]) * random.uniform(0.5, 2)
        cx = random.uniform(-1, 1); cy = random.uniform(-1, 1)
        Z = a * np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        features[1 if a > 0 else 2] = 1
        features[7] = 1

    elif func_type == "hyperboloid":
        a = random.uniform(0.3, 1.5); b = random.uniform(0.3, 1.5); c = random.uniform(0.5, 2)
        Z = c * np.sqrt(1 + (X / a) ** 2 + (Y / b) ** 2)
        features[[1, 5, 7]] = 1

    elif func_type == "spiral_surface":
        R = np.sqrt(X ** 2 + Y ** 2) + 1e-6
        theta = np.arctan2(Y, X)
        a = random.uniform(0.5, 2); freq = random.uniform(1, 3)
        Z = a * np.sin(freq * R + theta)
        features[[0, 3, 4, 5]] = 1
    else:
        raise ValueError(f"Unknown 3D function type: {func_type}")

    Z = np.clip(Z, -15, 15)
    return Z.astype(np.float32), features


# ── Rasterisation ────────────────────────────────────────────────────────────

def plot_to_image_2d(x: np.ndarray, y: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(2, 2), dpi=size // 2)
    ax.plot(x, y, "b-", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(-12, 12); ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    return np.array(Image.open(buf).convert("L").resize((size, size)), dtype=np.float32) / 255.0


def plot_to_image_3d(X, Y, Z, size: int = IMG_SIZE, elev=None, azim=None) -> np.ndarray:
    if elev is None: elev = random.uniform(20, 50)
    if azim is None: azim = random.uniform(20, 340)
    fig = plt.figure(figsize=(2, 2), dpi=size // 2)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9,
                    rstride=2, cstride=2, linewidth=0, antialiased=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
    z_range = max(abs(Z.min()), abs(Z.max()), 1)
    ax.set_zlim(-z_range, z_range); ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig); buf.seek(0)
    return np.array(Image.open(buf).convert("L").resize((size, size)), dtype=np.float32) / 255.0


# ── Dataset ──────────────────────────────────────────────────────────────────

class FunctionDataset(Dataset):
    """Generates `n` synthetic samples by cycling through FUNCTION_TYPES."""

    def __init__(self, n: int, img_size: int = IMG_SIZE, verbose: bool = True):
        x_1d = np.linspace(-5, 5, 200)
        grid = np.linspace(-3, 3, 60)
        X, Y = np.meshgrid(grid, grid)

        if verbose:
            print(f"Generating {n} samples...")
        imgs, labels, feats = [], [], []
        for i in range(n):
            if verbose and i and i % 1000 == 0:
                print(f"  {i}/{n}")
            ft = FUNCTION_TYPES[i % NUM_CLASSES]
            label_idx = FUNCTION_TYPES.index(ft)

            if ft in FUNCTION_TYPES_2D:
                y, f = generate_function_2d(ft, x_1d)
                img = plot_to_image_2d(x_1d, y, img_size)
            else:
                Z, f = generate_function_3d(ft, X, Y)
                img = plot_to_image_3d(X, Y, Z, img_size)

            imgs.append(img); labels.append(label_idx); feats.append(f)

        self.imgs = np.array(imgs)
        self.labels = np.array(labels, dtype=np.int64)
        self.feats = np.array(feats, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.imgs[i]).unsqueeze(0),
            torch.tensor(self.labels[i]),
            torch.tensor(self.feats[i]),
        )
