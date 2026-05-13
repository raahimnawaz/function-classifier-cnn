"""Training loop and evaluation for FunctionCNN / BaselineCNN.

Run directly:
    python -m src.train                  # train the full model
    python -m src.train --baseline       # train the 1-layer baseline
    python -m src.train --compare        # train both and print a side-by-side
"""
from __future__ import annotations

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import (
    FUNCTION_TYPES, FUNCTION_TYPES_3D, FEATURE_NAMES,
    FunctionDataset, generate_function_2d, generate_function_3d,
    plot_to_image_2d, plot_to_image_3d,
)
from .model import BaselineCNN, FunctionCNN

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE  = 64
NUM_EPOCHS  = 30
NUM_TRAIN   = 8000
NUM_VAL     = 1600
LR          = 5e-4
MODEL_PATH  = "function_cnn.pth"
BASELINE_PATH = "baseline_cnn.pth"
NUM_WORKERS = min(os.cpu_count() or 0, 8)
USE_AMP     = torch.cuda.is_available()


# ── Train / eval loops ───────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, scaler=None,
              recon_weight: float = 0.3, feat_weight: float = 0.5):
    model.train()
    cls_loss_fn = nn.CrossEntropyLoss()
    feat_loss_fn = nn.BCELoss()
    recon_loss_fn = nn.MSELoss()
    total_loss = correct = total = 0

    for imgs, labels, feats in loader:
        imgs = imgs.to(device); labels = labels.to(device); feats = feats.to(device)
        optimizer.zero_grad(set_to_none=True)

        def _step():
            logits, pred_feats, recon = model(imgs)
            loss = cls_loss_fn(logits, labels) \
                 + feat_weight * feat_loss_fn(pred_feats, feats) \
                 + recon_weight * recon_loss_fn(recon, imgs)
            return logits, loss

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits, loss = _step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = _step()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits, _, _ = model(imgs)
        else:
            logits, _, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ── Visualisation ────────────────────────────────────────────────────────────

def save_training_curves(history, path="training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["loss"]); ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()
    print(f"Saved {path}")


def save_predictions(model, dataset, device, path="predictions.png", n=8):
    model.eval()
    idx = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    with torch.no_grad():
        for col, i in enumerate(idx):
            img, lbl, _ = dataset[i]
            inp = img.unsqueeze(0).to(device)
            if USE_AMP:
                with torch.amp.autocast("cuda"):
                    logits, _, recon = model(inp)
            else:
                logits, _, recon = model(inp)
            pred = logits.argmax(1).item()
            color = "green" if pred == lbl.item() else "red"
            axes[0, col].imshow(img[0], cmap="gray")
            axes[0, col].set_title(FUNCTION_TYPES[lbl.item()][:6], fontsize=7)
            axes[0, col].axis("off")
            axes[1, col].imshow(img[0], cmap="gray")
            axes[1, col].set_title(FUNCTION_TYPES[pred][:6], fontsize=7, color=color)
            axes[1, col].axis("off")
            axes[2, col].imshow(recon.cpu()[0, 0], cmap="gray")
            axes[2, col].set_title("recon", fontsize=7)
            axes[2, col].axis("off")
    for row, lbl in enumerate(["Original", "Predicted", "Reconstructed"]):
        axes[row, 0].set_ylabel(lbl, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()
    print(f"Saved {path}")


# ── Inference helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def analyze(model, device, func_type: str | None = None, x=None, y=None,
            X=None, Y=None, Z=None, verbose: bool = True):
    model.eval()

    if func_type in FUNCTION_TYPES_3D:
        if X is None:
            grid = np.linspace(-3, 3, 60); X, Y = np.meshgrid(grid, grid)
        if Z is None:
            Z, true_feats = generate_function_3d(func_type, X, Y)
        else:
            true_feats = None
        img = plot_to_image_3d(X, Y, Z)
    else:
        if x is None:
            x = np.linspace(-5, 5, 200)
        if y is None:
            y, true_feats = generate_function_2d(func_type, x)
        else:
            true_feats = None
        img = plot_to_image_2d(x, y)

    t = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    if USE_AMP:
        with torch.amp.autocast("cuda"):
            logits, pred_feats, _ = model(t)
    else:
        logits, pred_feats, _ = model(t)
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    pred_feat = pred_feats.cpu().numpy()[0]
    pred_cls = probs.argmax()

    if verbose:
        dim_label = "3D" if func_type in FUNCTION_TYPES_3D else "2D"
        print("\n" + "=" * 52)
        print(f"  FUNCTION ANALYSIS ({dim_label})")
        print("=" * 52)
        if func_type:
            print(f"  True type : {func_type}")
        print(f"  Predicted : {FUNCTION_TYPES[pred_cls]}  ({probs[pred_cls]*100:.1f}% confidence)")
        print("\n  Top-3 predictions:")
        for i in probs.argsort()[::-1][:3]:
            print(f"    {FUNCTION_TYPES[i]:20s}  {probs[i]*100:.1f}%")
        print("\n  Detected properties:")
        for name, val in zip(FEATURE_NAMES, pred_feat):
            marker = "Y" if val > 0.5 else "."
            print(f"    {marker} {name:30s} ({val:.2f})")
    return probs, pred_feat


# ── Top-level training entry points ──────────────────────────────────────────

def build_loaders(num_train: int = NUM_TRAIN, num_val: int = NUM_VAL):
    train_set = FunctionDataset(num_train)
    val_set = FunctionDataset(num_val)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0)
    return train_set, val_set, train_loader, val_loader


def train_model(model_cls, save_path: str, num_epochs: int = NUM_EPOCHS,
                loaders=None, label: str = "model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{label}] device={device}  amp={USE_AMP}")

    if loaders is None:
        train_set, val_set, train_loader, val_loader = build_loaders()
    else:
        train_set, val_set, train_loader, val_loader = loaders

    model = model_cls().to(device)
    print(f"[{label}] parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_val = 0.0
    history = {"loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, num_epochs + 1):
        loss, tr_acc = run_epoch(model, train_loader, optimizer, device, scaler)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        history["loss"].append(loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
        print(f"  [{label}][{epoch:2d}/{num_epochs}] loss={loss:.4f} "
              f"train={tr_acc*100:.1f}% val={val_acc*100:.1f}%")

    print(f"[{label}] best val acc: {best_val*100:.1f}%")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model, history, best_val, (train_set, val_set, train_loader, val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Train the 1-layer baseline instead of the full model.")
    parser.add_argument("--compare", action="store_true",
                        help="Train both and print a side-by-side comparison.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    loaders = build_loaders()

    if args.compare:
        _, _, _, base_best = train_model(BaselineCNN, BASELINE_PATH,
                                         args.epochs, loaders, "baseline")[:4]
        model, history, full_best, _ = train_model(FunctionCNN, MODEL_PATH,
                                                   args.epochs, loaders, "full")
        save_training_curves(history)
        save_predictions(model, loaders[1], next(model.parameters()).device)
        print("\n=== Baseline vs Full ===")
        print(f"  Baseline (1-conv layer)        : {base_best*100:.1f}%")
        print(f"  Full ResNet+SE+multi-task      : {full_best*100:.1f}%")
        print(f"  Gain                           : +{(full_best - base_best)*100:.1f} pp")
    elif args.baseline:
        train_model(BaselineCNN, BASELINE_PATH, args.epochs, loaders, "baseline")
    else:
        model, history, _, (_, val_set, _, _) = train_model(
            FunctionCNN, MODEL_PATH, args.epochs, loaders, "full")
        save_training_curves(history)
        save_predictions(model, val_set, next(model.parameters()).device)


if __name__ == "__main__":
    main()
