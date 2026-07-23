"""Training loop and evaluation for FunctionCNN / BaselineCNN.

Run directly:
    python -m src.train                  # train the full model
    python -m src.train --baseline       # train the 1-layer baseline
    python -m src.train --compare        # train both and print a side-by-side
"""
from __future__ import annotations

import argparse
import json
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
    FEATURE_NAMES,
    FUNCTION_TYPES,
    FUNCTION_TYPES_2D,
    FUNCTION_TYPES_3D,
    NUM_CLASSES,
    NUM_FEATURES,
    FunctionDataset,
    generate_function_2d,
    generate_function_3d,
    plot_to_image_2d,
    plot_to_image_3d,
)
from .model import BaselineCNN, FunctionCNN

# ── Hyperparameters ──────────────────────────────────────────────────────────

BATCH_SIZE  = 64
NUM_EPOCHS  = 30
NUM_TRAIN   = 8000
NUM_VAL     = 1600
LR          = 5e-4
SEED        = 42
MODEL_PATH  = "function_cnn.pth"
BASELINE_PATH = "baseline_cnn.pth"
FIG_DIR     = "figures"
METRICS_PATH = "metrics.json"
NUM_WORKERS = min(os.cpu_count() or 0, 8)


def _amp_supported() -> bool:
    """AMP (fp16) is only a speed-up on Volta+ (compute capability >= 7.0).

    On older GPUs (e.g. Maxwell GTX 9xx) fp16 throughput is a fraction of fp32,
    so autocast makes training *slower*. Gate it on tensor-core-capable cards.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 7


USE_AMP = _amp_supported()


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG that affects data generation and training.

    The dataset is materialised eagerly in the main process, so seeding here
    makes both the sampled functions and the optimisation trajectory
    reproducible run to run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        def _step(imgs=imgs, labels=labels, feats=feats):
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


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Run the model over `loader` and return a full metrics dict.

    Reports 16-class accuracy (overall + 2D/3D split + per class), the 16x16
    confusion matrix, and multi-label property-detection metrics (per-property
    F1, macro-F1, exact-match ratio, Hamming accuracy). Pure NumPy — no sklearn.
    """
    model.eval()
    all_true, all_pred = [], []
    prop_true, prop_pred = [], []
    for imgs, labels, feats in loader:
        imgs = imgs.to(device)
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits, pfeats, _ = model(imgs)
        else:
            logits, pfeats, _ = model(imgs)
        all_true.append(labels.numpy())
        all_pred.append(logits.argmax(1).cpu().numpy())
        prop_true.append(feats.numpy())
        prop_pred.append((pfeats.float().cpu().numpy() > 0.5).astype(np.float32))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    p_true = np.concatenate(prop_true)
    p_pred = np.concatenate(prop_pred)

    # ── Classification ──
    overall = float((y_true == y_pred).mean())
    is_3d = y_true >= len(FUNCTION_TYPES_2D)
    acc_2d = float((y_true[~is_3d] == y_pred[~is_3d]).mean())
    acc_3d = float((y_true[is_3d] == y_pred[is_3d]).mean())

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {
        FUNCTION_TYPES[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0)
        for i in range(NUM_CLASSES)
    }

    # ── Property detection (multi-label) ──
    tp = (p_pred * p_true).sum(0)
    fp = (p_pred * (1 - p_true)).sum(0)
    fn = ((1 - p_pred) * p_true).sum(0)
    prec = tp / np.clip(tp + fp, 1e-9, None)
    rec = tp / np.clip(tp + fn, 1e-9, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    prop_f1 = {FEATURE_NAMES[i]: float(f1[i]) for i in range(NUM_FEATURES)}
    prop_acc = {FEATURE_NAMES[i]: float((p_pred[:, i] == p_true[:, i]).mean())
                for i in range(NUM_FEATURES)}

    return {
        "cls_accuracy": overall,
        "cls_accuracy_2d": acc_2d,
        "cls_accuracy_3d": acc_3d,
        "per_class_accuracy": per_class,
        "prop_macro_f1": float(f1.mean()),
        "prop_hamming_accuracy": float((p_pred == p_true).mean()),
        "prop_exact_match": float((p_pred == p_true).all(1).mean()),
        "prop_f1": prop_f1,
        "prop_accuracy": prop_acc,
        "confusion_matrix": cm.tolist(),
    }


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


def save_confusion_matrix(cm, path="confusion_matrix.png"):
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(1, keepdims=True)
    norm = cm / np.clip(row_sums, 1, None)  # row-normalised (recall per class)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(FUNCTION_TYPES, rotation=90, fontsize=7)
    ax.set_yticklabels(FUNCTION_TYPES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if norm[i, j] > 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
    print(f"Saved {path}")


def save_property_f1(prop_f1, path="property_f1.png"):
    names = list(prop_f1.keys())
    vals = [prop_f1[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, vals, color="#4c72b0")
    ax.set_xlim(0, 1); ax.set_xlabel("F1 score")
    ax.set_title("Per-property detection F1")
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(min(v + 0.01, 0.95), b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
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
            pass
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


def _fig(name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def report_full_model(model, history, val_set, val_loader, device, extra=None):
    """Save figures, compute full metrics, and write metrics.json."""
    save_training_curves(history, _fig("training_curves.png"))
    save_predictions(model, val_set, device, _fig("predictions.png"))
    metrics = evaluate_full(model, val_loader, device)
    save_confusion_matrix(metrics["confusion_matrix"], _fig("confusion_matrix.png"))
    save_property_f1(metrics["prop_f1"], _fig("property_f1.png"))

    record = {
        "hardware": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                     else "CPU"),
        "amp": USE_AMP,
        "epochs": len(history["loss"]),
        "num_train": NUM_TRAIN,
        "num_val": NUM_VAL,
        "seed": SEED,
        **(extra or {}),
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"],
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"Saved {METRICS_PATH}")

    print("\n" + "=" * 52)
    print("  FULL-MODEL METRICS (validation)")
    print("=" * 52)
    print(f"  16-class accuracy        : {metrics['cls_accuracy']*100:.1f}%")
    print(f"    2D curves              : {metrics['cls_accuracy_2d']*100:.1f}%")
    print(f"    3D surfaces            : {metrics['cls_accuracy_3d']*100:.1f}%")
    print(f"  Property macro-F1        : {metrics['prop_macro_f1']:.3f}")
    print(f"  Property Hamming acc     : {metrics['prop_hamming_accuracy']*100:.1f}%")
    print(f"  Property exact-match     : {metrics['prop_exact_match']*100:.1f}%")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Train the 1-layer baseline instead of the full model.")
    parser.add_argument("--compare", action="store_true",
                        help="Train both and print a side-by-side comparison.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="RNG seed for reproducible data + training.")
    args = parser.parse_args()

    set_seed(args.seed)
    loaders = build_loaders()
    _, val_set, _, val_loader = loaders

    if args.compare:
        _, _, base_best, _ = train_model(BaselineCNN, BASELINE_PATH,
                                         args.epochs, loaders, "baseline")
        model, history, full_best, _ = train_model(FunctionCNN, MODEL_PATH,
                                                   args.epochs, loaders, "full")
        device = next(model.parameters()).device
        report_full_model(model, history, val_set, val_loader, device, extra={
            "baseline_accuracy": base_best,
            "full_accuracy": full_best,
        })
        print("\n=== Baseline vs Full ===")
        print(f"  Baseline (1-conv layer)        : {base_best*100:.1f}%")
        print(f"  Full ResNet+SE+multi-task      : {full_best*100:.1f}%")
        print(f"  Gain                           : +{(full_best - base_best)*100:.1f} pp")
    elif args.baseline:
        train_model(BaselineCNN, BASELINE_PATH, args.epochs, loaders, "baseline")
    else:
        model, history, full_best, _ = train_model(
            FunctionCNN, MODEL_PATH, args.epochs, loaders, "full")
        device = next(model.parameters()).device
        report_full_model(model, history, val_set, val_loader, device,
                          extra={"full_accuracy": full_best})


if __name__ == "__main__":
    main()
