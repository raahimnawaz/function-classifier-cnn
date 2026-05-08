import os
import random
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── Constants ────────────────────────────────────────────────────────────────

FUNCTION_TYPES_2D = [
    'linear', 'quadratic', 'cubic', 'sine', 'cosine',
    'exponential', 'logarithmic', 'absolute',
]

FUNCTION_TYPES_3D = [
    'paraboloid', 'saddle', 'sine_surface', 'gaussian',
    'ripple', 'cone', 'hyperboloid', 'spiral_surface',
]

FUNCTION_TYPES = FUNCTION_TYPES_2D + FUNCTION_TYPES_3D
NUM_CLASSES = len(FUNCTION_TYPES)

FEATURE_NAMES = [
    'is_periodic',
    'is_monotone_increasing',
    'is_monotone_decreasing',
    'has_multiple_peaks',
    'is_bounded_above',
    'is_bounded_below',
    'is_3d',
    'is_symmetric',
    'has_saddle_point',
]
NUM_FEATURES = len(FEATURE_NAMES)

IMG_SIZE    = 128
BATCH_SIZE  = 64
NUM_EPOCHS  = 30
NUM_TRAIN   = 8000
NUM_VAL     = 1600
LR          = 5e-4
MODEL_PATH  = 'function_cnn.pth'
NUM_WORKERS = min(os.cpu_count() or 0, 8)

USE_AMP = torch.cuda.is_available()

# ── 2D Data generation ──────────────────────────────────────────────────────

def generate_function_2d(func_type, x):
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    if func_type == 'linear':
        a = random.uniform(-3, 3)
        b = random.uniform(-5, 5)
        y = a * x + b
        if a > 0.1:   features[1] = 1
        elif a < -0.1: features[2] = 1

    elif func_type == 'quadratic':
        a = random.choice([-1, 1]) * random.uniform(0.5, 3)
        b = random.uniform(-3, 3)
        c = random.uniform(-5, 5)
        y = a * x**2 + b * x + c
        if a > 0: features[5] = 1
        else:     features[4] = 1
        features[7] = 1  # symmetric about vertex

    elif func_type == 'cubic':
        a = random.choice([-1, 1]) * random.uniform(0.1, 1)
        b = random.uniform(-2, 2)
        c = random.uniform(-3, 3)
        d = random.uniform(-5, 5)
        y = a * x**3 + b * x**2 + c * x + d
        features[3] = 1

    elif func_type == 'sine':
        a = random.uniform(1, 4)
        b = random.uniform(0.5, 3)
        c = random.uniform(0, 2 * np.pi)
        y = a * np.sin(b * x + c)
        features[[0, 3, 4, 5]] = 1

    elif func_type == 'cosine':
        a = random.uniform(1, 4)
        b = random.uniform(0.5, 3)
        c = random.uniform(0, 2 * np.pi)
        y = a * np.cos(b * x + c)
        features[[0, 3, 4, 5]] = 1

    elif func_type == 'exponential':
        a = random.uniform(0.5, 2)
        b = random.choice([-1, 1]) * random.uniform(0.2, 1)
        y = a * np.exp(b * x)
        if b > 0: features[1] = 1
        else:     features[2] = 1
        features[5] = 1

    elif func_type == 'logarithmic':
        a = random.choice([-1, 1]) * random.uniform(0.5, 3)
        b = random.uniform(-3, 3)
        y = a * np.log(np.abs(x) + 1) + b
        if a > 0: features[1] = 1
        else:     features[2] = 1

    elif func_type == 'absolute':
        a = random.choice([-1, 1]) * random.uniform(0.5, 3)
        b = random.uniform(-3, 3)
        c = random.uniform(-5, 5)
        y = a * np.abs(x + b) + c
        if a > 0: features[5] = 1
        else:     features[4] = 1
        features[7] = 1

    y = np.clip(y, -15, 15)
    return y.astype(np.float32), features


# ── 3D Data generation ──────────────────────────────────────────────────────

def generate_function_3d(func_type, X, Y):
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    features[6] = 1  # is_3d

    if func_type == 'paraboloid':
        a = random.choice([-1, 1]) * random.uniform(0.3, 2)
        b = random.choice([-1, 1]) * random.uniform(0.3, 2)
        cx = random.uniform(-1, 1)
        cy = random.uniform(-1, 1)
        Z = a * (X - cx)**2 + b * (Y - cy)**2
        if a > 0 and b > 0:
            features[5] = 1
            features[7] = 1
        elif a < 0 and b < 0:
            features[4] = 1
            features[7] = 1

    elif func_type == 'saddle':
        a = random.uniform(0.3, 2)
        b = random.uniform(0.3, 2)
        Z = a * X**2 - b * Y**2
        features[8] = 1
        features[7] = 1

    elif func_type == 'sine_surface':
        a = random.uniform(1, 3)
        bx = random.uniform(0.5, 2)
        by = random.uniform(0.5, 2)
        Z = a * np.sin(bx * X) * np.cos(by * Y)
        features[[0, 3, 4, 5, 7]] = 1

    elif func_type == 'gaussian':
        a = random.uniform(1, 4)
        sx = random.uniform(0.5, 2)
        sy = random.uniform(0.5, 2)
        cx = random.uniform(-1, 1)
        cy = random.uniform(-1, 1)
        Z = a * np.exp(-((X - cx)**2 / (2 * sx**2) + (Y - cy)**2 / (2 * sy**2)))
        features[[4, 5, 7]] = 1

    elif func_type == 'ripple':
        a = random.uniform(1, 3)
        freq = random.uniform(1, 3)
        R = np.sqrt(X**2 + Y**2) + 1e-6
        Z = a * np.sin(freq * R) / R
        features[[0, 3, 4, 5, 7]] = 1

    elif func_type == 'cone':
        a = random.choice([-1, 1]) * random.uniform(0.5, 2)
        cx = random.uniform(-1, 1)
        cy = random.uniform(-1, 1)
        Z = a * np.sqrt((X - cx)**2 + (Y - cy)**2)
        if a > 0: features[1] = 1
        else:     features[2] = 1
        features[7] = 1

    elif func_type == 'hyperboloid':
        a = random.uniform(0.3, 1.5)
        b = random.uniform(0.3, 1.5)
        c = random.uniform(0.5, 2)
        Z = c * np.sqrt(1 + (X / a)**2 + (Y / b)**2)
        features[[1, 5, 7]] = 1

    elif func_type == 'spiral_surface':
        R = np.sqrt(X**2 + Y**2) + 1e-6
        theta = np.arctan2(Y, X)
        a = random.uniform(0.5, 2)
        freq = random.uniform(1, 3)
        Z = a * np.sin(freq * R + theta)
        features[[0, 3, 4, 5]] = 1

    Z = np.clip(Z, -15, 15)
    return Z.astype(np.float32), features


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_to_image_2d(x, y, size=IMG_SIZE):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=size // 2)
    ax.plot(x, y, 'b-', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-12, 12)
    ax.axis('off')
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('L').resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


def plot_to_image_3d(X, Y, Z, size=IMG_SIZE, elev=None, azim=None):
    if elev is None:
        elev = random.uniform(20, 50)
    if azim is None:
        azim = random.uniform(20, 340)

    fig = plt.figure(figsize=(2, 2), dpi=size // 2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                    rstride=2, cstride=2, linewidth=0, antialiased=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    z_range = max(abs(Z.min()), abs(Z.max()), 1)
    ax.set_zlim(-z_range, z_range)
    ax.axis('off')
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('L').resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


# ── Dataset ──────────────────────────────────────────────────────────────────

class FunctionDataset(Dataset):
    def __init__(self, n=NUM_TRAIN, img_size=IMG_SIZE):
        x_1d = np.linspace(-5, 5, 200)
        grid = np.linspace(-3, 3, 60)
        X, Y = np.meshgrid(grid, grid)

        print(f"Generating {n} samples...")
        imgs, labels, feats = [], [], []
        for i in range(n):
            if i % 1000 == 0 and i > 0:
                print(f"  {i}/{n}")
            ft = FUNCTION_TYPES[i % NUM_CLASSES]
            label_idx = FUNCTION_TYPES.index(ft)

            if ft in FUNCTION_TYPES_2D:
                y, f = generate_function_2d(ft, x_1d)
                img = plot_to_image_2d(x_1d, y, img_size)
            else:
                Z, f = generate_function_3d(ft, X, Y)
                img = plot_to_image_3d(X, Y, Z, img_size)

            imgs.append(img)
            labels.append(label_idx)
            feats.append(f)

        self.imgs   = np.array(imgs)
        self.labels = np.array(labels, dtype=np.int64)
        self.feats  = np.array(feats,  dtype=np.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.imgs[i]).unsqueeze(0),
            torch.tensor(self.labels[i]),
            torch.tensor(self.feats[i]),
        )


# ── Model (ResNet-style with SE blocks) ────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv(x)
        out = self.se(out)
        out = self.relu(out + identity)
        return self.drop(self.pool(out))


class FunctionCNN(nn.Module):
    """
    Multi-task CNN with residual blocks, SE attention, and three heads:
      1. Classifier  – predicts function type (2D or 3D)
      2. Detector    – predicts binary function properties
      3. Decoder     – reconstructs the input image
    """
    def __init__(self):
        super().__init__()

        # Backbone: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.backbone = nn.Sequential(
            ResConvBlock(1,   32,  0.05),
            ResConvBlock(32,  64,  0.10),
            ResConvBlock(64,  128, 0.15),
            ResConvBlock(128, 256, 0.20),
            ResConvBlock(256, 512, 0.25),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, NUM_CLASSES)

        self.detector = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, NUM_FEATURES), nn.Sigmoid(),
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512 * 4 * 4), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z      = self.backbone(x)
        pooled = self.global_pool(z).flatten(1)
        shared = self.shared_fc(pooled)
        logits = self.classifier(shared)
        feats  = self.detector(shared)
        recon  = self.decoder(self.dec_fc(shared).view(-1, 512, 4, 4))
        return logits, feats, recon


# ── Training with AMP ────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    cls_loss_fn   = nn.CrossEntropyLoss()
    feat_loss_fn  = nn.BCELoss()
    recon_loss_fn = nn.MSELoss()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels, feats in loader:
        imgs, labels, feats = imgs.to(device), labels.to(device), feats.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, pred_feats, recon = model(imgs)
                loss = (cls_loss_fn(logits, labels)
                        + 0.5 * feat_loss_fn(pred_feats, feats)
                        + 0.3 * recon_loss_fn(recon, imgs))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, pred_feats, recon = model(imgs)
            loss = (cls_loss_fn(logits, labels)
                    + 0.5 * feat_loss_fn(pred_feats, feats)
                    + 0.3 * recon_loss_fn(recon, imgs))
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if USE_AMP:
            with torch.amp.autocast('cuda'):
                logits, _, _ = model(imgs)
        else:
            logits, _, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Visualisation ────────────────────────────────────────────────────────────

def save_predictions(model, dataset, device, path='predictions.png', n=8):
    model.eval()
    idx = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    with torch.no_grad():
        for col, i in enumerate(idx):
            img, lbl, _ = dataset[i]
            inp = img.unsqueeze(0).to(device)
            if USE_AMP:
                with torch.amp.autocast('cuda'):
                    logits, _, recon = model(inp)
            else:
                logits, _, recon = model(inp)
            pred = logits.argmax(1).item()
            color = 'green' if pred == lbl.item() else 'red'

            axes[0, col].imshow(img[0], cmap='gray')
            axes[0, col].set_title(FUNCTION_TYPES[lbl.item()][:6], fontsize=7)
            axes[0, col].axis('off')

            axes[1, col].imshow(img[0], cmap='gray')
            axes[1, col].set_title(FUNCTION_TYPES[pred][:6], fontsize=7, color=color)
            axes[1, col].axis('off')

            axes[2, col].imshow(recon.cpu()[0, 0], cmap='gray')
            axes[2, col].set_title('recon', fontsize=7)
            axes[2, col].axis('off')

    for row, lbl in enumerate(['Original', 'Predicted', 'Reconstructed']):
        axes[row, 0].set_ylabel(lbl, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved {path}")


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze(model, device, func_type=None, x=None, y=None, X=None, Y=None, Z=None):
    model.eval()

    if func_type in FUNCTION_TYPES_3D:
        if X is None:
            grid = np.linspace(-3, 3, 60)
            X, Y = np.meshgrid(grid, grid)
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
        with torch.amp.autocast('cuda'):
            logits, pred_feats, _ = model(t)
    else:
        logits, pred_feats, _ = model(t)
    probs     = torch.softmax(logits, 1).cpu().numpy()[0]
    pred_feat = pred_feats.cpu().numpy()[0]
    pred_cls  = probs.argmax()

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
    if true_feats is not None:
        print("\n  True properties:")
        for name, val in zip(FEATURE_NAMES, true_feats):
            marker = "Y" if val > 0.5 else "."
            print(f"    {marker} {name}")
    return probs, pred_feat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data workers: {NUM_WORKERS}")
    print(f"Mixed precision: {USE_AMP}")
    print(f"Function types: {NUM_CLASSES} ({len(FUNCTION_TYPES_2D)} 2D + {len(FUNCTION_TYPES_3D)} 3D)")

    train_set = FunctionDataset(NUM_TRAIN)
    val_set   = FunctionDataset(NUM_VAL)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)

    model = FunctionCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    best_val, history = 0.0, {'loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nTraining {NUM_EPOCHS} epochs ...")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, train_acc = run_epoch(model, train_loader, optimizer, device, scaler)
        val_acc         = evaluate(model, val_loader, device)
        scheduler.step()

        history['loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"  [{epoch:2d}/{NUM_EPOCHS}]  loss={loss:.4f}  "
              f"train={train_acc*100:.1f}%  val={val_acc*100:.1f}%")

    print(f"\nBest val accuracy: {best_val*100:.1f}%")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history['loss']);  ax1.set_title('Loss');     ax1.set_xlabel('Epoch')
    ax2.plot(history['train_acc'], label='train')
    ax2.plot(history['val_acc'],   label='val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100)
    plt.close()
    print("Saved training_curves.png")

    save_predictions(model, val_set, device)

    print("\n-- Demo: one sample per function type --")
    for ft in FUNCTION_TYPES:
        analyze(model, device, func_type=ft)


if __name__ == '__main__':
    main()
