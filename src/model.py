"""Model definitions.

`FunctionCNN`  — main multi-task model with residual blocks, SE attention,
                 and three heads (classifier / property detector / decoder).
`BaselineCNN`  — deliberately weak 1-conv-layer reference for comparison.
"""
from __future__ import annotations

import torch.nn as nn

from .data import NUM_CLASSES, NUM_FEATURES

# ── Building blocks ──────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
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
    """Residual conv block: two 3x3 convs + BN, SE attention, skip, pool, dropout."""

    def __init__(self, in_ch: int, out_ch: int, drop: float = 0.1):
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


# ── Main model ───────────────────────────────────────────────────────────────

class FunctionCNN(nn.Module):
    """Residual CNN with SE attention and three heads:

      1. Classifier — predicts function type (16 classes)
      2. Detector   — predicts 9 binary structural properties
      3. Decoder    — reconstructs the input image (auxiliary task)
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
        z = self.backbone(x)
        pooled = self.global_pool(z).flatten(1)
        shared = self.shared_fc(pooled)
        logits = self.classifier(shared)
        feats  = self.detector(shared)
        recon  = self.decoder(self.dec_fc(shared).view(-1, 512, 4, 4))
        return logits, feats, recon


# ── Baseline ─────────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """A deliberately minimal 1-conv-layer CNN for honest comparison.

    Same input/output contract as `FunctionCNN` so it can drop into the same
    training loop, but the detector and decoder heads are trivial.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),  # 128 -> 32
            nn.AdaptiveAvgPool2d(8),  # 32 -> 8
        )
        self.classifier = nn.Linear(16 * 8 * 8, NUM_CLASSES)
        self.detector_head = nn.Sequential(
            nn.Linear(16 * 8 * 8, NUM_FEATURES), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.features(x).flatten(1)
        logits = self.classifier(z)
        feats = self.detector_head(z)
        # Decoder is a no-op placeholder so the training loop API matches.
        recon = x
        return logits, feats, recon
