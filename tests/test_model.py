"""Smoke tests for FunctionCNN and BaselineCNN forward passes."""
import torch

from src.data import IMG_SIZE, NUM_CLASSES, NUM_FEATURES
from src.model import BaselineCNN, FunctionCNN


def _check_forward(model_cls, batch=2):
    model = model_cls().eval()
    x = torch.randn(batch, 1, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        logits, feats, recon = model(x)
    assert logits.shape == (batch, NUM_CLASSES)
    assert feats.shape == (batch, NUM_FEATURES)
    assert recon.shape[0] == batch
    assert recon.shape[1] == 1
    # detector head is sigmoid, must be in [0, 1]
    assert float(feats.min()) >= 0.0 and float(feats.max()) <= 1.0


def test_function_cnn_forward():
    _check_forward(FunctionCNN)


def test_baseline_cnn_forward():
    _check_forward(BaselineCNN)


def test_function_cnn_param_count_reasonable():
    n = sum(p.numel() for p in FunctionCNN().parameters())
    # Should be in the millions, not billions or thousands.
    assert 1_000_000 < n < 50_000_000


def test_baseline_smaller_than_full():
    nb = sum(p.numel() for p in BaselineCNN().parameters())
    nf = sum(p.numel() for p in FunctionCNN().parameters())
    assert nb < nf // 10  # baseline should be at least 10x smaller
