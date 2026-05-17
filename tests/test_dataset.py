"""Smoke tests for the synthetic dataset and generators."""
import numpy as np

from src.data import (
    FUNCTION_TYPES,
    FUNCTION_TYPES_2D,
    FUNCTION_TYPES_3D,
    IMG_SIZE,
    NUM_CLASSES,
    NUM_FEATURES,
    FunctionDataset,
    generate_function_2d,
    generate_function_3d,
)


def test_taxonomy_sizes():
    assert NUM_CLASSES == len(FUNCTION_TYPES) == 16
    assert NUM_FEATURES == 9
    assert set(FUNCTION_TYPES) == set(FUNCTION_TYPES_2D) | set(FUNCTION_TYPES_3D)


def test_generate_2d_shapes_and_features():
    x = np.linspace(-5, 5, 200)
    for ft in FUNCTION_TYPES_2D:
        y, feats = generate_function_2d(ft, x)
        assert y.shape == x.shape
        assert y.dtype == np.float32
        assert feats.shape == (NUM_FEATURES,)
        assert feats[6] == 0.0  # is_3d must be false for 2D


def test_generate_3d_shapes_and_features():
    grid = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(grid, grid)
    for ft in FUNCTION_TYPES_3D:
        Z, feats = generate_function_3d(ft, X, Y)
        assert Z.shape == X.shape
        assert feats[6] == 1.0  # is_3d must be true for 3D


def test_dataset_smoke():
    ds = FunctionDataset(n=NUM_CLASSES, verbose=False)
    assert len(ds) == NUM_CLASSES
    img, label, feats = ds[0]
    assert img.shape == (1, IMG_SIZE, IMG_SIZE)
    assert 0 <= int(label) < NUM_CLASSES
    assert feats.shape == (NUM_FEATURES,)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
