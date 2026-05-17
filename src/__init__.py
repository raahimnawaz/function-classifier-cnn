"""Function CNN package: data generation, model, training utilities."""
from .data import (
    FEATURE_NAMES,
    FUNCTION_TYPES,
    FUNCTION_TYPES_2D,
    FUNCTION_TYPES_3D,
    IMG_SIZE,
    NUM_CLASSES,
    NUM_FEATURES,
    FunctionDataset,
    generate_function_2d,
    generate_function_3d,
    plot_to_image_2d,
    plot_to_image_3d,
)
from .model import BaselineCNN, FunctionCNN, ResConvBlock, SEBlock

__all__ = [
    "BaselineCNN",
    "FEATURE_NAMES",
    "FUNCTION_TYPES",
    "FUNCTION_TYPES_2D",
    "FUNCTION_TYPES_3D",
    "FunctionCNN",
    "FunctionDataset",
    "IMG_SIZE",
    "NUM_CLASSES",
    "NUM_FEATURES",
    "ResConvBlock",
    "SEBlock",
    "generate_function_2d",
    "generate_function_3d",
    "plot_to_image_2d",
    "plot_to_image_3d",
]
