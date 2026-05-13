"""Function CNN package: data generation, model, training utilities."""
from .model import FunctionCNN, BaselineCNN, SEBlock, ResConvBlock
from .data import (
    FUNCTION_TYPES, FUNCTION_TYPES_2D, FUNCTION_TYPES_3D,
    FEATURE_NAMES, NUM_CLASSES, NUM_FEATURES, IMG_SIZE,
    FunctionDataset, generate_function_2d, generate_function_3d,
    plot_to_image_2d, plot_to_image_3d,
)
