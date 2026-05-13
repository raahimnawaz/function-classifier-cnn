"""Backwards-compatible shim — the original single-file script.

The real code now lives in `src/`. This file is kept so that any external
references (notebook imports, blog-post links) keep working. New code should
import from `src` directly.
"""
from src.data import *  # noqa: F401,F403
from src.model import *  # noqa: F401,F403
from src.train import (  # noqa: F401
    BATCH_SIZE, NUM_EPOCHS, NUM_TRAIN, NUM_VAL, LR, MODEL_PATH,
    NUM_WORKERS, USE_AMP,
    run_epoch, evaluate, save_predictions, analyze, main,
)


if __name__ == "__main__":
    main()
