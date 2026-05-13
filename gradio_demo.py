"""Tiny Gradio demo: draw a curve, see the prediction.

Run:
    python gradio_demo.py

The user sketches a 2D curve on a white canvas; the sketch is downsampled to
128x128, fed through the trained `FunctionCNN`, and the top-3 function-type
predictions plus detected structural properties are returned.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

import gradio as gr

from src.data import FEATURE_NAMES, FUNCTION_TYPES, IMG_SIZE
from src.model import FunctionCNN

MODEL_PATH = "function_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> FunctionCNN:
    if not os.path.exists(MODEL_PATH):
        sys.stderr.write(
            f"\n[gradio_demo] Trained weights not found at '{MODEL_PATH}'.\n"
            f"             Train the model first:\n\n"
            f"                 python -m src.train\n\n"
            f"             Then re-run `python gradio_demo.py`.\n\n"
        )
        sys.exit(1)
    model = FunctionCNN().to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    except Exception as e:
        sys.stderr.write(
            f"\n[gradio_demo] Failed to load weights from '{MODEL_PATH}': {e}\n"
            f"             The checkpoint may be from a different model version.\n"
            f"             Re-train with `python -m src.train`.\n\n"
        )
        sys.exit(1)
    model.eval()
    return model


MODEL = load_model()


def _to_tensor(sketch) -> torch.Tensor:
    """Accepts a Gradio Sketchpad dict or a numpy array; returns 1x1x128x128 in [0,1]."""
    if sketch is None:
        return None
    if isinstance(sketch, dict):
        # Gradio >=4 ImageEditor returns {"composite": np.ndarray, ...}
        arr = sketch.get("composite", sketch.get("image"))
        if arr is None and "layers" in sketch and sketch["layers"]:
            arr = sketch["layers"][0]
    else:
        arr = sketch
    if arr is None:
        return None
    img = Image.fromarray(np.asarray(arr)).convert("L")
    # Drawings are dark strokes on a light canvas. Match training: dark ink on
    # white background, normalised to [0, 1]. Invert if average is dark.
    if np.mean(img) < 128:
        img = ImageOps.invert(img)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(sketch):
    t = _to_tensor(sketch)
    if t is None:
        return {}, "Draw a curve on the canvas first."
    logits, feats, _ = MODEL(t)
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    feats = feats.cpu().numpy()[0]
    top = {FUNCTION_TYPES[i]: float(probs[i]) for i in probs.argsort()[::-1][:5]}
    detected = [name for name, v in zip(FEATURE_NAMES, feats) if v > 0.5]
    summary = "Detected properties: " + (", ".join(detected) if detected else "(none above 0.5)")
    return top, summary


def build_interface():
    with gr.Blocks(title="Function CNN") as demo:
        gr.Markdown(
            "# Function CNN — sketch demo\n"
            "Draw a 2D curve (sine, parabola, line, exponential, ...) on the canvas. "
            "The model returns its top-5 guess and the structural properties it sees."
        )
        with gr.Row():
            with gr.Column():
                canvas = gr.Sketchpad(
                    label="Draw here",
                    height=400,
                    canvas_size=(400, 400),
                )
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                label = gr.Label(num_top_classes=5, label="Top predictions")
                props = gr.Textbox(label="Properties", lines=2)
        btn.click(predict, inputs=canvas, outputs=[label, props])
    return demo


if __name__ == "__main__":
    build_interface().launch()
