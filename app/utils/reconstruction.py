"""Reconstruction helpers for the Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.evaluation.reconstruction_analysis import compute_error_heatmap
from .data_loader import load_test_loader
from .model_loader import load_vae_model


@lru_cache(maxsize=1)
def get_reconstruction_sample(index: int = 10, model_path: str | Path | None = None):
    test_loader = load_test_loader()
    dataset = test_loader.dataset
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} is outside the test dataset range")

    image, label = dataset[index]
    model = load_vae_model(model_path)

    with torch.no_grad():
        reconstruction, _, _ = model(image.unsqueeze(0))

    # Convert tensors to 2D numpy arrays (H, W) for Streamlit/Matplotlib
    original_tensor = image.squeeze().cpu()
    reconstructed_tensor = reconstruction.squeeze().cpu()
    original_np = np.asarray(original_tensor.tolist(), dtype=np.float32)
    reconstructed_np = np.asarray(reconstructed_tensor.tolist(), dtype=np.float32)
    heatmap_tensor = compute_error_heatmap(original_tensor, reconstructed_tensor)
    heatmap_np = np.asarray(heatmap_tensor.squeeze().cpu().tolist(), dtype=np.float32)
    return original_np, reconstructed_np, heatmap_np, int(label)
