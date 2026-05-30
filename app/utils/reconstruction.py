"""Reconstruction helpers for the Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
    original_np = image.squeeze().cpu().numpy()
    reconstructed_np = reconstruction.squeeze().cpu().numpy()
    heatmap_tensor = compute_error_heatmap(torch.from_numpy(original_np), torch.from_numpy(reconstructed_np))
    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    return original_np, reconstructed_np, heatmap_np, int(label)
