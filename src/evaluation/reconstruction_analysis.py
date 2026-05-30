"""Reconstruction analysis helpers for the VAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def reconstruct_sample(model: torch.nn.Module, image: torch.Tensor, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        reconstructed, _, _ = model(image_batch)

    return image.squeeze(0).cpu(), reconstructed.squeeze(0).cpu()


def compute_error_heatmap(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    return torch.abs(original - reconstructed)


def save_image(tensor: torch.Tensor, output_path: str | Path, title: str | None = None, cmap: str = "gray"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tensor.detach().cpu().tolist()).squeeze()

    fig, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(array, cmap=cmap)
    axis.axis("off")
    if title is not None:
        axis.set_title(title)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)


def save_heatmap(tensor: torch.Tensor, output_path: str | Path, title: str | None = None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tensor.detach().cpu().tolist()).squeeze()

    fig, axis = plt.subplots(figsize=(3, 3))
    image = axis.imshow(array, cmap="magma")
    axis.axis("off")
    if title is not None:
        axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
