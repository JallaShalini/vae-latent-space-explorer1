"""Latent-space helpers for the Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.evaluation.latent_analysis import collect_latent_vectors, project_latent_space
from .data_loader import load_test_loader
from .model_loader import load_vae_model


@lru_cache(maxsize=1)
def get_latent_space_data(model_path: str | Path | None = None):
    model = load_vae_model(model_path)
    test_loader = load_test_loader()
    latent_vectors, labels = collect_latent_vectors(model, test_loader)
    pca_embedding, tsne_embedding = project_latent_space(latent_vectors)
    return latent_vectors, labels, pca_embedding, tsne_embedding


def decode_latent_vector(latent_vector: np.ndarray, model_path: str | Path | None = None):
    model = load_vae_model(model_path)
    import torch

    with torch.no_grad():
        tensor = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0)
        # squeeze all singleton dimensions so the returned array is 2D (H, W)
        reconstruction = np.asarray(model.decode(tensor).squeeze().detach().cpu().tolist(), dtype=np.float32)
    return reconstruction
