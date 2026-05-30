"""Latent-space analysis helpers for the VAE."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def collect_latent_vectors(model: torch.nn.Module, dataloader, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    latent_vectors = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            mu, _ = model.encode(inputs)
            latent_vectors.append(np.asarray(mu.detach().cpu().tolist(), dtype=np.float32))
            labels.append(np.asarray(targets.detach().cpu().tolist(), dtype=np.int64))

    if latent_vectors:
        latent_vectors_array = np.concatenate(latent_vectors, axis=0)
        labels_array = np.concatenate(labels, axis=0)
    else:
        latent_vectors_array = np.empty((0, model.latent_dim))
        labels_array = np.empty((0,), dtype=np.int64)

    return latent_vectors_array, labels_array


def project_latent_space(latent_vectors: np.ndarray, random_state: int = 42, tsne_sample_size: int = 2000):
    if latent_vectors.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    pca_embedding = PCA(n_components=2, random_state=random_state).fit_transform(latent_vectors)

    sample_size = min(tsne_sample_size, latent_vectors.shape[0])
    if sample_size < 2:
        tsne_embedding = np.zeros((latent_vectors.shape[0], 2), dtype=np.float32)
        return pca_embedding, tsne_embedding

    if sample_size < latent_vectors.shape[0]:
        rng = np.random.default_rng(random_state)
        sample_indices = np.sort(rng.choice(latent_vectors.shape[0], size=sample_size, replace=False))
        tsne_source = latent_vectors[sample_indices]
        tsne_coords = TSNE(
            n_components=2,
            perplexity=max(5, min(30, sample_size - 1)),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(tsne_source)
        tsne_embedding = np.full((latent_vectors.shape[0], 2), np.nan, dtype=np.float32)
        tsne_embedding[sample_indices] = tsne_coords
    else:
        tsne_embedding = TSNE(
            n_components=2,
            perplexity=max(5, min(30, latent_vectors.shape[0] - 1)),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(latent_vectors)

    return pca_embedding, tsne_embedding


def save_latent_vectors(latent_vectors: np.ndarray, labels: np.ndarray, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "latent_vectors.npy", latent_vectors)
    np.save(output_dir / "latent_labels.npy", labels)


def plot_latent_space(pca_embedding: np.ndarray, tsne_embedding: np.ndarray, labels: np.ndarray, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    scatter_kwargs = {"c": labels, "cmap": "tab10", "s": 8, "alpha": 0.7}

    axes[0].scatter(pca_embedding[:, 0], pca_embedding[:, 1], **scatter_kwargs)
    axes[0].set_title("PCA Latent Projection")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")

    finite_mask = np.isfinite(tsne_embedding).all(axis=1)
    if finite_mask.any():
        axes[1].scatter(tsne_embedding[finite_mask, 0], tsne_embedding[finite_mask, 1], **{**scatter_kwargs, "c": labels[finite_mask]})
    axes[1].set_title("t-SNE Latent Projection")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")

    colorbar = fig.colorbar(axes[0].collections[0], ax=axes, shrink=0.95)
    colorbar.set_label("Digit Label")
    fig.suptitle("VAE Latent Space Map", fontsize=16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
