"""KL divergence analysis by latent dimension."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def collect_mu_logvar(model: torch.nn.Module, dataloader, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mu_values = []
    logvar_values = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            mu, logvar = model.encode(inputs)
            mu_values.append(np.asarray(mu.detach().cpu().tolist(), dtype=np.float32))
            logvar_values.append(np.asarray(logvar.detach().cpu().tolist(), dtype=np.float32))

    if mu_values:
        mu_array = np.concatenate(mu_values, axis=0)
        logvar_array = np.concatenate(logvar_values, axis=0)
    else:
        mu_array = np.empty((0, model.latent_dim))
        logvar_array = np.empty((0, model.latent_dim))

    return mu_array, logvar_array


def compute_kl_per_dimension(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    if mu.size == 0 or logvar.size == 0:
        return np.empty((0,), dtype=np.float32)
    kl_per_sample = -0.5 * (1 + logvar - np.square(mu) - np.exp(logvar))
    return kl_per_sample.mean(axis=0)


def save_kl_plot(kl_values: np.ndarray, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 4))
    x_positions = np.arange(len(kl_values))
    axis.bar(x_positions, kl_values, color="#4c78a8")
    axis.set_xlabel("Latent Dimension")
    axis.set_ylabel("Mean KL Divergence")
    axis.set_title("KL Divergence Per Latent Dimension")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([str(index) for index in x_positions])
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
