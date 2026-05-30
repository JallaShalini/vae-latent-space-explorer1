"""Loss functions for training the VAE."""

import torch
import torch.nn.functional as F


def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.binary_cross_entropy(reconstructed, original, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + beta * kl_divergence
    return total_loss, reconstruction_loss, kl_divergence
