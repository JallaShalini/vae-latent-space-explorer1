from __future__ import annotations

import torch
import torch.nn.functional as F

from src.training.losses import vae_loss


def test_vae_loss_matches_manual_formula() -> None:
    reconstructed = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]], dtype=torch.float32)
    original = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    mu = torch.tensor([[0.2, -0.3]], dtype=torch.float32)
    logvar = torch.tensor([[0.1, -0.2]], dtype=torch.float32)

    total_loss, reconstruction_loss, kl_divergence = vae_loss(reconstructed, original, mu, logvar, beta=0.5)

    expected_reconstruction = F.binary_cross_entropy(reconstructed, original, reduction="sum")
    expected_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    expected_total = expected_reconstruction + 0.5 * expected_kl

    assert torch.isclose(reconstruction_loss, expected_reconstruction)
    assert torch.isclose(kl_divergence, expected_kl)
    assert torch.isclose(total_loss, expected_total)
