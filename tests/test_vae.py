from __future__ import annotations

import torch

from src.models.vae import VAE


def test_vae_forward_and_reparameterize_shapes() -> None:
    model = VAE()
    inputs = torch.randn(2, 1, 28, 28)

    reconstructed, mu, logvar = model(inputs)
    latent = model.reparameterize(mu, logvar)

    assert reconstructed.shape == (2, 1, 28, 28)
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
    assert latent.shape == (2, 16)
