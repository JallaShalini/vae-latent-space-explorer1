from __future__ import annotations

import torch

from src.models.decoder import Decoder


def test_decoder_output_shape_and_range() -> None:
    model = Decoder()
    latent = torch.randn(4, 16)

    reconstruction = model(latent)

    assert reconstruction.shape == (4, 1, 28, 28)
    assert float(reconstruction.min()) >= 0.0
    assert float(reconstruction.max()) <= 1.0
