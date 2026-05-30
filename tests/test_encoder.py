from __future__ import annotations

import torch

from src.models.encoder import Encoder


def test_encoder_output_shapes() -> None:
    model = Encoder()
    inputs = torch.randn(4, 1, 28, 28)

    mu, logvar = model(inputs)

    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)
