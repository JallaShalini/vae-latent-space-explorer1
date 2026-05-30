from __future__ import annotations

from pathlib import Path

import torch


def test_checkpoints_load_with_torch() -> None:
    vae_path = Path("models") / "vae.pt"
    best_path = Path("models") / "best_vae.pt"

    vae_state = torch.load(vae_path, map_location="cpu")
    best_state = torch.load(best_path, map_location="cpu")

    assert isinstance(vae_state, dict)
    assert isinstance(best_state, dict)
    assert len(vae_state) > 0
    assert len(best_state) > 0
