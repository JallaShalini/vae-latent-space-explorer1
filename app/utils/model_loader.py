"""Model loading helpers for the Streamlit app."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from src.config.config import settings
from src.models.vae import VAE
from src.config.paths import MODELS_PATH


@lru_cache(maxsize=1)
def load_vae_model(model_path: str | Path | None = None) -> VAE:
    # Prefer an explicitly provided path. If none provided, prefer a "best_vae.pt"
    # checkpoint in the models directory when available, otherwise fall back to
    # the configured default in `settings.model_path`.
    if model_path:
        resolved_path = Path(model_path)
    else:
        candidate = Path(MODELS_PATH) / "best_vae.pt"
        if candidate.exists():
            resolved_path = candidate
        else:
            resolved_path = Path(settings.model_path)
    model = VAE(latent_dim=settings.latent_dim)
    state_dict = torch.load(resolved_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
