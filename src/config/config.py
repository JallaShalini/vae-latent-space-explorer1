"""Project configuration entrypoint.

This module centralizes defaults, environment overrides, and reusable paths
so the rest of the codebase can import a single settings object.
"""

from dataclasses import dataclass
import os

from .constants import BATCH_SIZE, EPOCHS, LATENT_DIM, LEARNING_RATE
from .paths import DATA_PATH, MODELS_PATH, RESULTS_PATH


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class Settings:
    latent_dim: int = LATENT_DIM
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    model_path: str = str(MODELS_PATH / "vae.pt")
    results_path: str = str(RESULTS_PATH)
    data_path: str = str(DATA_PATH)


def get_settings() -> Settings:
    return Settings(
        latent_dim=_env_int("LATENT_DIM", LATENT_DIM),
        batch_size=_env_int("BATCH_SIZE", BATCH_SIZE),
        learning_rate=_env_float("LEARNING_RATE", LEARNING_RATE),
        epochs=_env_int("EPOCHS", EPOCHS),
        model_path=os.getenv("MODEL_PATH", str(MODELS_PATH / "vae.pt")),
        results_path=os.getenv("RESULTS_PATH", str(RESULTS_PATH)),
        data_path=os.getenv("DATA_PATH", str(DATA_PATH)),
    )


settings = get_settings()
