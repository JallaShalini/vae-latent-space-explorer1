"""Dataset helpers for the Streamlit app."""

from __future__ import annotations

from functools import lru_cache

from src.config.config import settings
from src.data.dataloader import get_mnist_dataloaders


@lru_cache(maxsize=1)
def load_test_loader(batch_size: int | None = None):
    _, test_loader = get_mnist_dataloaders(batch_size=batch_size or settings.batch_size, download=True)
    return test_loader
