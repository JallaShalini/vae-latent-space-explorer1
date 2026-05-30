"""High-level training entrypoint used by scripts and tests."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config.config import settings
from src.data.dataloader import get_mnist_dataloaders
from src.models.vae import VAE
from .trainer import VAETrainer


def train_model(
    epochs: int = settings.epochs,
    batch_size: int = settings.batch_size,
    learning_rate: float = settings.learning_rate,
    annealing_epochs: int = 20,
    max_batches: int | None = None,
    log_path: str | Path = Path("results") / "training_log.csv",
    model_path: str | Path = Path("models") / "vae.pt",
    best_model_path: str | Path = Path("models") / "best_vae.pt",
    download: bool = True,
):
    train_loader, _ = get_mnist_dataloaders(batch_size=batch_size, download=download)
    model = VAE(latent_dim=settings.latent_dim)
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        learning_rate=learning_rate,
        log_path=log_path,
        model_path=model_path,
        best_model_path=best_model_path,
    )
    return trainer.fit(epochs=epochs, annealing_epochs=annealing_epochs, max_batches=max_batches)
