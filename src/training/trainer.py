"""Training loop implementation for the VAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.config.config import settings
from src.models.vae import VAE
from .kl_annealing import get_beta
from .losses import vae_loss
from .logger import TrainingLogger


@dataclass
class EpochMetrics:
    reconstruction_loss: float
    kl_divergence: float


class VAETrainer:
    def __init__(
        self,
        model: VAE,
        train_loader,
        device: torch.device | None = None,
        learning_rate: float = settings.learning_rate,
        log_path: str | Path = Path("results") / "training_log.csv",
        model_path: str | Path = Path("models") / "vae.pt",
        best_model_path: str | Path = Path("models") / "best_vae.pt",
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.logger = TrainingLogger(log_path)
        self.model_path = Path(model_path)
        self.best_model_path = Path(best_model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def train_epoch(self, epoch: int, annealing_epochs: int = 20, max_batches: int | None = None) -> EpochMetrics:
        self.model.train()
        beta = get_beta(epoch, annealing_epochs=annealing_epochs)
        reconstruction_total = 0.0
        kl_total = 0.0
        batches_seen = 0

        for batch_index, (inputs, _) in enumerate(self.train_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = inputs.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            reconstructed, mu, logvar = self.model(inputs)
            loss, reconstruction_loss, kl_divergence = vae_loss(
                reconstructed,
                inputs,
                mu,
                logvar,
                beta=beta,
            )
            loss.backward()
            self.optimizer.step()

            reconstruction_total += reconstruction_loss.item()
            kl_total += kl_divergence.item()
            batches_seen += 1

        if batches_seen == 0:
            return EpochMetrics(reconstruction_loss=0.0, kl_divergence=0.0)

        return EpochMetrics(
            reconstruction_loss=reconstruction_total / batches_seen,
            kl_divergence=kl_total / batches_seen,
        )

    def fit(self, epochs: int, annealing_epochs: int = 20, max_batches: int | None = None):
        history = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(epoch, annealing_epochs=annealing_epochs, max_batches=max_batches)
            history.append(
                {
                    "epoch": epoch,
                    "reconstruction_loss": metrics.reconstruction_loss,
                    "kl_divergence": metrics.kl_divergence,
                }
            )
            self.logger.log_epoch(epoch, metrics.reconstruction_loss, metrics.kl_divergence)

            current_loss = metrics.reconstruction_loss + metrics.kl_divergence
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                torch.save(self.model.state_dict(), self.best_model_path)

        torch.save(self.model.state_dict(), self.model_path)
        return history
