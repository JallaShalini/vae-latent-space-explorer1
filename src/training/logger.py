"""CSV logging utilities for VAE training."""

from __future__ import annotations

import csv
from pathlib import Path


class TrainingLogger:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=["epoch", "reconstruction_loss", "kl_divergence"],
            )
            writer.writeheader()

    def log_epoch(self, epoch: int, reconstruction_loss: float, kl_divergence: float) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=["epoch", "reconstruction_loss", "kl_divergence"],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_divergence": kl_divergence,
                }
            )
