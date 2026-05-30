"""Command-line entry point for training the VAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VAE on MNIST.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--annealing-epochs", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        epochs=args.epochs or 50,
        batch_size=args.batch_size or 128,
        learning_rate=args.learning_rate or 1e-3,
        annealing_epochs=args.annealing_epochs,
        max_batches=args.max_batches,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
