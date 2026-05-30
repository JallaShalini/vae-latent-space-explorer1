"""Generate a KL-per-dimension plot for the trained VAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config.config import settings
from src.data.dataloader import get_mnist_dataloaders
from src.evaluation.kl_analysis import collect_mu_logvar, compute_kl_per_dimension, save_kl_plot
from src.models.vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a KL divergence plot per latent dimension.")
    parser.add_argument("--model-path", type=str, default=str(Path("models") / "vae.pt"))
    parser.add_argument("--output-path", type=str, default=str(Path("results") / "kl_per_dimension.png"))
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, test_loader = get_mnist_dataloaders(batch_size=settings.batch_size, download=True)

    model = VAE(latent_dim=settings.latent_dim)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if args.max_batches is not None and args.max_batches > 0:
        truncated_batches = []
        for batch_index, batch in enumerate(test_loader):
            if batch_index >= args.max_batches:
                break
            truncated_batches.append(batch)
        test_loader = truncated_batches

    mu, logvar = collect_mu_logvar(model, test_loader)
    kl_values = compute_kl_per_dimension(mu, logvar)
    save_kl_plot(kl_values, args.output_path)


if __name__ == "__main__":
    main()
