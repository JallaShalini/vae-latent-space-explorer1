"""Generate a reconstruction comparison for a selected MNIST test sample index."""

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
from src.evaluation.reconstruction_analysis import (
    compute_error_heatmap,
    reconstruct_sample,
    save_heatmap,
    save_image,
)
from src.models.vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a reconstruction comparison for a test image.")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model-path", type=str, default=str(Path("models") / "vae.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, test_loader = get_mnist_dataloaders(batch_size=1, download=True)

    test_dataset = test_loader.dataset
    if args.index < 0 or args.index >= len(test_dataset):
        raise IndexError(f"Index {args.index} is outside the test dataset range of 0 to {len(test_dataset) - 1}.")

    image, _ = test_dataset[args.index]

    model = VAE(latent_dim=settings.latent_dim)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    original, reconstructed = reconstruct_sample(model, image)
    error_heatmap = compute_error_heatmap(original, reconstructed)

    output_dir = Path("results")
    save_image(original, output_dir / f"original_{args.index}.png", title="Original")
    save_image(reconstructed, output_dir / f"reconstructed_{args.index}.png", title="Reconstructed")
    save_heatmap(error_heatmap, output_dir / f"heatmap_{args.index}.png", title="Absolute Error")


if __name__ == "__main__":
    main()
