"""Export PCA and t-SNE latent space visualizations for the trained VAE."""

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
from src.evaluation.latent_analysis import (
    collect_latent_vectors,
    plot_latent_space,
    project_latent_space,
    save_latent_vectors,
)
from src.models.vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the latent space map for the trained VAE.")
    parser.add_argument("--model-path", type=str, default=str(Path("models") / "vae.pt"))
    parser.add_argument("--output-path", type=str, default=str(Path("results") / "latent_space_map.png"))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, test_loader = get_mnist_dataloaders(batch_size=settings.batch_size, download=True)

    model = VAE(latent_dim=settings.latent_dim)
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    latent_vectors, labels = collect_latent_vectors(model, test_loader)
    if args.max_samples is not None and args.max_samples > 0 and latent_vectors.shape[0] > args.max_samples:
        latent_vectors = latent_vectors[: args.max_samples]
        labels = labels[: args.max_samples]

    save_latent_vectors(latent_vectors, labels, Path(args.output_path).parent)
    pca_embedding, tsne_embedding = project_latent_space(latent_vectors)
    plot_latent_space(pca_embedding, tsne_embedding, labels, args.output_path)


if __name__ == "__main__":
    main()
