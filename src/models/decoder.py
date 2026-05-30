"""CNN decoder for the MNIST VAE."""

import torch
from torch import nn

from src.config.config import settings


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = settings.latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.hidden(z)
        x = x.view(-1, 64, 7, 7)
        return self.decoder(x)
