"""CNN encoder for the MNIST VAE."""

import torch
from torch import nn

from src.config.config import settings


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = settings.latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.flatten(x)
        x = self.hidden(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
