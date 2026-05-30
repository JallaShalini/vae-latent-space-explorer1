"""Reusable MNIST dataloaders."""

from torch.utils.data import DataLoader

from src.config.config import settings
from src.config.paths import MNIST_DATA_PATH
from .dataset import get_mnist_datasets


def get_mnist_dataloaders(batch_size=settings.batch_size, root=MNIST_DATA_PATH, download=True):
    train_dataset, test_dataset = get_mnist_datasets(root=root, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


try:
    train_loader, test_loader = get_mnist_dataloaders(download=True)
except Exception:
    train_loader = None
    test_loader = None
