"""MNIST dataset helpers."""

from torchvision.datasets import MNIST

from src.config.paths import MNIST_DATA_PATH
from .transforms import get_mnist_transforms


def get_mnist_datasets(root=MNIST_DATA_PATH, download=True):
    transform = get_mnist_transforms()
    train_dataset = MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=download, transform=transform)
    return train_dataset, test_dataset
