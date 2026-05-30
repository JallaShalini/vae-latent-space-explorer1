"""Data transforms for MNIST preprocessing."""

from torchvision import transforms


def get_mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])
