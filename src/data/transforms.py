"""Data transforms for MNIST preprocessing."""


def get_mnist_transforms():
    """Return the transform used by the custom MNIST dataset.

    The custom dataset already emits normalized float tensors, so no extra
    torchvision transform is needed.
    """
    return None
