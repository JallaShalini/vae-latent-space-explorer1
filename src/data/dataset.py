"""MNIST dataset helpers."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.config.paths import MNIST_DATA_PATH
from .transforms import get_mnist_transforms


class RawMNISTDataset(Dataset):
    def __init__(self, root=MNIST_DATA_PATH, train=True, download=True, transform=None):
        self.root = Path(root)
        self.transform = transform

        if download:
            from torchvision.datasets import MNIST as TorchvisionMNIST

            TorchvisionMNIST(root=self.root, train=train, download=True)

        raw_dir = self.root / "MNIST" / "raw"
        if train:
            image_file = raw_dir / "train-images-idx3-ubyte"
            label_file = raw_dir / "train-labels-idx1-ubyte"
        else:
            image_file = raw_dir / "t10k-images-idx3-ubyte"
            label_file = raw_dir / "t10k-labels-idx1-ubyte"

        self.images = self._read_idx_images(image_file)
        self.labels = self._read_idx_labels(label_file)

    @staticmethod
    def _open_idx(path: Path):
        return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")

    @classmethod
    def _read_idx_images(cls, path: Path) -> torch.Tensor:
        with cls._open_idx(path) as file_handle:
            magic, count, rows, cols = struct.unpack(">IIII", file_handle.read(16))
            if magic != 2051:
                raise ValueError(f"Unexpected MNIST image magic number: {magic}")
            buffer = bytearray(file_handle.read(count * rows * cols))
        images = torch.frombuffer(buffer, dtype=torch.uint8).clone()
        return images.view(count, 1, rows, cols).float().div_(255.0)

    @classmethod
    def _read_idx_labels(cls, path: Path) -> torch.Tensor:
        with cls._open_idx(path) as file_handle:
            magic, count = struct.unpack(">II", file_handle.read(8))
            if magic != 2049:
                raise ValueError(f"Unexpected MNIST label magic number: {magic}")
            buffer = bytearray(file_handle.read(count))
        return torch.frombuffer(buffer, dtype=torch.uint8).clone().long()

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index].item())
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_mnist_datasets(root=MNIST_DATA_PATH, download=True):
    transform = get_mnist_transforms()
    train_dataset = RawMNISTDataset(root=root, train=True, download=download, transform=transform)
    test_dataset = RawMNISTDataset(root=root, train=False, download=download, transform=transform)
    return train_dataset, test_dataset
