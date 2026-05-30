"""Resolved project paths used across training, inference, and the app."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
APP_PATH = PROJECT_ROOT / "app"
MODELS_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MNIST_DATA_PATH = DATA_PATH / "mnist"
