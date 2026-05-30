"""Create training curves from the saved VAE training log."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    log_path = Path("results") / "training_log.csv"
    output_path = Path("results") / "training_curves.png"

    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found at {log_path}")

    training_log = pd.read_csv(log_path)
    required_columns = {"epoch", "reconstruction_loss", "kl_divergence"}
    missing_columns = required_columns.difference(training_log.columns)
    if missing_columns:
        raise ValueError(f"Training log is missing columns: {sorted(missing_columns)}")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(training_log["epoch"], training_log["reconstruction_loss"], marker="o", label="Reconstruction Loss")
    axis.plot(training_log["epoch"], training_log["kl_divergence"], marker="o", label="KL Divergence")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("VAE Training Curves")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
