from __future__ import annotations

import csv
from pathlib import Path


def test_training_log_has_required_columns() -> None:
    log_path = Path("results") / "training_log.csv"
    assert log_path.exists()

    with log_path.open("r", encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))

    assert rows
    assert set(rows[0].keys()) == {"epoch", "reconstruction_loss", "kl_divergence"}
    int(rows[0]["epoch"])
    float(rows[0]["reconstruction_loss"])
    float(rows[0]["kl_divergence"])
