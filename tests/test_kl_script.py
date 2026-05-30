from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_kl_plot_script_outputs_png() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_kl_plot.py", "--max-batches", "1"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (Path("results") / "kl_per_dimension.png").exists()
