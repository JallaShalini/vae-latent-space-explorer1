from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_generate_reconstruction_script_outputs_images() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_reconstruction.py", "--index", "10"],
        check=True,
        capture_output=True,
        text=True,
    )

    results_dir = Path("results")
    assert (results_dir / "original_10.png").exists()
    assert (results_dir / "reconstructed_10.png").exists()
    assert (results_dir / "heatmap_10.png").exists()
