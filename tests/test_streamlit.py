from __future__ import annotations

import subprocess
import sys
import time
from urllib.request import urlopen


def test_streamlit_app_starts_and_serves_root_page() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/app.py",
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8502",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.time() + 90
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with urlopen("http://127.0.0.1:8502", timeout=2) as response:
                    assert response.status == 200
                    return
            except Exception as error:  # pragma: no cover - retry loop
                last_error = error
                time.sleep(1)
        raise AssertionError(f"Streamlit app did not start in time: {last_error}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
