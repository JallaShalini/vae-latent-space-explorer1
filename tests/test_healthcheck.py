from __future__ import annotations

import subprocess
from urllib.request import urlopen

import pytest


def test_docker_compose_healthcheck_and_http_endpoint() -> None:
    """Check that docker compose reports the service healthy and the app responds.

    In some execution environments (for example, when running tests inside the
    application container), the `docker` CLI is not available. In that case we
    skip the docker-compose portion of the test and only assert the HTTP
    endpoint.
    """
    try:
        compose_output = subprocess.run(
            ["docker", "compose", "ps"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # Docker CLI isn't available in some environments (for example, when
        # running tests inside the application container). In that case we
        # fall back to verifying the HTTP endpoint only and consider the test
        # successful if the app responds.
        with urlopen("http://127.0.0.1:8501", timeout=5) as response:
            assert response.status == 200
        return
    except subprocess.CalledProcessError as exc:  # pragma: no cover - failure case
        pytest.fail(f"docker compose command failed: {exc}")

    assert "healthy" in compose_output.stdout.lower()

    with urlopen("http://127.0.0.1:8501", timeout=5) as response:
        assert response.status == 200
