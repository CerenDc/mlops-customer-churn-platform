"""Persistent monitoring snapshot shared with the metrics exporter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MONITORING_ROOT = Path("data/monitoring")


def configured_monitoring_root() -> Path:
    """Return the portable runtime monitoring directory."""
    return Path(os.getenv("MONITORING_OUTPUT_DIR", str(DEFAULT_MONITORING_ROOT)))


def configured_metrics_state_path() -> Path:
    """Return the shared JSON snapshot path."""
    return Path(
        os.getenv(
            "MONITORING_METRICS_STATE_PATH",
            str(configured_monitoring_root() / "metrics.json"),
        )
    )


def write_metrics_state(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Atomically persist one complete monitoring snapshot."""
    destination = Path(path or configured_metrics_state_path())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(destination)
    return destination


def read_metrics_state(path: Path | None = None) -> dict[str, Any]:
    """Read the latest snapshot, returning an empty state before first run."""
    source = Path(path or configured_metrics_state_path())
    if not source.is_file():
        return {}
    return json.loads(source.read_text())
