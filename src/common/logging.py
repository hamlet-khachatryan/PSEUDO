from __future__ import annotations

from pathlib import Path

import eliot


def setup_eliot_logging(log_dir: Path, name: str) -> None:
    """
    Open (or append to) an eliot NDJSON log file for one run or experiment.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.ndjson"
    eliot.to_file(open(log_file, "a"))
