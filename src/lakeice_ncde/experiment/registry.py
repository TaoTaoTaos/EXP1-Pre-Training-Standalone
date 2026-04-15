from __future__ import annotations

from pathlib import Path

from lakeice_ncde.utils.io import append_csv_row


def append_experiment_registry(output_root: Path, row: dict) -> Path:
    """Append one row to the global experiment registry."""
    registry_path = output_root / "experiment_registry.csv"
    append_csv_row(registry_path, row)
    return registry_path
