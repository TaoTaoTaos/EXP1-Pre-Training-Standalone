from __future__ import annotations

from pathlib import Path

import pandas as pd

from lakeice_ncde.utils.io import save_dataframe


class HistoryLogger:
    """Collect and persist epoch-level metrics."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log_epoch(self, row: dict) -> None:
        """Store one epoch summary row."""
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the logged history as a dataframe."""
        return pd.DataFrame(self.rows)

    def save(self, path: Path) -> None:
        """Save the epoch history to disk."""
        save_dataframe(self.to_dataframe(), path)
