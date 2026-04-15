from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe with UTF-8 encoding and parent creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_dataframe(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV dataframe with optional date parsing."""
    return pd.read_csv(path, parse_dates=parse_dates, encoding="utf-8-sig")


def save_json(data: Any, path: Path) -> None:
    """Save JSON data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_yaml(data: Any, path: Path) -> None:
    """Save YAML data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append a row to a CSV file, creating a header on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
