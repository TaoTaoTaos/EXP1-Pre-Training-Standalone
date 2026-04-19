from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from lakeice_ncde.utils.locking import PathLock


def _atomic_write(path: Path, write_fn) -> None:
    """Write a file via a same-directory temp path and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        write_fn(temp_path)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe with UTF-8 encoding and parent creation."""
    _atomic_write(path, lambda temp_path: df.to_csv(temp_path, index=False, encoding="utf-8-sig"))


def load_dataframe(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV dataframe with optional date parsing."""
    return pd.read_csv(path, parse_dates=parse_dates, encoding="utf-8-sig")


def save_json(data: Any, path: Path) -> None:
    """Save JSON data to disk."""
    def _write(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    _atomic_write(path, _write)


def save_yaml(data: Any, path: Path) -> None:
    """Save YAML data to disk."""
    def _write(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)

    _atomic_write(path, _write)


def save_torch(data: Any, path: Path) -> None:
    """Save a torch object via an atomic replace."""
    _atomic_write(path, lambda temp_path: torch.save(data, temp_path))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append a row to a CSV file, creating a header on first write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with PathLock(lock_path):
        write_header = not path.exists()
        with path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
