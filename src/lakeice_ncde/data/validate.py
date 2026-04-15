from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Ensure all required columns exist."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_dataframe(df: pd.DataFrame, config: dict, raw_excel_path: Path) -> dict[str, Any]:
    """Build a validation report for the raw dataframe."""
    data_cfg = config["data"]
    lake_column = data_cfg["lake_column"]
    datetime_column = data_cfg["datetime_column"]
    target_column = data_cfg["target_column"]
    lake_id_column = data_cfg["lake_id_column"]

    validate_required_columns(df, data_cfg["required_columns"])

    parsed_dt = pd.to_datetime(df[datetime_column], errors="coerce")
    invalid_dt_count = int(parsed_dt.isna().sum())
    target_series = pd.to_numeric(df[target_column], errors="coerce")
    negative_target_count = int((target_series < 0).sum())

    report = {
        "raw_excel_path": str(raw_excel_path),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "unique_lakes": int(df[lake_column].nunique(dropna=True)),
        "missing_lake_name": int(df[lake_column].isna().sum()),
        "missing_lake_id": int(df[lake_id_column].isna().sum()),
        "invalid_sample_datetime": invalid_dt_count,
        "missing_target": int(target_series.isna().sum()),
        "negative_target_count": negative_target_count,
        "target_min": None if target_series.isna().all() else float(target_series.min()),
        "target_max": None if target_series.isna().all() else float(target_series.max()),
        "sample_datetime_min": None if parsed_dt.isna().all() else str(parsed_dt.min()),
        "sample_datetime_max": None if parsed_dt.isna().all() else str(parsed_dt.max()),
    }
    return report
