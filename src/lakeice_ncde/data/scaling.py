from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class StandardScalerBundle:
    """Simple standard-scaler statistics for dataframe features."""

    feature_columns: list[str]
    mean: dict[str, float]
    std: dict[str, float]
    target_transform: str
    target_column: str

    def to_dict(self) -> dict:
        """Convert the scaler bundle to a serializable dictionary."""
        return asdict(self)


def fit_feature_scaler(train_df: pd.DataFrame, feature_columns: list[str], target_transform: str, target_column: str) -> StandardScalerBundle:
    """Fit feature scaling statistics from the train subset only."""
    mean = {}
    std = {}
    for column in feature_columns:
        series = pd.to_numeric(train_df[column], errors="coerce")
        column_mean = float(series.mean())
        column_std = float(series.std(ddof=0))
        mean[column] = column_mean
        std[column] = column_std if column_std > 0 else 1.0
    return StandardScalerBundle(
        feature_columns=feature_columns,
        mean=mean,
        std=std,
        target_transform=target_transform,
        target_column=target_column,
    )


def apply_feature_scaler(df: pd.DataFrame, scaler: StandardScalerBundle) -> pd.DataFrame:
    """Apply feature scaling without touching the target."""
    output = df.copy()
    for column in scaler.feature_columns:
        values = pd.to_numeric(output[column], errors="coerce")
        centered = values.fillna(scaler.mean[column]) - scaler.mean[column]
        output[column] = centered / scaler.std[column]
    return output


def transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply a target transform."""
    if transform == "none":
        return values
    if transform == "log1p":
        if np.any(values < 0):
            raise ValueError("log1p target transform received a negative target.")
        return np.log1p(values)
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    """Invert a target transform."""
    if transform == "none":
        return values
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"Unsupported target transform: {transform}")
