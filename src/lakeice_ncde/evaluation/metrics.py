from __future__ import annotations

import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute core regression metrics."""
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty target array.")
    residuals = y_pred - y_true
    mse = float(np.mean(np.square(residuals)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(residuals)) / denom) if denom > 0 else 0.0
    negative_count = int(np.sum(y_pred < 0))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "negative_count": float(negative_count),
    }
