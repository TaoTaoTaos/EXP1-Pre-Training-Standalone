from __future__ import annotations

import pandas as pd

from lakeice_ncde.evaluation.metrics import compute_regression_metrics


def compute_per_lake_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for each lake separately."""
    rows = []
    for lake_name, lake_df in predictions.groupby("lake_name"):
        metrics = compute_regression_metrics(
            y_true=lake_df["y_true"].to_numpy(),
            y_pred=lake_df["y_pred"].to_numpy(),
        )
        row = {"lake_name": lake_name, "count": int(len(lake_df))}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lake_name").reset_index(drop=True)
