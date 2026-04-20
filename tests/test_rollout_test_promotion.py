from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from lakeice_ncde.workflows.xiaoxingkai_transfer import _promote_seasonal_rollout_as_test


@pytest.fixture
def dummy_logger():
    class _Logger:
        def info(self, *args, **kwargs) -> None:
            return None

    return _Logger()


def test_promote_seasonal_rollout_as_test_replaces_test_metrics(tmp_path, dummy_logger) -> None:
    pd.DataFrame(
        [
            {"split": "val", "loss": 0.04, "rmse": 0.2, "mae": 0.2, "r2": 0.1, "bias": 0.0, "negative_count": 0.0},
            {"split": "test", "loss": 99.0, "rmse": 9.9, "mae": 9.9, "r2": -9.0, "bias": 9.9, "negative_count": 0.0},
        ]
    ).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"best_epoch": 3, "best_val_rmse": 0.2, "final_val_loss": 0.04, "final_test_loss": 99.0}),
        encoding="utf-8",
    )

    overlap_df = pd.DataFrame(
        [
            {"lake_name": "Xiaoxingkai", "sample_datetime": "2026-01-01 12:00:00", "y_true": 0.20, "y_pred": 0.22},
            {"lake_name": "Xiaoxingkai", "sample_datetime": "2026-01-03 12:00:00", "y_true": 0.30, "y_pred": 0.28},
        ]
    )
    overlap_predictions_path = tmp_path / "seasonal_rollout_overlap_predictions.csv"
    overlap_df.to_csv(overlap_predictions_path, index=False)

    overlap_metrics = {
        "count": 2,
        "rollout_rows": 90,
        "test_start_datetime": "2025-10-01 12:00:00",
        "rollout_start_datetime": "2025-10-01 12:00:00",
        "rollout_end_datetime": "2026-03-01 12:00:00",
        "overlap_start_datetime": "2026-01-01 12:00:00",
        "overlap_end_datetime": "2026-01-03 12:00:00",
        "rmse": 0.02,
        "mae": 0.02,
        "r2": 0.84,
        "bias": 0.0,
        "negative_count": 0.0,
    }
    overlap_metrics_json_path = tmp_path / "seasonal_rollout_overlap_metrics.json"
    overlap_metrics_json_path.write_text(json.dumps(overlap_metrics), encoding="utf-8")

    artifacts = SimpleNamespace(
        overlap_predictions_path=overlap_predictions_path,
        overlap_metrics_json_path=overlap_metrics_json_path,
    )
    result = _promote_seasonal_rollout_as_test(tmp_path, artifacts, dummy_logger)

    metrics_df = pd.read_csv(tmp_path / "metrics.csv")
    test_row = metrics_df.loc[metrics_df["split"] == "test"].iloc[0]
    run_summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    test_predictions = pd.read_csv(tmp_path / "test_predictions.csv")

    assert float(test_row["rmse"]) == pytest.approx(0.02)
    assert float(test_row["loss"]) == pytest.approx(0.0004)
    assert run_summary["test_method"] == "seasonal_rollout_overlap"
    assert run_summary["seasonal_test_start_datetime"] == "2025-10-01 12:00:00"
    assert run_summary["seasonal_rollout_overlap_count"] == 2
    assert len(test_predictions) == 2
    assert result["overlap_count"] == 2


def test_promote_seasonal_rollout_as_test_ignores_nonfinite_predictions(tmp_path, dummy_logger) -> None:
    pd.DataFrame(
        [
            {"split": "val", "loss": 0.04, "rmse": 0.2, "mae": 0.2, "r2": 0.1, "bias": 0.0, "negative_count": 0.0},
            {"split": "test", "loss": 99.0, "rmse": 9.9, "mae": 9.9, "r2": -9.0, "bias": 9.9, "negative_count": 0.0},
        ]
    ).to_csv(tmp_path / "metrics.csv", index=False)
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"best_epoch": 3, "best_val_rmse": 0.2, "final_val_loss": 0.04, "final_test_loss": 99.0}),
        encoding="utf-8",
    )

    overlap_df = pd.DataFrame(
        [
            {"lake_name": "Xiaoxingkai", "sample_datetime": "2026-01-01 12:00:00", "y_true": 0.20, "y_pred": 0.22},
            {"lake_name": "Xiaoxingkai", "sample_datetime": "2026-01-03 12:00:00", "y_true": 0.30, "y_pred": float("nan")},
        ]
    )
    overlap_predictions_path = tmp_path / "seasonal_rollout_overlap_predictions.csv"
    overlap_df.to_csv(overlap_predictions_path, index=False)

    overlap_metrics_json_path = tmp_path / "seasonal_rollout_overlap_metrics.json"
    overlap_metrics_json_path.write_text(
        json.dumps(
            {
                "count": 1,
                "observed_count": 2,
                "invalid_prediction_count": 1,
                "rollout_rows": 90,
                "test_start_datetime": "2025-10-01 12:00:00",
                "rollout_start_datetime": "2025-10-01 12:00:00",
                "rollout_end_datetime": "2026-03-01 12:00:00",
                "overlap_start_datetime": "2026-01-01 12:00:00",
                "overlap_end_datetime": "2026-01-03 12:00:00",
                "rmse": 0.02,
                "mae": 0.02,
                "r2": 0.0,
                "bias": 0.02,
                "negative_count": 0.0,
            }
        ),
        encoding="utf-8",
    )

    artifacts = SimpleNamespace(
        overlap_predictions_path=overlap_predictions_path,
        overlap_metrics_json_path=overlap_metrics_json_path,
    )
    result = _promote_seasonal_rollout_as_test(tmp_path, artifacts, dummy_logger)

    metrics_df = pd.read_csv(tmp_path / "metrics.csv")
    test_row = metrics_df.loc[metrics_df["split"] == "test"].iloc[0]
    run_summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    test_predictions = pd.read_csv(tmp_path / "test_predictions.csv")

    assert float(test_row["rmse"]) == pytest.approx(0.02)
    assert float(test_row["loss"]) == pytest.approx(0.0004)
    assert float(test_row["r2"]) == pytest.approx(0.0)
    assert run_summary["seasonal_rollout_overlap_count"] == 1
    assert len(test_predictions) == 2
    assert result["overlap_count"] == 1
