from __future__ import annotations

from pathlib import Path

import pandas as pd

from lakeice_ncde.pipeline import plot_from_run
from lakeice_ncde.utils.io import save_dataframe, save_json, save_yaml
from lakeice_ncde.visualization.pdf_report import _collect_report_data


def _write_minimal_report_run(tmp_path: Path) -> tuple[Path, Path]:
    raw_excel = tmp_path / "data" / "raw" / "lakeice.xlsx"
    raw_excel.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "lake_name": "Lake Xiaoxingkai",
                "sample_datetime": "2025-12-28 12:00:00",
                "doy": 362,
                "total_ice_m": 0.10,
                "Air_Temperature_celsius": -12.0,
            },
            {
                "lake_name": "Lake Xiaoxingkai",
                "sample_datetime": "2025-12-29 12:00:00",
                "doy": 363,
                "total_ice_m": 0.12,
                "Air_Temperature_celsius": -11.0,
            },
            {
                "lake_name": "Lake Xiaoxingkai",
                "sample_datetime": "2026-01-01 12:00:00",
                "doy": 1,
                "total_ice_m": 0.16,
                "Air_Temperature_celsius": -10.0,
            },
            {
                "lake_name": "Lake Xiaoxingkai",
                "sample_datetime": "2026-01-02 12:00:00",
                "doy": 2,
                "total_ice_m": 0.17,
                "Air_Temperature_celsius": -9.0,
            },
            {
                "lake_name": "Lake Other",
                "sample_datetime": "2025-12-28 12:00:00",
                "doy": 362,
                "total_ice_m": 0.08,
                "Air_Temperature_celsius": -13.0,
            },
            {
                "lake_name": "Lake Other",
                "sample_datetime": "2025-12-29 12:00:00",
                "doy": 363,
                "total_ice_m": 0.09,
                "Air_Temperature_celsius": -12.0,
            },
            {
                "lake_name": "Lake Other",
                "sample_datetime": "2026-01-01 12:00:00",
                "doy": 1,
                "total_ice_m": 0.11,
                "Air_Temperature_celsius": -11.0,
            },
            {
                "lake_name": "Lake Other",
                "sample_datetime": "2026-01-02 12:00:00",
                "doy": 2,
                "total_ice_m": 0.13,
                "Air_Temperature_celsius": -10.0,
            },
        ]
    ).to_excel(raw_excel, index=False, sheet_name="lakeice_era5")

    run_dir = tmp_path / "outputs" / "runs" / "EXP2-B-tc2020" / "[01]_EXP2-B-tc2020_20260419_141138"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(
        {
            "paths": {
                "raw_excel": "data/raw/lakeice.xlsx",
                "prepared_csv": "data/interim/missing_prepared.csv",
            },
            "data": {
                "excel_sheet_name": "lakeice_era5",
                "lake_column": "lake_name",
                "datetime_column": "sample_datetime",
                "target_column": "total_ice_m",
                "doy_column": "doy",
                "include_lakes": None,
            },
            "features": {
                "time_channel_name": "relative_time",
                "feature_columns": ["Air_Temperature_celsius"],
                "target_transform": "none",
            },
            "window": {"window_days": 7},
            "custom_split": {
                "val_fraction": 0.25,
                "min_val_rows_per_lake": 1,
                "min_train_rows_per_lake": 1,
                "target_lake_test_start": "2026-01-01 12:00:00",
            },
            "experiment": {"name": "EXP2-B-tc2020"},
            "runtime": {"project_root": str(tmp_path)},
        },
        run_dir / "config_merged.yaml",
    )
    save_json({}, artifacts_dir / "run_manifest.json")
    save_dataframe(
        pd.DataFrame(
            [
                {"split": "val", "loss": 0.01, "rmse": 0.10, "mae": 0.08, "r2": 0.80, "bias": 0.00, "negative_count": 0},
                {"split": "test", "loss": 0.02, "rmse": 0.12, "mae": 0.09, "r2": 0.70, "bias": 0.01, "negative_count": 0},
            ]
        ),
        run_dir / "metrics.csv",
    )
    save_dataframe(
        pd.DataFrame(
            [
                {"epoch": 1, "train_loss": 0.20, "val_loss": 0.10, "val_rmse": 0.12, "val_mae": 0.09, "val_r2": 0.70, "lr": 2.0e-4},
                {"epoch": 2, "train_loss": 0.10, "val_loss": 0.08, "val_rmse": 0.10, "val_mae": 0.08, "val_r2": 0.80, "lr": 2.0e-4},
            ]
        ),
        run_dir / "epoch_summary.csv",
    )
    prediction_rows = [
        {"lake_name": "Lake Xiaoxingkai", "sample_datetime": "2026-01-01 12:00:00", "y_true": 0.16, "y_pred": 0.15},
        {"lake_name": "Lake Xiaoxingkai", "sample_datetime": "2026-01-02 12:00:00", "y_true": 0.17, "y_pred": 0.16},
    ]
    save_dataframe(pd.DataFrame(prediction_rows), run_dir / "val_predictions.csv")
    save_dataframe(pd.DataFrame(prediction_rows), run_dir / "test_predictions.csv")
    save_dataframe(pd.DataFrame(prediction_rows), run_dir / "per_lake_metrics.csv")
    save_json(
        {
            "best_epoch": 2,
            "best_val_rmse": 0.10,
            "final_val_loss": 0.08,
            "final_test_loss": 0.02,
        },
        run_dir / "run_summary.json",
    )
    return run_dir, raw_excel


def test_collect_report_data_rebuilds_missing_prepared_csv(tmp_path: Path) -> None:
    run_dir, raw_excel = _write_minimal_report_run(tmp_path)

    report_data = _collect_report_data(run_dir)

    assert not (tmp_path / "data" / "interim" / "missing_prepared.csv").exists()
    assert report_data["raw_excel"] == raw_excel.resolve()
    assert sorted(report_data["row_stats"]["lake_name"].tolist()) == ["Lake Other", "Lake Xiaoxingkai"]
    assert report_data["totals"]["source_rows"] == 8
    assert report_data["totals"]["test_rows"] == 2


def test_plot_from_run_does_not_raise_when_pdf_build_fails(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "EXP2-B-tc2020" / "[01]_EXP2-B-tc2020_20260419_141138"
    run_dir.mkdir(parents=True, exist_ok=True)

    class DummyLogger:
        def __init__(self) -> None:
            self.exception_calls: list[str] = []
            self.info_calls: list[str] = []

        def exception(self, message: str, *args) -> None:
            self.exception_calls.append(message % args)

        def info(self, message: str, *args) -> None:
            self.info_calls.append(message % args)

    logger = DummyLogger()

    def _raise(*_args, **_kwargs) -> None:
        raise FileNotFoundError("missing prepared dataframe")

    monkeypatch.setattr("lakeice_ncde.pipeline.build_pdf_report", _raise)

    plot_from_run(run_dir, logger)

    assert logger.exception_calls == [f"Failed to build PDF report for {run_dir}"]
    assert not logger.info_calls
