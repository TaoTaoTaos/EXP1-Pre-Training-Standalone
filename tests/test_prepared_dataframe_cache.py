from __future__ import annotations

import json

import pandas as pd

from lakeice_ncde.pipeline import load_or_prepare_dataframe
from lakeice_ncde.utils.paths import ProjectPaths


class _DummyLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message, *args) -> None:
        self.messages.append(str(message) % args if args else str(message))

    def warning(self, message, *args) -> None:
        self.messages.append(str(message) % args if args else str(message))


def _build_config(*, season_start_month: int) -> dict:
    return {
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
        "train": {
            "physics_loss": {
                "enabled": True,
                "mode": "tc2020_curve",
                "temperature_column": "Air_Temperature_celsius",
                "afdd_column": "afdd",
                "atdd_column": "atdd",
                "growth_phase_column": "is_growth_phase",
                "decay_phase_column": "is_decay_phase",
                "stable_ice_mask_column": "stable_ice_mask",
                "season_start_month": season_start_month,
                "stable_ice_min_m": 0.03,
                "phase_tolerance_m": 1.0e-3,
            }
        },
    }


def _build_paths(tmp_path) -> ProjectPaths:
    return ProjectPaths(
        project_root=tmp_path,
        raw_excel=tmp_path / "raw.xlsx",
        prepared_csv=tmp_path / "prepared.csv",
        validation_report_json=tmp_path / "validation.json",
        feature_schema_json=tmp_path / "schema.json",
        split_root=tmp_path / "splits",
        window_root=tmp_path / "windows",
        coeff_root=tmp_path / "coeffs",
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
    )


def test_load_or_prepare_dataframe_rebuilds_when_tc2020_season_start_month_changes(
    tmp_path,
    monkeypatch,
) -> None:
    raw_df = pd.DataFrame(
        [
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-10-05 00:00:00",
                "doy": 278,
                "total_ice_m": 0.00,
                "Air_Temperature_celsius": -1.0,
            },
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-10-25 00:00:00",
                "doy": 298,
                "total_ice_m": 0.04,
                "Air_Temperature_celsius": -2.0,
            },
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-11-05 00:00:00",
                "doy": 309,
                "total_ice_m": 0.08,
                "Air_Temperature_celsius": -2.0,
            },
        ]
    )
    load_calls = {"count": 0}

    def fake_load_raw_excel(raw_excel_path, sheet_name=None):
        del raw_excel_path, sheet_name
        load_calls["count"] += 1
        return raw_df.copy()

    monkeypatch.setattr("lakeice_ncde.pipeline.load_raw_excel", fake_load_raw_excel)

    logger = _DummyLogger()
    paths = _build_paths(tmp_path)

    first_df = load_or_prepare_dataframe(_build_config(season_start_month=10), paths, logger)
    assert load_calls["count"] == 1
    assert first_df["afdd"].tolist() == [0.0, 40.0, 62.0]

    metadata_path = tmp_path / "prepared_metadata.json"
    first_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert first_metadata["tc2020_curve_preprocessing"]["season_start_month"] == 10

    second_df = load_or_prepare_dataframe(_build_config(season_start_month=11), paths, logger)
    assert load_calls["count"] == 2
    assert second_df["afdd"].tolist() == [0.0, 40.0, 22.0]

    second_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert second_metadata["tc2020_curve_preprocessing"]["season_start_month"] == 11
    assert any("metadata is missing or stale" in message for message in logger.messages)
