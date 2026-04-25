from __future__ import annotations

import pandas as pd

from lakeice_ncde.evaluation.seasonal_rollout import (
    _apply_open_water_projection,
    _resolve_rollout_initial_state,
    build_seasonal_rollout_dataframe,
)


def test_build_seasonal_rollout_dataframe_resolves_era5_path_from_project_root(tmp_path) -> None:
    era5_path = tmp_path / "data" / "raw" / "era5.csv"
    era5_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "datetime": ["2025-09-30 12:00:00", "2025-10-01 12:00:00", "2025-10-02 12:00:00"],
            "Air_Temperature_celsius": [-4.0, -5.0, -6.0],
        }
    ).to_csv(era5_path, index=False)

    prepared_df = pd.DataFrame(
        {
            "lake_name": ["Xiaoxingkai"],
            "sample_datetime": ["2025-10-01 12:00:00"],
            "total_ice_m": [0.1],
            "latitude": [45.0],
            "longitude": [132.0],
        }
    )

    config = {
        "runtime": {"project_root": str(tmp_path)},
        "data": {
            "datetime_column": "sample_datetime",
            "target_column": "total_ice_m",
            "lake_column": "lake_name",
        },
        "seasonal_rollout": {
            "era5_csv": "data/raw/era5.csv",
            "target_lake_name": "Xiaoxingkai",
            "test_start_datetime": "2025-10-01 12:00:00",
            "end_datetime": "2025-10-02 12:00:00",
            "daily_hour": 12,
        },
    }

    rollout_df = build_seasonal_rollout_dataframe(config, prepared_df)
    assert len(rollout_df) == 2
    assert rollout_df["lake_name"].nunique() == 1
    assert str(rollout_df["sample_datetime"].min()) == "2025-10-01 12:00:00"


def test_rollout_initial_state_resets_to_open_water_from_configured_month() -> None:
    prepared_df = pd.DataFrame(
        {
            "lake_name": ["Xiaoxingkai"],
            "sample_datetime": ["2025-02-17 12:00:00"],
            "total_ice_m": [0.7122],
        }
    )
    start_time = pd.Timestamp("2025-09-12 12:00:00")

    previous_ice, previous_time = _resolve_rollout_initial_state(
        prepared_df=prepared_df,
        lake_column="lake_name",
        target_column="total_ice_m",
        time_column="sample_datetime",
        target_lake="Xiaoxingkai",
        start_time=start_time,
        reset_from_month=8,
    )

    assert previous_ice == 0.0
    assert previous_time == start_time


def test_open_water_projection_keeps_warm_open_water_at_zero() -> None:
    projected = _apply_open_water_projection(
        pred_value=9.9,
        previous_ice=0.0,
        anchor_row=pd.Series({"Air_Temperature_celsius": 20.0}),
        anchor_time=pd.Timestamp("2025-09-25 12:00:00"),
        rollout_cfg={
            "reset_initial_state_from_month": 8,
            "open_water_projection_enabled": True,
            "open_water_temperature_column": "Air_Temperature_celsius",
            "open_water_temperature_threshold_celsius": 0.0,
            "open_water_prev_ice_max_m": 0.05,
        },
    )

    assert projected == 0.0


def test_open_water_projection_keeps_established_ice_prediction() -> None:
    projected = _apply_open_water_projection(
        pred_value=0.6,
        previous_ice=0.4,
        anchor_row=pd.Series({"Air_Temperature_celsius": 1.5}),
        anchor_time=pd.Timestamp("2025-12-10 12:00:00"),
        rollout_cfg={
            "reset_initial_state_from_month": 8,
            "open_water_projection_enabled": True,
            "open_water_temperature_column": "Air_Temperature_celsius",
            "open_water_temperature_threshold_celsius": 0.0,
            "open_water_prev_ice_max_m": 0.05,
        },
    )

    assert projected == 0.6
