from __future__ import annotations

import pandas as pd
import pytest

from lakeice_ncde.data.load_excel import standardize_dataframe


def test_standardize_dataframe_adds_tc2020_curve_columns() -> None:
    config = {
        "data": {
            "lake_column": "lake_name",
            "datetime_column": "sample_datetime",
            "target_column": "total_ice_m",
            "doy_column": "doy",
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
                "season_start_month": 7,
                "stable_ice_min_m": 0.03,
                "phase_tolerance_m": 1.0e-3,
            }
        },
    }
    raw_df = pd.DataFrame(
        [
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-11-01 00:00:00",
                "doy": 305,
                "total_ice_m": 0.00,
                "Air_Temperature_celsius": -2.0,
            },
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-11-03 00:00:00",
                "doy": 307,
                "total_ice_m": 0.04,
                "Air_Temperature_celsius": -4.0,
            },
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-11-05 00:00:00",
                "doy": 309,
                "total_ice_m": 0.08,
                "Air_Temperature_celsius": -1.0,
            },
            {
                "lake_name": "Lake A",
                "sample_datetime": "2026-02-01 00:00:00",
                "doy": 32,
                "total_ice_m": 0.06,
                "Air_Temperature_celsius": 3.0,
            },
        ]
    )

    prepared_df, schema = standardize_dataframe(raw_df, config)

    assert schema.feature_columns == ["Air_Temperature_celsius"]
    assert {"afdd", "atdd", "is_growth_phase", "is_decay_phase", "stable_ice_mask"}.issubset(
        prepared_df.columns
    )
    assert prepared_df["afdd"].tolist() == [0.0, 8.0, 10.0, 10.0]
    assert prepared_df["atdd"].tolist() == [0.0, 0.0, 0.0, 264.0]
    assert prepared_df["is_growth_phase"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert prepared_df["is_decay_phase"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert prepared_df["stable_ice_mask"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_standardize_dataframe_requires_explicit_tc2020_preprocessing_config() -> None:
    config = {
        "data": {
            "lake_column": "lake_name",
            "datetime_column": "sample_datetime",
            "target_column": "total_ice_m",
            "doy_column": "doy",
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
                "stable_ice_min_m": 0.03,
                "phase_tolerance_m": 1.0e-3,
            }
        },
    }
    raw_df = pd.DataFrame(
        [
            {
                "lake_name": "Lake A",
                "sample_datetime": "2025-11-01 00:00:00",
                "doy": 305,
                "total_ice_m": 0.00,
                "Air_Temperature_celsius": -2.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="season_start_month"):
        standardize_dataframe(raw_df, config)
