from __future__ import annotations

import pandas as pd

from lakeice_ncde.evaluation.seasonal_rollout import build_seasonal_rollout_dataframe


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
