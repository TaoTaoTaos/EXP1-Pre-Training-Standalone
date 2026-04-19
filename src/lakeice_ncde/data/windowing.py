from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lakeice_ncde.data.scaling import apply_feature_scaler, fit_feature_scaler, transform_target
from lakeice_ncde.utils.io import save_dataframe, save_torch, save_yaml


@dataclass
class WindowBundlePaths:
    """Paths for a saved window bundle."""

    bundle_path: Path
    metadata_path: Path
    manifest_path: Path
    scaler_path: Path


def select_debug_lakes(
    df: pd.DataFrame,
    lake_column: str,
    max_lakes: int | None,
    split_column: str | None = None,
) -> pd.DataFrame:
    """Restrict a dataframe to a small number of lakes for fast debugging."""
    if max_lakes is None:
        return df
    keep_lakes: list[str] = []
    if split_column is not None and split_column in df.columns:
        for split_name in ("train", "val", "test"):
            candidates = (
                df.loc[df[split_column] == split_name, lake_column].dropna().astype(str).drop_duplicates().tolist()
            )
            if candidates and len(keep_lakes) < max_lakes:
                keep_lakes.append(sorted(candidates)[0])
        if len(keep_lakes) < max_lakes:
            remaining = sorted(
                set(df[lake_column].dropna().astype(str).unique().tolist()) - set(keep_lakes)
            )
            keep_lakes.extend(remaining[: max_lakes - len(keep_lakes)])
    else:
        keep_lakes = sorted(df[lake_column].dropna().astype(str).unique().tolist())[:max_lakes]
    return df.loc[df[lake_column].isin(keep_lakes)].copy()


def _build_single_window(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    time_column: str,
    target_column: str,
    window_days: int,
    lake_name: str,
    anchor_index: int,
) -> dict[str, Any] | None:
    anchor_row = history_df.iloc[anchor_index]
    anchor_time = anchor_row[time_column]
    start_time = anchor_time - pd.Timedelta(days=window_days)
    window_df = history_df.loc[(history_df[time_column] >= start_time) & (history_df[time_column] <= anchor_time)].copy()
    if len(window_df) < 2:
        return None

    diffs = window_df[time_column].diff().dropna().dt.total_seconds().to_numpy()
    if len(diffs) and np.any(diffs <= 0):
        raise ValueError(
            f"Window time must be strictly increasing. lake={lake_name}, anchor_index={anchor_index}, diffs={diffs[:10]}"
        )

    elapsed_days = (
        window_df[time_column] - window_df[time_column].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=np.float32) / 86400.0
    relative_time = elapsed_days / float(window_days)

    features = window_df[feature_columns].to_numpy(dtype=np.float32)
    path = np.concatenate([relative_time[:, None], features], axis=1)
    target_value = float(anchor_row[target_column])
    if target_value < 0:
        raise ValueError(f"Negative target encountered. lake={lake_name}, target={target_value}")

    return {
        "path": torch.tensor(path, dtype=torch.float32),
        "target": target_value,
        "target_datetime": anchor_time,
        "length": int(path.shape[0]),
        "window_days": int(window_days),
    }


def build_window_bundles(
    df: pd.DataFrame,
    assignments: dict[str, str],
    config: dict,
    split_name: str,
    window_root: Path,
    logger=None,
) -> dict[str, WindowBundlePaths]:
    """Build and save irregular windows for each split."""
    data_cfg = config["data"]
    feature_cfg = config["features"]
    debug_cfg = config.get("debug", {})
    window_cfg = config["window"]

    lake_column = data_cfg["lake_column"]
    time_column = data_cfg["datetime_column"]
    target_column = data_cfg["target_column"]
    feature_columns = feature_cfg["feature_columns"]

    assigned_df = df.copy()
    assigned_df["split"] = assigned_df[lake_column].astype(str).map(assignments)
    assigned_df = assigned_df.dropna(subset=["split"]).reset_index(drop=True)
    assigned_df = select_debug_lakes(assigned_df, lake_column, debug_cfg.get("max_lakes"), split_column="split")

    train_df = assigned_df.loc[assigned_df["split"] == "train"].copy()
    scaler = fit_feature_scaler(
        train_df=train_df,
        feature_columns=feature_columns,
        target_transform=feature_cfg["target_transform"],
        target_column=target_column,
    )
    scaled_df = apply_feature_scaler(assigned_df, scaler)

    split_dir = window_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, WindowBundlePaths] = {}
    for current_split in ("train", "val", "test"):
        split_df = scaled_df.loc[scaled_df["split"] == current_split].copy()
        lake_groups = list(split_df.groupby(lake_column))
        if logger is not None:
            logger.info(
                "Building %s windows for split '%s': %d rows across %d lakes.",
                current_split,
                split_name,
                len(split_df),
                len(lake_groups),
            )
        windows: list[torch.Tensor] = []
        targets: list[float] = []
        transformed_targets: list[float] = []
        metadata_rows: list[dict[str, Any]] = []

        for lake_index, (lake_name, lake_df) in enumerate(lake_groups, start=1):
            lake_df = lake_df.sort_values(time_column).reset_index(drop=True)
            for anchor_index in range(len(lake_df)):
                built = _build_single_window(
                    history_df=lake_df,
                    feature_columns=feature_columns,
                    time_column=time_column,
                    target_column=target_column,
                    window_days=int(window_cfg["window_days"]),
                    lake_name=str(lake_name),
                    anchor_index=anchor_index,
                )
                if built is None:
                    continue
                windows.append(built["path"])
                targets.append(float(built["target"]))
                transformed_targets.append(float(transform_target(np.array([built["target"]], dtype=np.float32), feature_cfg["target_transform"])[0]))
                metadata_rows.append(
                    {
                        "window_id": f"{current_split}_{len(metadata_rows):06d}",
                        "split": current_split,
                        "lake_name": str(lake_name),
                        "target_datetime": built["target_datetime"],
                        "length": built["length"],
                        "window_days": built["window_days"],
                        "target_raw": float(built["target"]),
                        "target_transformed": transformed_targets[-1],
                    }
                )
            if logger is not None and (
                lake_index == len(lake_groups)
                or lake_index == 1
                or lake_index % max(1, len(lake_groups) // 5) == 0
            ):
                logger.info(
                    "Window progress for split '%s': %d/%d lakes processed, %d windows built.",
                    current_split,
                    lake_index,
                    len(lake_groups),
                    len(metadata_rows),
                )

        max_windows = debug_cfg.get("max_windows_per_split")
        if max_windows is not None:
            windows = windows[: max_windows]
            targets = targets[: max_windows]
            transformed_targets = transformed_targets[: max_windows]
            metadata_rows = metadata_rows[: max_windows]

        bundle = {
            "windows": windows,
            "targets_raw": torch.tensor(targets, dtype=torch.float32),
            "targets_transformed": torch.tensor(transformed_targets, dtype=torch.float32),
            "metadata": metadata_rows,
            "feature_columns": feature_columns,
            "input_channels": [feature_cfg["time_channel_name"], *feature_columns],
            "target_column": target_column,
            "target_transform": feature_cfg["target_transform"],
            "split_name": split_name,
            "split": current_split,
        }

        bundle_path = split_dir / f"{current_split}_windows.pt"
        metadata_path = split_dir / f"{current_split}_windows_metadata.csv"
        manifest_path = split_dir / f"{current_split}_windows_manifest.yaml"
        scaler_path = split_dir / "feature_scaler.yaml"

        save_torch(bundle, bundle_path)
        metadata_df = pd.DataFrame(metadata_rows)
        save_dataframe(metadata_df, metadata_path)
        save_yaml(
            {
                "split_name": split_name,
                "split": current_split,
                "bundle_path": str(bundle_path),
                "metadata_path": str(metadata_path),
                "feature_scaler_path": str(scaler_path),
                "count": len(metadata_rows),
            },
            manifest_path,
        )
        if not scaler_path.exists():
            save_yaml(scaler.to_dict(), scaler_path)

        outputs[current_split] = WindowBundlePaths(
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            scaler_path=scaler_path,
        )
        if logger is not None:
            logger.info(
                "Saved %s windows for split '%s': %d samples -> %s",
                current_split,
                split_name,
                len(metadata_rows),
                bundle_path,
            )

    return outputs
