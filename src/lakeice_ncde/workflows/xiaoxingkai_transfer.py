from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lakeice_ncde.data.coeffs import compute_coefficients_for_windows, save_coeff_bundle
from lakeice_ncde.data.datasets import create_dataloader
from lakeice_ncde.data.load_excel import (
    resolve_required_physics_columns,
    resolve_tc2020_curve_preprocessing_config,
)
from lakeice_ncde.data.scaling import apply_feature_scaler, fit_feature_scaler, transform_target
from lakeice_ncde.data.windowing import _build_single_window
from lakeice_ncde.evaluation.per_lake_summary import compute_per_lake_metrics
from lakeice_ncde.evaluation.seasonal_rollout import run_seasonal_rollout
from lakeice_ncde.experiment.registry import append_experiment_registry
from lakeice_ncde.experiment.tracker import create_run_context
from lakeice_ncde.models.neural_cde import build_model
from lakeice_ncde.pipeline import load_or_prepare_dataframe, plot_from_run, validate_and_save
from lakeice_ncde.training.engine import Trainer
from lakeice_ncde.utils.io import load_dataframe, save_dataframe, save_json, save_torch, save_yaml
from lakeice_ncde.utils.locking import PathLock
from lakeice_ncde.utils.logging import setup_logging
from lakeice_ncde.utils.seed import set_seed


XIAOXINGKAI_NAME = "【Li】Lake Xiaoxingkai"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    target_lake: str
    train_lakes: list[str]


def _time_split_training_lake(
    lake_df: pd.DataFrame,
    val_fraction: float,
    min_val_rows: int,
    min_train_rows: int,
) -> pd.DataFrame:
    lake_df = lake_df.sort_values("sample_datetime").reset_index(drop=True).copy()
    total_rows = len(lake_df)
    proposed_val_rows = max(min_val_rows, int(np.ceil(total_rows * val_fraction)))
    val_rows = min(proposed_val_rows, max(1, total_rows - min_train_rows))
    if total_rows - val_rows < min_train_rows:
        raise ValueError(
            f"Unable to create temporal validation split for {lake_df['lake_name'].iloc[0]}: "
            f"total_rows={total_rows}, val_rows={val_rows}, min_train_rows={min_train_rows}."
        )
    split_index = total_rows - val_rows
    lake_df["row_split"] = "train"
    lake_df.loc[split_index:, "row_split"] = "val"
    return lake_df


def _build_fold_dataframe(df: pd.DataFrame, fold: FoldSpec, config: dict) -> pd.DataFrame:
    custom_cfg = config["custom_split"]
    pieces: list[pd.DataFrame] = []
    for lake_name in fold.train_lakes:
        lake_df = df.loc[df[config["data"]["lake_column"]] == lake_name].copy()
        pieces.append(
            _time_split_training_lake(
                lake_df=lake_df,
                val_fraction=float(custom_cfg["val_fraction"]),
                min_val_rows=int(custom_cfg["min_val_rows_per_lake"]),
                min_train_rows=int(custom_cfg["min_train_rows_per_lake"]),
            )
        )
    target_df = df.loc[df[config["data"]["lake_column"]] == fold.target_lake].copy()
    target_df = target_df.sort_values(config["data"]["datetime_column"]).reset_index(drop=True)
    target_test_start = custom_cfg.get("target_lake_test_start")
    if target_test_start:
        cutoff = pd.Timestamp(target_test_start)
        target_history_df = target_df.loc[pd.to_datetime(target_df[config["data"]["datetime_column"]]) < cutoff].copy()
        target_test_df = target_df.loc[pd.to_datetime(target_df[config["data"]["datetime_column"]]) >= cutoff].copy()
        target_history_df = _time_split_training_lake(
            lake_df=target_history_df,
            val_fraction=float(custom_cfg["val_fraction"]),
            min_val_rows=int(custom_cfg["min_val_rows_per_lake"]),
            min_train_rows=int(custom_cfg["min_train_rows_per_lake"]),
        )
        target_test_df["row_split"] = "test"
        pieces.append(target_history_df)
        pieces.append(target_test_df)
    else:
        target_df["row_split"] = "test"
        pieces.append(target_df)
    return pd.concat(pieces, ignore_index=True)


def _save_fold_manifest(fold_df: pd.DataFrame, fold: FoldSpec, split_root: Path, lake_column: str) -> Path:
    split_dir = split_root / fold.name
    split_dir.mkdir(parents=True, exist_ok=True)

    assignment_rows: list[dict[str, Any]] = []
    for lake_name, lake_df in fold_df.groupby(lake_column):
        split_names = sorted(lake_df["row_split"].astype(str).unique().tolist())
        assignment_rows.append({"lake_name": str(lake_name), "row_splits": ",".join(split_names)})
    assignments_path = split_dir / "lake_assignments.csv"
    save_dataframe(pd.DataFrame(assignment_rows), assignments_path)

    split_lakes = {
        split_name: sorted(
            fold_df.loc[fold_df["row_split"] == split_name, lake_column].dropna().astype(str).unique().tolist()
        )
        for split_name in ("train", "val", "test")
    }
    manifest = {
        "split_name": fold.name,
        "split_type": "target_lake_temporal_holdout" if "test" in fold_df["row_split"].values else "target_lake_temporal_validation",
        "target_lake": fold.target_lake,
        "train_lakes": fold.train_lakes,
        "splits": split_lakes,
        "assignment_file": str(assignments_path),
    }
    manifest_path = split_dir / "split_manifest.yaml"
    save_yaml(manifest, manifest_path)
    return manifest_path


def _build_window_cache_key(config: dict) -> str:
    physics_cfg = config["train"].get("physics_loss", {})
    physics_mode = str(physics_cfg.get("mode", "legacy_stefan"))
    physics_cache_controls: dict[str, Any] = {}
    if physics_mode == "tc2020_curve":
        physics_cache_controls = resolve_tc2020_curve_preprocessing_config(config)
    payload = {
        "experiment_name": config["experiment"]["name"],
        "raw_excel": config["paths"]["raw_excel"],
        "data": config["data"],
        "features": config["features"],
        "window": config["window"],
        "custom_split": config["custom_split"],
        "physics_mode": physics_mode,
        "physics_fields": resolve_required_physics_columns(config),
        "physics_cache_controls": physics_cache_controls,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def _shared_window_split_dir(window_root: Path, split_name: str, config: dict) -> Path:
    cache_key = _build_window_cache_key(config)
    return window_root / "_shared" / cache_key / split_name


def _shared_window_cache_ready(split_dir: Path) -> bool:
    scaler_path = split_dir / "feature_scaler.yaml"
    return scaler_path.exists() and all(
        (split_dir / f"{current_split}_windows.pt").exists()
        for current_split in ("train", "val")
    )


def _ensure_shared_window_bundles(
    fold_df: pd.DataFrame,
    config: dict,
    split_name: str,
    window_root: Path,
    logger,
) -> tuple[dict[str, Path], Path]:
    split_dir = _shared_window_split_dir(window_root, split_name, config)
    scaler_path = split_dir / "feature_scaler.yaml"
    bundle_paths = {
        current_split: split_dir / f"{current_split}_windows.pt"
        for current_split in ("train", "val")
    }
    if _shared_window_cache_ready(split_dir):
        logger.info("Reusing shared window cache for split '%s': %s", split_name, split_dir)
        return bundle_paths, scaler_path

    lock_path = split_dir / ".build.lock"
    with PathLock(lock_path):
        if _shared_window_cache_ready(split_dir):
            logger.info("Reusing shared window cache for split '%s': %s", split_name, split_dir)
            return bundle_paths, scaler_path
        logger.info("Building shared window cache for split '%s': %s", split_name, split_dir)
        bundle_paths = _build_row_split_window_bundles(
            fold_df=fold_df,
            config=config,
            split_name=split_name,
            window_root=split_dir.parent,
            logger=logger,
        )
    return bundle_paths, scaler_path


def _materialize_runtime_window_bundles(
    shared_bundle_paths: dict[str, Path],
    shared_scaler_path: Path,
    run_dir: Path,
    split_name: str,
) -> tuple[dict[str, Path], Path]:
    runtime_split_dir = run_dir / "artifacts" / "runtime_cache" / "windows" / split_name
    runtime_split_dir.mkdir(parents=True, exist_ok=True)

    runtime_scaler_path = runtime_split_dir / "feature_scaler.yaml"
    shutil.copy2(shared_scaler_path, runtime_scaler_path)

    runtime_bundle_paths: dict[str, Path] = {}
    for current_split, shared_bundle_path in shared_bundle_paths.items():
        runtime_bundle_path = runtime_split_dir / shared_bundle_path.name
        shutil.copy2(shared_bundle_path, runtime_bundle_path)
        metadata_path = shared_bundle_path.parent / f"{current_split}_windows_metadata.csv"
        manifest_path = shared_bundle_path.parent / f"{current_split}_windows_manifest.yaml"
        if metadata_path.exists():
            shutil.copy2(metadata_path, runtime_split_dir / metadata_path.name)
        if manifest_path.exists():
            shutil.copy2(manifest_path, runtime_split_dir / manifest_path.name)
        runtime_bundle_paths[current_split] = runtime_bundle_path
    return runtime_bundle_paths, runtime_scaler_path


def _build_row_split_window_bundles(
    fold_df: pd.DataFrame,
    config: dict,
    split_name: str,
    window_root: Path,
    logger,
) -> dict[str, Path]:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    window_cfg = config["window"]
    lake_column = data_cfg["lake_column"]
    time_column = data_cfg["datetime_column"]
    target_column = data_cfg["target_column"]
    feature_columns = feature_cfg["feature_columns"]
    physics_cfg = config["train"].get("physics_loss", {})
    physics_field_columns = resolve_required_physics_columns(config)
    if physics_cfg.get("enabled", False):
        missing_columns = [
            column for column in physics_field_columns.values() if column not in fold_df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Physics loss requires the following prepared-data columns, but they are missing: "
                f"{missing_columns}"
            )

    train_rows = fold_df.loc[fold_df["row_split"] == "train"].copy()
    scaler = fit_feature_scaler(
        train_df=train_rows,
        feature_columns=feature_columns,
        target_transform=feature_cfg["target_transform"],
        target_column=target_column,
    )
    scaled_df = apply_feature_scaler(fold_df.copy(), scaler)

    split_dir = window_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = split_dir / "feature_scaler.yaml"
    save_yaml(scaler.to_dict(), scaler_path)

    bundle_paths: dict[str, Path] = {}
    for current_split in ("train", "val"):
        anchor_df = scaled_df.loc[scaled_df["row_split"] == current_split].copy()
        logger.info(
            "Building %s windows for split '%s': %d rows across %d lakes.",
            current_split,
            split_name,
            len(anchor_df),
            anchor_df[lake_column].nunique(),
        )
        windows: list[torch.Tensor] = []
        targets: list[float] = []
        transformed_targets: list[float] = []
        physics_rows: dict[str, list[float]] = {field_name: [] for field_name in physics_field_columns}
        metadata_rows: list[dict[str, Any]] = []

        lake_groups = list(anchor_df.groupby(lake_column))
        for lake_index, (lake_name, lake_anchor_df) in enumerate(lake_groups, start=1):
            lake_history_df = scaled_df.loc[scaled_df[lake_column] == lake_name].copy()
            lake_history_df = lake_history_df.sort_values(time_column).reset_index(drop=True)
            raw_lake_history_df = fold_df.loc[fold_df[lake_column] == lake_name].copy()
            raw_lake_history_df = raw_lake_history_df.sort_values(time_column).reset_index(drop=True)
            if len(lake_history_df) != len(raw_lake_history_df):
                raise ValueError(f"Scaled/raw lake history length mismatch for lake={lake_name}.")
            anchor_times = set(pd.to_datetime(lake_anchor_df[time_column]).tolist())

            for anchor_index in range(len(lake_history_df)):
                anchor_time = lake_history_df.iloc[anchor_index][time_column]
                if anchor_time not in anchor_times:
                    continue
                built = _build_single_window(
                    history_df=lake_history_df.iloc[: anchor_index + 1].copy(),
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
                transformed_targets.append(
                    float(transform_target(np.array([built["target"]], dtype=np.float32), feature_cfg["target_transform"])[0])
                )
                physics_values = {
                    field_name: float(
                        pd.to_numeric(raw_lake_history_df.iloc[anchor_index][column_name], errors="coerce")
                    )
                    for field_name, column_name in physics_field_columns.items()
                }
                for field_name, field_value in physics_values.items():
                    physics_rows[field_name].append(field_value)
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
                        **{field_name: field_value for field_name, field_value in physics_values.items()},
                    }
                )

            logger.info(
                "Window progress for split '%s': %d/%d lakes processed, %d windows built.",
                current_split,
                lake_index,
                len(lake_groups),
                len(metadata_rows),
            )

        bundle = {
            "windows": windows,
            "targets_raw": torch.tensor(targets, dtype=torch.float32),
            "targets_transformed": torch.tensor(transformed_targets, dtype=torch.float32),
            "physics_context": {
                field_name: torch.tensor(values, dtype=torch.float32) for field_name, values in physics_rows.items()
            },
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
        save_torch(bundle, bundle_path)
        save_dataframe(pd.DataFrame(metadata_rows), metadata_path)
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
        logger.info("Saved %s windows for split '%s': %d samples -> %s", current_split, split_name, len(metadata_rows), bundle_path)
        bundle_paths[current_split] = bundle_path
    return bundle_paths


def _subsample_bundle_by_lake(bundle_path: Path, max_windows_per_lake: int, seed: int) -> None:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    metadata = bundle["metadata"]
    if not metadata:
        return

    grouped_indices: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        grouped_indices.setdefault(str(row["lake_name"]), []).append(index)

    rng = random.Random(seed)
    keep_indices: list[int] = []
    for indices in grouped_indices.values():
        if len(indices) <= max_windows_per_lake:
            keep_indices.extend(indices)
            continue
        keep_indices.extend(sorted(rng.sample(indices, max_windows_per_lake)))

    keep_indices = sorted(keep_indices)
    bundle["windows"] = [bundle["windows"][index] for index in keep_indices]
    bundle["metadata"] = [bundle["metadata"][index] for index in keep_indices]
    bundle["targets_raw"] = bundle["targets_raw"][keep_indices]
    bundle["targets_transformed"] = bundle["targets_transformed"][keep_indices]
    physics_context = bundle.get("physics_context")
    if physics_context is not None:
        bundle["physics_context"] = {
            field_name: values[keep_indices]
            for field_name, values in physics_context.items()
        }
    save_torch(bundle, bundle_path)

    metadata_df = pd.DataFrame(bundle["metadata"])
    metadata_path = bundle_path.parent / f"{bundle['split']}_windows_metadata.csv"
    manifest_path = bundle_path.parent / f"{bundle['split']}_windows_manifest.yaml"
    save_dataframe(metadata_df, metadata_path)
    save_yaml(
        {
            "split_name": bundle["split_name"],
            "split": bundle["split"],
            "bundle_path": str(bundle_path),
            "metadata_path": str(metadata_path),
            "feature_scaler_path": str(bundle_path.parent / "feature_scaler.yaml"),
            "count": int(len(metadata_df)),
        },
        manifest_path,
    )


def _write_data_processing_report(
    run_dir: Path,
    validation_report: dict[str, Any],
    prepared_df: pd.DataFrame,
    config: dict[str, Any],
    validation_report_path: Path,
) -> Path:
    time_column = config["data"]["datetime_column"]
    lake_column = config["data"]["lake_column"]
    prepared_times = pd.to_datetime(prepared_df[time_column], errors="coerce") if not prepared_df.empty else pd.Series(dtype="datetime64[ns]")
    report = {
        "raw_validation_report_path": str(validation_report_path),
        "raw_excel_path": validation_report.get("raw_excel_path"),
        "raw_row_count": int(validation_report.get("row_count", 0)),
        "raw_unique_lakes": int(validation_report.get("unique_lakes", 0)),
        "invalid_sample_datetime": int(validation_report.get("invalid_sample_datetime", 0)),
        "missing_target": int(validation_report.get("missing_target", 0)),
        "negative_target_count": int(validation_report.get("negative_target_count", 0)),
        "prepared_row_count": int(len(prepared_df)),
        "prepared_unique_lakes": int(prepared_df[lake_column].nunique(dropna=True)),
        "prepared_datetime_min": None if prepared_times.empty else str(prepared_times.min()),
        "prepared_datetime_max": None if prepared_times.empty else str(prepared_times.max()),
        "rows_removed_from_raw_to_prepared": int(validation_report.get("row_count", 0) - len(prepared_df)),
        "prepared_csv_path": config["paths"]["prepared_csv"],
        "note": "prepared_row_count reflects all preprocessing effects, including configured lake filtering and dropping rows with invalid datetime or missing target.",
    }
    output_path = run_dir / "artifacts" / "data_processing_report.json"
    save_json(report, output_path)
    return output_path


def _write_experiment_summary(output_root: Path, experiment_name: str, fold_summary: dict[str, Any]) -> None:
    experiment_root = output_root / experiment_name
    summary_dir = experiment_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([fold_summary])
    save_dataframe(summary_df, summary_dir / f"{experiment_name}_summary.csv")
    save_json([fold_summary], summary_dir / f"{experiment_name}_summary.json")

    analysis_lines = [
        f"# {experiment_name} Summary",
        "",
        f"- Target lake: {fold_summary['target_lake']}",
        f"- Training lakes: {fold_summary['train_lakes']}",
        f"- Validation RMSE: {fold_summary['val_rmse']:.4f}",
        f"- Test RMSE: {fold_summary['test_rmse']:.4f}" if not np.isnan(fold_summary["test_rmse"]) else "- Test RMSE: n/a",
        f"- Test method: {fold_summary.get('test_method', 'n/a')}",
        f"- Seasonal rollout test start: {fold_summary.get('seasonal_test_start_datetime', 'n/a')}",
        "",
        "## Interpretation",
        "",
        "- Xiaoxingkai rows before the cutoff are split into train and validation by time order.",
        "- Xiaoxingkai rows from the cutoff onward are withheld from training and only used as observation checkpoints during seasonal-rollout testing.",
        "- Source lakes still use their own temporal train/val split.",
        "- The reported test metrics come from the overlap between the continuous autoregressive seasonal rollout and the observed Xiaoxingkai dates.",
    ]
    (summary_dir / f"{experiment_name}_analysis.md").write_text("\n".join(analysis_lines), encoding="utf-8")


def _promote_seasonal_rollout_as_test(
    run_dir: Path,
    seasonal_rollout_artifacts,
    logger,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    run_summary_path = run_dir / "run_summary.json"
    test_predictions_path = run_dir / "test_predictions.csv"
    per_lake_metrics_path = run_dir / "per_lake_metrics.csv"

    metrics_df = load_dataframe(metrics_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    metrics_df = metrics_df.loc[metrics_df["split"] != "test"].reset_index(drop=True)

    if seasonal_rollout_artifacts is None:
        test_row = {
            "split": "test",
            "loss": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "negative_count": np.nan,
        }
        rollout_metrics: dict[str, Any] = {}
        overlap_count = 0
        save_dataframe(pd.DataFrame(columns=["lake_name", "sample_datetime", "y_true", "y_pred"]), test_predictions_path)
        save_dataframe(pd.DataFrame(), per_lake_metrics_path)
    else:
        overlap_df = load_dataframe(
            seasonal_rollout_artifacts.overlap_predictions_path,
            parse_dates=["sample_datetime"],
        )
        rollout_metrics = json.loads(
            seasonal_rollout_artifacts.overlap_metrics_json_path.read_text(encoding="utf-8")
        )
        overlap_count = int(rollout_metrics.get("count", 0))
        if overlap_df.empty:
            test_row = {
                "split": "test",
                "loss": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "bias": np.nan,
                "negative_count": np.nan,
            }
            save_dataframe(overlap_df, test_predictions_path)
            save_dataframe(pd.DataFrame(), per_lake_metrics_path)
        else:
            residuals = overlap_df["y_pred"].to_numpy() - overlap_df["y_true"].to_numpy()
            test_loss = float(np.mean(np.square(residuals)))
            test_row = {
                "split": "test",
                "loss": test_loss,
                "rmse": float(rollout_metrics["rmse"]),
                "mae": float(rollout_metrics["mae"]),
                "r2": float(rollout_metrics["r2"]),
                "bias": float(rollout_metrics["bias"]),
                "negative_count": float(rollout_metrics["negative_count"]),
            }
            save_dataframe(overlap_df, test_predictions_path)
            save_dataframe(compute_per_lake_metrics(overlap_df), per_lake_metrics_path)

    metrics_df = pd.concat([metrics_df, pd.DataFrame([test_row])], ignore_index=True)
    save_dataframe(metrics_df, metrics_path)

    run_summary["final_test_loss"] = test_row["loss"]
    run_summary["test_method"] = "seasonal_rollout_overlap"
    run_summary["seasonal_test_start_datetime"] = rollout_metrics.get("test_start_datetime")
    run_summary["seasonal_rollout_start_datetime"] = rollout_metrics.get("rollout_start_datetime")
    run_summary["seasonal_rollout_end_datetime"] = rollout_metrics.get("rollout_end_datetime")
    run_summary["seasonal_rollout_overlap_start_datetime"] = rollout_metrics.get("overlap_start_datetime")
    run_summary["seasonal_rollout_overlap_end_datetime"] = rollout_metrics.get("overlap_end_datetime")
    run_summary["seasonal_rollout_overlap_count"] = overlap_count
    run_summary["seasonal_rollout_rows"] = int(rollout_metrics.get("rollout_rows", 0))
    save_json(run_summary, run_summary_path)
    logger.info(
        "Promoted seasonal rollout overlap as the only test metric source | overlap_count=%d | test_start=%s",
        overlap_count,
        run_summary.get("seasonal_test_start_datetime"),
    )
    return {
        "metrics": test_row,
        "overlap_count": overlap_count,
        "rollout_metrics": rollout_metrics,
    }


def run(config: dict, paths, base_logger) -> dict[str, Any]:
    """Run the standalone Xiaoxingkai transfer experiment end to end."""
    set_seed(int(config["train"]["seed"]))
    run_context = create_run_context(paths.output_root, config["experiment"]["name"], config)
    logger = setup_logging(run_context.log_path)
    logger.info("Starting experiment '%s'", config["experiment"]["name"])
    logger.info("Run directory: %s", run_context.run_dir)

    validation_report = validate_and_save(config, paths, logger)
    prepared_df = load_or_prepare_dataframe(config, paths, logger)
    data_processing_report_path = _write_data_processing_report(
        run_dir=run_context.run_dir,
        validation_report=validation_report,
        prepared_df=prepared_df,
        config=config,
        validation_report_path=paths.validation_report_json,
    )
    logger.info("Data processing report saved to %s", data_processing_report_path)
    if validation_report.get("invalid_sample_datetime", 0) or validation_report.get("missing_target", 0):
        logger.warning(
            "Raw data quality issues detected | invalid_sample_datetime=%d | missing_target=%d | validation_report=%s",
            int(validation_report.get("invalid_sample_datetime", 0)),
            int(validation_report.get("missing_target", 0)),
            paths.validation_report_json,
        )
    available_lakes = sorted(prepared_df[config["data"]["lake_column"]].dropna().astype(str).unique().tolist())
    target_lake = _resolve_xiaoxingkai_lake_name(available_lakes)
    fold = FoldSpec(
        name=str(config["experiment"]["name"]),
        target_lake=target_lake,
        train_lakes=[lake for lake in available_lakes if lake != target_lake],
    )
    logger.info("Running target transfer | target=%s | train=%s", fold.target_lake, fold.train_lakes)
    if config["custom_split"].get("target_lake_test_start"):
        logger.info(
            "Target lake test starts at %s; target rows before the cutoff are split into train and val.",
            config["custom_split"]["target_lake_test_start"],
        )
    logger.info(
        "Config summary | interpolation=%s | method=%s | window_days=%s | batch_size=%s | target_transform=%s",
        config["coeffs"]["interpolation"],
        config["model"]["method"],
        config["window"]["window_days"],
        config["train"]["batch_size"],
        config["features"]["target_transform"],
    )
    fold_df = _build_fold_dataframe(prepared_df, fold, config)
    manifest_path = _save_fold_manifest(
        fold_df,
        fold,
        run_context.artifacts_dir / "runtime_cache" / "splits",
        config["data"]["lake_column"],
    )

    for split_name in ("train", "val", "test"):
        split_lakes = sorted(
            fold_df.loc[fold_df["row_split"] == split_name, config["data"]["lake_column"]].dropna().astype(str).unique().tolist()
        )
        logger.info("Fold '%s' -> %s lakes (%d): %s", fold.name, split_name, len(split_lakes), split_lakes)

    shared_bundle_paths, scaler_path = _ensure_shared_window_bundles(
        fold_df=fold_df,
        config=config,
        split_name=fold.name,
        window_root=paths.window_root,
        logger=logger,
    )
    bundle_paths, scaler_path = _materialize_runtime_window_bundles(
        shared_bundle_paths=shared_bundle_paths,
        shared_scaler_path=scaler_path,
        run_dir=run_context.run_dir,
        split_name=fold.name,
    )
    custom_cfg = config["custom_split"]
    _subsample_bundle_by_lake(bundle_paths["train"], int(custom_cfg["max_train_windows_per_lake"]), int(config["train"]["seed"]))
    _subsample_bundle_by_lake(bundle_paths["val"], int(custom_cfg["max_val_windows_per_lake"]), int(config["train"]["seed"]) + 1)
    logger.info(
        "Balanced windows with train cap=%d and val cap=%d",
        int(custom_cfg["max_train_windows_per_lake"]),
        int(custom_cfg["max_val_windows_per_lake"]),
    )

    coeff_paths: dict[str, Path] = {}
    runtime_coeff_root = run_context.artifacts_dir / "runtime_cache" / "coeffs"
    for current_split, bundle_path in bundle_paths.items():
        coeff_bundle = compute_coefficients_for_windows(bundle_path, interpolation=config["coeffs"]["interpolation"], logger=logger)
        coeff_path, _, _ = save_coeff_bundle(coeff_bundle, runtime_coeff_root, split_name=fold.name, split=current_split)
        coeff_paths[current_split] = coeff_path
    logger.info("Saved %d coefficient bundle(s) under %s", len(coeff_paths), runtime_coeff_root)

    train_dataset, train_loader = create_dataloader(
        coeff_paths["train"],
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
        batch_parallel=bool(config["train"].get("batch_parallel", False)),
    )
    _, val_loader = create_dataloader(
        coeff_paths["val"],
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        batch_parallel=bool(config["train"].get("batch_parallel", False)),
    )
    test_loader = None
    test_dataset_size = 0
    logger.info(
        "Data summary | train=%d | val=%d | rollout_test=%s | input_channels=%d | batch_parallel=%s",
        len(train_dataset),
        len(val_loader.dataset),
        "enabled" if bool(config.get("seasonal_rollout", {}).get("enabled", False)) else "disabled",
        len(train_dataset.input_channels),
        bool(config["train"].get("batch_parallel", False)),
    )

    build_result = build_model(config, input_channels=len(train_dataset.input_channels))
    trainer = Trainer(build_result.model, config, run_context.run_dir, logger)
    artifacts = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_transform=train_dataset.target_transform,
    )

    seasonal_rollout_artifacts = run_seasonal_rollout(
        model=trainer.model,
        device=trainer.device,
        config=config,
        prepared_df=prepared_df,
        scaler_path=scaler_path,
        coeff_root=runtime_coeff_root,
        split_name=fold.name,
        run_dir=run_context.run_dir,
        logger=logger,
    )
    rollout_test_summary = _promote_seasonal_rollout_as_test(
        run_dir=run_context.run_dir,
        seasonal_rollout_artifacts=seasonal_rollout_artifacts,
        logger=logger,
    )

    run_manifest = {
        "split_name": fold.name,
        "split_type": "target_lake_temporal_rollout_overlap_test",
        "target_lake": fold.target_lake,
        "train_lakes": fold.train_lakes,
        "split_manifest_path": str(manifest_path),
        "train_coeff_path": str(coeff_paths["train"]),
        "val_coeff_path": str(coeff_paths["val"]),
        "test_coeff_path": None,
        "train_window_path": str(bundle_paths["train"]),
        "val_window_path": str(bundle_paths["val"]),
        "test_window_path": None,
        "project_root": str(paths.project_root),
        "validation_report_path": str(paths.validation_report_json),
        "data_processing_report_path": str(data_processing_report_path),
        "seasonal_rollout_predictions_path": None if seasonal_rollout_artifacts is None else str(seasonal_rollout_artifacts.predictions_path),
        "seasonal_rollout_overlap_predictions_path": None if seasonal_rollout_artifacts is None else str(seasonal_rollout_artifacts.overlap_predictions_path),
        "seasonal_rollout_overlap_metrics_path": None if seasonal_rollout_artifacts is None else str(seasonal_rollout_artifacts.overlap_metrics_path),
        "seasonal_rollout_window_path": None if seasonal_rollout_artifacts is None else str(seasonal_rollout_artifacts.window_path),
        "seasonal_rollout_coeff_path": None if seasonal_rollout_artifacts is None else str(seasonal_rollout_artifacts.coeff_path),
    }
    save_json(run_manifest, run_context.artifacts_dir / "run_manifest.json")
    plot_from_run(run_context.run_dir, logger)

    summary = json.loads(artifacts.run_summary_path.read_text(encoding="utf-8"))
    registry_row = {
        "run_name": run_context.run_name,
        "experiment_name": config["experiment"]["name"],
        "split_name": fold.name,
        "interpolation": config["coeffs"]["interpolation"],
        "method": config["model"]["method"],
        "window_days": config["window"]["window_days"],
        "target_transform": config["features"]["target_transform"],
        "batch_size": config["train"]["batch_size"],
        "learning_rate": config["train"]["learning_rate"],
        "weight_decay": config["train"]["weight_decay"],
        "best_epoch": summary["best_epoch"],
        "best_val_rmse": summary["best_val_rmse"],
        "duration_seconds": summary["duration_seconds"],
        "run_dir": str(run_context.run_dir),
    }
    if bool(config.get("experiment", {}).get("append_registry", True)):
        append_experiment_registry(paths.output_root, registry_row)

    metrics_df = load_dataframe(artifacts.metrics_path)
    test_rows = metrics_df.loc[metrics_df["split"] == "test"]
    test_metrics = test_rows.iloc[0].to_dict() if not test_rows.empty else {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    val_metrics = metrics_df.loc[metrics_df["split"] == "val"].iloc[0].to_dict()
    rollout_metrics = rollout_test_summary.get("rollout_metrics", {})

    fold_summary = {
        "fold_name": fold.name,
        "target_lake": fold.target_lake,
        "train_lakes": " | ".join(fold.train_lakes),
        "run_dir": str(run_context.run_dir),
        "train_windows": int(len(train_dataset)),
        "val_windows": int(len(val_loader.dataset)),
        "test_windows": int(rollout_test_summary.get("overlap_count", 0)),
        "best_epoch": int(summary["best_epoch"]),
        "best_val_rmse": float(summary["best_val_rmse"]),
        "final_val_loss": float(summary["final_val_loss"]),
        "final_test_loss": float(summary["final_test_loss"]) if not np.isnan(summary["final_test_loss"]) else np.nan,
        "val_rmse": float(val_metrics["rmse"]),
        "val_mae": float(val_metrics["mae"]),
        "val_r2": float(val_metrics["r2"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_r2": float(test_metrics["r2"]),
        "test_method": "seasonal_rollout_overlap",
        "seasonal_test_start_datetime": rollout_metrics.get("test_start_datetime"),
        "seasonal_rollout_overlap_start_datetime": rollout_metrics.get("overlap_start_datetime"),
        "seasonal_rollout_overlap_end_datetime": rollout_metrics.get("overlap_end_datetime"),
    }
    _write_experiment_summary(paths.output_root, config["experiment"]["name"], fold_summary)
    base_logger.info("Finished experiment '%s'. Run directory: %s", config["experiment"]["name"], run_context.run_dir)
    return fold_summary


def _bundle_count(bundle_path: Path) -> int:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return int(len(bundle["coeffs"]))


def _resolve_xiaoxingkai_lake_name(available_lakes: list[str]) -> str:
    matches = [lake_name for lake_name in available_lakes if "xiaoxingkai" in _normalize_lake_name(lake_name)]
    if len(matches) != 1:
        raise ValueError(
            "Unable to resolve Xiaoxingkai lake name from prepared dataframe. "
            f"available_sample={available_lakes[:12]}"
        )
    return matches[0]


def _normalize_lake_name(value: str) -> str:
    ascii_text = str(value).encode("ascii", errors="ignore").decode("ascii")
    return "".join(character.lower() for character in ascii_text if character.isalnum())
