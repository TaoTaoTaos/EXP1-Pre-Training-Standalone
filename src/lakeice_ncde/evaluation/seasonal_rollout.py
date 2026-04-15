from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from lakeice_ncde.data.coeffs import compute_coefficients_for_windows, save_coeff_bundle
from lakeice_ncde.data.scaling import StandardScalerBundle, apply_feature_scaler, inverse_transform_target
from lakeice_ncde.data.windowing import _build_single_window
from lakeice_ncde.evaluation.metrics import compute_regression_metrics
from lakeice_ncde.utils.io import save_dataframe, save_json


@dataclass(frozen=True)
class SeasonalRolloutArtifacts:
    predictions_path: Path
    overlap_metrics_path: Path
    overlap_metrics_json_path: Path
    window_path: Path
    coeff_path: Path


def run_seasonal_rollout(
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    prepared_df: pd.DataFrame,
    scaler_path: Path,
    coeff_root: Path,
    split_name: str,
    run_dir: Path,
    logger,
) -> SeasonalRolloutArtifacts | None:
    rollout_cfg = config.get("seasonal_rollout", {})
    if not rollout_cfg.get("enabled", False):
        return None

    rollout_df = build_seasonal_rollout_dataframe(config, prepared_df)
    if rollout_df.empty:
        logger.warning("Seasonal rollout dataframe is empty; skipping seasonal evaluation.")
        return None

    scaler = _load_scaler(scaler_path)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    window_path = artifacts_dir / "seasonal_rollout_windows.pt"
    if rollout_cfg.get("autoregressive_history", False):
        bundle, coeff_bundle, predictions_df = run_autoregressive_rollout(
            model=model,
            device=device,
            config=config,
            rollout_df=rollout_df,
            prepared_df=prepared_df,
            scaler=scaler,
        )
        torch.save(bundle, window_path)
        coeff_path, _, _ = save_coeff_bundle(
            coeff_bundle,
            coeff_root=coeff_root,
            split_name=split_name,
            split="seasonal_rollout",
        )
    else:
        scaled_rollout_df = apply_feature_scaler(rollout_df, scaler)
        bundle = build_inference_window_bundle(scaled_rollout_df, config, split_name="seasonal_rollout")
        torch.save(bundle, window_path)
        coeff_bundle = compute_coefficients_for_windows(window_path, interpolation=config["coeffs"]["interpolation"], logger=logger)
        coeff_path, _, _ = save_coeff_bundle(coeff_bundle, coeff_root=coeff_root, split_name=split_name, split="seasonal_rollout")
        predictions_df = predict_coeff_bundle(model, device, coeff_bundle, target_transform=config["features"]["target_transform"])

    predictions_path = run_dir / "seasonal_rollout_predictions.csv"
    save_dataframe(predictions_df, predictions_path)

    overlap_df = predictions_df.loc[predictions_df["y_true"].notna()].copy()
    if overlap_df.empty:
        overlap_metrics = {"count": 0}
    else:
        overlap_metrics = {"count": int(len(overlap_df)), **compute_regression_metrics(overlap_df["y_true"].to_numpy(), overlap_df["y_pred"].to_numpy())}
    overlap_metrics_path = run_dir / "seasonal_rollout_overlap_metrics.csv"
    overlap_metrics_json_path = run_dir / "seasonal_rollout_overlap_metrics.json"
    save_dataframe(pd.DataFrame([overlap_metrics]), overlap_metrics_path)
    save_json(overlap_metrics, overlap_metrics_json_path)
    return SeasonalRolloutArtifacts(
        predictions_path=predictions_path,
        overlap_metrics_path=overlap_metrics_path,
        overlap_metrics_json_path=overlap_metrics_json_path,
        window_path=window_path,
        coeff_path=coeff_path,
    )


def build_seasonal_rollout_dataframe(config: dict[str, Any], prepared_df: pd.DataFrame) -> pd.DataFrame:
    rollout_cfg = config["seasonal_rollout"]
    time_column = config["data"]["datetime_column"]
    target_column = config["data"]["target_column"]
    lake_column = config["data"]["lake_column"]

    era5_df = pd.read_csv(rollout_cfg["era5_csv"])
    era5_df["datetime"] = pd.to_datetime(era5_df["datetime"])
    era5_df = era5_df.loc[era5_df["datetime"].dt.hour == int(rollout_cfg.get("daily_hour", 12))].copy()

    start = pd.Timestamp(rollout_cfg["start_datetime"])
    end = pd.Timestamp(rollout_cfg["end_datetime"]) if rollout_cfg.get("end_datetime") else era5_df["datetime"].max()
    era5_df = era5_df.loc[(era5_df["datetime"] >= start) & (era5_df["datetime"] <= end)].copy()

    target_lake = _resolve_target_lake_name(prepared_df, lake_column, rollout_cfg.get("target_lake_name"))
    observed_df = prepared_df.loc[prepared_df[lake_column].astype(str) == target_lake].copy()
    observed_df[time_column] = pd.to_datetime(observed_df[time_column])

    if observed_df.empty:
        raise ValueError(f"Target lake '{target_lake}' not found in prepared dataframe.")

    latitude = float(observed_df["latitude"].iloc[0])
    longitude = float(observed_df["longitude"].iloc[0])

    rollout_df = era5_df.rename(columns={"datetime": time_column}).copy()
    rollout_df["era5_datetime"] = rollout_df[time_column]
    rollout_df["doy"] = rollout_df[time_column].dt.dayofyear
    rollout_df["latitude"] = latitude
    rollout_df["longitude"] = longitude
    rollout_df[lake_column] = target_lake
    rollout_df[target_column] = np.nan

    observed_merge = observed_df[[time_column, target_column]].drop_duplicates(subset=[time_column], keep="last")
    rollout_df = rollout_df.merge(observed_merge, on=time_column, how="left", suffixes=("", "_observed"))
    if f"{target_column}_observed" in rollout_df.columns:
        rollout_df[target_column] = rollout_df[f"{target_column}_observed"]
        rollout_df = rollout_df.drop(columns=[f"{target_column}_observed"])

    radians = 2.0 * np.pi * (rollout_df["doy"].fillna(0.0) / 365.25)
    rollout_df["doy_sin"] = np.sin(radians)
    rollout_df["doy_cos"] = np.cos(radians)
    return rollout_df.sort_values(time_column).reset_index(drop=True)


def _resolve_target_lake_name(prepared_df: pd.DataFrame, lake_column: str, requested_name: Any) -> str:
    available = prepared_df[lake_column].dropna().astype(str)
    if available.empty:
        raise ValueError("Prepared dataframe does not contain any lake rows.")

    if requested_name is not None:
        requested_text = str(requested_name).strip()
        exact_matches = available.loc[available == requested_text].unique().tolist()
        if exact_matches:
            return str(exact_matches[0])

        requested_lower = requested_text.lower()
        fuzzy_matches = [name for name in available.unique().tolist() if requested_lower in name.lower()]
        if len(fuzzy_matches) == 1:
            return str(fuzzy_matches[0])

    xiaoxingkai_matches = [name for name in available.unique().tolist() if "xiaoxingkai" in name.lower()]
    if len(xiaoxingkai_matches) == 1:
        return str(xiaoxingkai_matches[0])

    preview = available.unique().tolist()[:10]
    raise ValueError(
        "Unable to resolve seasonal rollout target lake. "
        f"requested={requested_name!r}, available_sample={preview}"
    )


def build_inference_window_bundle(df: pd.DataFrame, config: dict[str, Any], split_name: str) -> dict[str, Any]:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    time_column = data_cfg["datetime_column"]
    target_column = data_cfg["target_column"]
    feature_columns = feature_cfg["feature_columns"]
    windows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    for anchor_index in range(len(df)):
        built = _build_single_window(
            history_df=df.iloc[: anchor_index + 1].copy(),
            feature_columns=feature_columns,
            time_column=time_column,
            target_column=target_column,
            window_days=int(config["window"]["window_days"]),
            lake_name=str(df.iloc[anchor_index][data_cfg["lake_column"]]),
            anchor_index=anchor_index,
        )
        if built is None:
            continue
        windows.append(built["path"])
        target_value = df.iloc[anchor_index][target_column]
        metadata.append(
            {
                "window_id": f"seasonal_rollout_{len(metadata):06d}",
                "split": "seasonal_rollout",
                "lake_name": str(df.iloc[anchor_index][data_cfg["lake_column"]]),
                "target_datetime": built["target_datetime"],
                "length": built["length"],
                "window_days": built["window_days"],
                "target_raw": None if pd.isna(target_value) else float(target_value),
            }
        )

    return {
        "windows": windows,
        "metadata": metadata,
        "feature_columns": feature_columns,
        "input_channels": [feature_cfg["time_channel_name"], *feature_columns],
        "target_column": target_column,
        "target_transform": feature_cfg["target_transform"],
        "split_name": split_name,
        "split": "seasonal_rollout",
    }


def run_autoregressive_rollout(
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    rollout_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    scaler: StandardScalerBundle,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    time_column = data_cfg["datetime_column"]
    target_column = data_cfg["target_column"]
    lake_column = data_cfg["lake_column"]
    feature_columns = feature_cfg["feature_columns"]
    interpolation = config["coeffs"]["interpolation"]
    target_transform = feature_cfg["target_transform"]

    rollout_state_df = rollout_df.copy().reset_index(drop=True)
    for column, default_value in (
        ("ice_prev_m", 0.0),
        ("ice_prev_gap_days", 0.0),
        ("ice_prev_available", 0.0),
    ):
        if column in feature_columns and column not in rollout_state_df.columns:
            rollout_state_df[column] = default_value

    target_lake = str(rollout_state_df[lake_column].iloc[0])
    start_time = pd.Timestamp(rollout_state_df[time_column].min())
    previous_ice, previous_time = _resolve_rollout_initial_state(
        prepared_df=prepared_df,
        lake_column=lake_column,
        target_column=target_column,
        time_column=time_column,
        target_lake=target_lake,
        start_time=start_time,
    )

    windows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    coeffs: list[Any] = []
    coeff_shapes: list[str] = []
    prediction_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for anchor_index in range(len(rollout_state_df)):
            anchor_time = pd.Timestamp(rollout_state_df.at[anchor_index, time_column])
            has_previous = previous_ice is not None and previous_time is not None
            if "ice_prev_m" in feature_columns:
                rollout_state_df.at[anchor_index, "ice_prev_m"] = 0.0 if previous_ice is None else float(previous_ice)
            if "ice_prev_gap_days" in feature_columns:
                gap_days = 0.0 if previous_time is None else (anchor_time - previous_time).total_seconds() / 86400.0
                rollout_state_df.at[anchor_index, "ice_prev_gap_days"] = float(max(gap_days, 0.0))
            if "ice_prev_available" in feature_columns:
                rollout_state_df.at[anchor_index, "ice_prev_available"] = 1.0 if has_previous else 0.0

            scaled_history_df = apply_feature_scaler(rollout_state_df.iloc[: anchor_index + 1].copy(), scaler)
            built = _build_single_window(
                history_df=scaled_history_df,
                feature_columns=feature_columns,
                time_column=time_column,
                target_column=target_column,
                window_days=int(config["window"]["window_days"]),
                lake_name=target_lake,
                anchor_index=anchor_index,
            )
            if built is None:
                continue

            coeff = _compute_single_window_coefficients(built["path"], interpolation)
            pred_transformed = _predict_single_coeff(model, coeff, device)
            pred_value = float(inverse_transform_target(np.array([pred_transformed], dtype=np.float32), target_transform)[0])
            pred_value = max(pred_value, 0.0)

            target_value = rollout_state_df.at[anchor_index, target_column]
            metadata_row = {
                "window_id": f"seasonal_rollout_{len(metadata):06d}",
                "split": "seasonal_rollout",
                "lake_name": target_lake,
                "target_datetime": built["target_datetime"],
                "length": built["length"],
                "window_days": built["window_days"],
                "target_raw": None if pd.isna(target_value) else float(target_value),
                "input_prev_ice_m": None if previous_ice is None else float(previous_ice),
                "input_prev_gap_days": 0.0 if previous_time is None else float((anchor_time - previous_time).total_seconds() / 86400.0),
                "input_prev_available": bool(has_previous),
            }
            metadata.append(metadata_row)
            windows.append(built["path"])
            coeffs.append(coeff)
            coeff_shapes.append(_describe_coeff_shape(coeff))
            prediction_rows.append(
                {
                    "window_id": metadata_row["window_id"],
                    "split": metadata_row["split"],
                    "lake_name": metadata_row["lake_name"],
                    "sample_datetime": metadata_row["target_datetime"],
                    "length": metadata_row["length"],
                    "y_true": metadata_row["target_raw"],
                    "y_pred": pred_value,
                    "y_pred_transformed": float(pred_transformed),
                    "has_observation": metadata_row["target_raw"] is not None,
                    "input_prev_ice_m": metadata_row["input_prev_ice_m"],
                    "input_prev_gap_days": metadata_row["input_prev_gap_days"],
                    "input_prev_available": metadata_row["input_prev_available"],
                }
            )

            previous_ice = pred_value
            previous_time = anchor_time

    bundle = {
        "windows": windows,
        "metadata": metadata,
        "feature_columns": feature_columns,
        "input_channels": [feature_cfg["time_channel_name"], *feature_columns],
        "target_column": target_column,
        "target_transform": target_transform,
        "split_name": "seasonal_rollout",
        "split": "seasonal_rollout",
    }
    coeff_bundle = {
        **bundle,
        "coeffs": coeffs,
        "interpolation": interpolation,
        "coeff_shapes": coeff_shapes,
    }
    predictions_df = pd.DataFrame(prediction_rows)
    return bundle, coeff_bundle, predictions_df


def _resolve_rollout_initial_state(
    prepared_df: pd.DataFrame,
    lake_column: str,
    target_column: str,
    time_column: str,
    target_lake: str,
    start_time: pd.Timestamp,
) -> tuple[float | None, pd.Timestamp | None]:
    observed_df = prepared_df.loc[prepared_df[lake_column].astype(str) == target_lake].copy()
    observed_df[time_column] = pd.to_datetime(observed_df[time_column])
    history_df = observed_df.loc[observed_df[time_column] < start_time].sort_values(time_column)
    if history_df.empty:
        return None, None
    last_row = history_df.iloc[-1]
    return float(last_row[target_column]), pd.Timestamp(last_row[time_column])


def _compute_single_window_coefficients(window: torch.Tensor, interpolation: str) -> Any:
    try:
        import torchcde  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchcde is required for seasonal rollout coefficient computation."
        ) from exc

    x = window.unsqueeze(0)
    if interpolation == "hermite":
        coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
    elif interpolation == "linear":
        coeff = torchcde.linear_interpolation_coeffs(x)
    elif interpolation == "rectilinear":
        coeff = torchcde.linear_interpolation_coeffs(x, rectilinear=0)
    else:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    if isinstance(coeff, tuple):
        return tuple(component.squeeze(0) for component in coeff)
    return coeff.squeeze(0)


def _predict_single_coeff(model: torch.nn.Module, coeff: Any, device: torch.device) -> float:
    coeff_batch = _move_coeff_to_device(_stack_coeff_group([coeff]), device)
    prediction = model(coeff_batch)
    if prediction.ndim == 0:
        return float(prediction.detach().cpu().item())
    return float(prediction.detach().cpu().reshape(-1)[0].item())


def _describe_coeff_shape(coeff: Any) -> str:
    if isinstance(coeff, tuple):
        return "|".join(str(tuple(component.shape)) for component in coeff)
    return str(tuple(coeff.shape))


def predict_coeff_bundle(model: torch.nn.Module, device: torch.device, coeff_bundle: dict[str, Any], target_transform: str) -> pd.DataFrame:
    model.eval()
    coeffs = coeff_bundle["coeffs"]
    metadata = coeff_bundle["metadata"]
    predictions = _predict_grouped(model, coeffs, device)
    y_pred = inverse_transform_target(predictions, target_transform)

    rows: list[dict[str, Any]] = []
    for meta, pred_transformed, pred_value in zip(metadata, predictions, y_pred):
        rows.append(
            {
                "window_id": meta["window_id"],
                "split": meta["split"],
                "lake_name": meta["lake_name"],
                "sample_datetime": meta["target_datetime"],
                "length": meta["length"],
                "y_true": meta["target_raw"],
                "y_pred": float(pred_value),
                "y_pred_transformed": float(pred_transformed),
                "has_observation": meta["target_raw"] is not None,
            }
        )
    return pd.DataFrame(rows)


def _predict_grouped(model: torch.nn.Module, coeffs_list: list[Any], device: torch.device) -> np.ndarray:
    grouped_indices: dict[tuple[Any, ...], list[int]] = {}
    grouped_coeffs: dict[tuple[Any, ...], list[Any]] = {}
    for index, coeff in enumerate(coeffs_list):
        signature = _coeff_signature(coeff)
        grouped_indices.setdefault(signature, []).append(index)
        grouped_coeffs.setdefault(signature, []).append(coeff)

    predictions = np.zeros(len(coeffs_list), dtype=np.float32)
    with torch.no_grad():
        for signature, indices in grouped_indices.items():
            coeff_batch = _stack_coeff_group(grouped_coeffs[signature])
            coeff_batch = _move_coeff_to_device(coeff_batch, device)
            group_pred = model(coeff_batch)
            if group_pred.ndim == 0:
                group_pred = group_pred.unsqueeze(0)
            predictions[np.array(indices, dtype=np.int64)] = group_pred.detach().cpu().numpy()
    return predictions


def _coeff_signature(coeff: Any) -> tuple[Any, ...]:
    if isinstance(coeff, tuple):
        return ("tuple", *(tuple(component.shape) for component in coeff))
    return ("tensor", tuple(coeff.shape))


def _stack_coeff_group(coeffs: list[Any]) -> Any:
    first = coeffs[0]
    if isinstance(first, tuple):
        return tuple(torch.stack([coeff[i] for coeff in coeffs], dim=0) for i in range(len(first)))
    return torch.stack(coeffs, dim=0)


def _load_scaler(path: Path) -> StandardScalerBundle:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return StandardScalerBundle(**data)


def _move_coeff_to_device(coeff: Any, device: torch.device) -> Any:
    if isinstance(coeff, tuple):
        return tuple(component.to(device) for component in coeff)
    return coeff.to(device)
