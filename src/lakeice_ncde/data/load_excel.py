from __future__ import annotations

from math import pi
from pathlib import Path
import re

import numpy as np
import pandas as pd

from lakeice_ncde.data.schema import FeatureSchema


def load_raw_excel(raw_excel_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load the raw Excel file."""
    return pd.read_excel(raw_excel_path, sheet_name=sheet_name)


def filter_include_lakes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Restrict the dataframe to an explicit lake subset when configured."""
    data_cfg = config["data"]
    include_lakes = data_cfg.get("include_lakes")
    if not include_lakes:
        return df

    lake_column = data_cfg["lake_column"]
    available_lakes = sorted(df[lake_column].dropna().astype(str).unique().tolist())
    requested_lakes = [str(lake_name) for lake_name in include_lakes]
    resolved_lakes = _resolve_requested_lakes(available_lakes, requested_lakes)
    filtered_df = df.loc[df[lake_column].astype(str).isin(set(resolved_lakes))].reset_index(drop=True)
    if filtered_df.empty:
        preview = available_lakes[:20]
        raise ValueError(
            "Configured data.include_lakes did not match any rows. "
            f"Requested={sorted(requested_lakes)}. "
            f"Available lake names sample={preview} (total={len(available_lakes)})."
        )
    return filtered_df


def standardize_dataframe(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, FeatureSchema]:
    """Parse datetimes, add cyclical features, sort rows, and return the feature schema."""
    data_cfg = config["data"]
    feature_cfg = config["features"]

    df = df.copy()
    dt_col = data_cfg["datetime_column"]
    doy_col = data_cfg["doy_column"]
    target_col = data_cfg["target_column"]

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[data_cfg["lake_column"], dt_col, target_col]).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    df = filter_include_lakes(df, config)

    df[doy_col] = pd.to_numeric(df[doy_col], errors="coerce")
    radians = 2.0 * pi * (df[doy_col].fillna(0.0) / 365.25)
    df["doy_sin"] = np.sin(radians)
    df["doy_cos"] = np.cos(radians)

    numeric_columns = feature_cfg["feature_columns"] + [target_col]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = _add_history_target_features(df, config)
    df = _add_tc2020_curve_features(df, config)

    df = df.sort_values([data_cfg["lake_column"], dt_col]).reset_index(drop=True)

    schema = FeatureSchema(
        time_channel=feature_cfg["time_channel_name"],
        feature_columns=feature_cfg["feature_columns"],
        input_channels=[feature_cfg["time_channel_name"], *feature_cfg["feature_columns"]],
        target_column=target_col,
        target_transform=feature_cfg["target_transform"],
    )
    return df, schema


def _add_history_target_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    feature_columns = set(config["features"]["feature_columns"])
    required_history_features = {"ice_prev_m", "ice_prev_gap_days", "ice_prev_available"}
    if not required_history_features.intersection(feature_columns):
        return df

    lake_col = data_cfg["lake_column"]
    dt_col = data_cfg["datetime_column"]
    target_col = data_cfg["target_column"]

    output = df.sort_values([lake_col, dt_col]).reset_index(drop=True).copy()
    previous_time = output.groupby(lake_col)[dt_col].shift(1)
    previous_target = output.groupby(lake_col)[target_col].shift(1)

    output["ice_prev_available"] = previous_target.notna().astype(float)
    output["ice_prev_m"] = previous_target.fillna(0.0).astype(float)
    output["ice_prev_gap_days"] = (
        (output[dt_col] - previous_time).dt.total_seconds() / 86400.0
    ).fillna(0.0)
    return output


def _require_physics_config_value(physics_cfg: dict, field_name: str) -> object:
    if field_name not in physics_cfg:
        raise ValueError(
            f"Physics loss mode '{physics_cfg.get('mode', 'legacy_stefan')}' requires config field '{field_name}'."
        )
    return physics_cfg[field_name]


def resolve_tc2020_curve_preprocessing_config(config: dict) -> dict[str, object]:
    physics_cfg = config["train"].get("physics_loss", {})
    mode = str(physics_cfg.get("mode", "legacy_stefan"))
    if mode != "tc2020_curve":
        raise ValueError(
            "TC2020 curve preprocessing config can only be resolved when "
            "train.physics_loss.mode == 'tc2020_curve'."
        )

    tc2020_cfg = {
        "temperature_column": str(
            _require_physics_config_value(physics_cfg, "temperature_column")
        ),
        "afdd_column": str(_require_physics_config_value(physics_cfg, "afdd_column")),
        "atdd_column": str(_require_physics_config_value(physics_cfg, "atdd_column")),
        "growth_phase_column": str(
            _require_physics_config_value(physics_cfg, "growth_phase_column")
        ),
        "decay_phase_column": str(
            _require_physics_config_value(physics_cfg, "decay_phase_column")
        ),
        "stable_ice_mask_column": str(
            _require_physics_config_value(physics_cfg, "stable_ice_mask_column")
        ),
        "season_start_month": int(
            _require_physics_config_value(physics_cfg, "season_start_month")
        ),
        "stable_ice_min_m": float(
            _require_physics_config_value(physics_cfg, "stable_ice_min_m")
        ),
        "phase_tolerance_m": float(
            _require_physics_config_value(physics_cfg, "phase_tolerance_m")
        ),
    }
    return tc2020_cfg


def resolve_required_physics_columns(config: dict) -> dict[str, str]:
    physics_cfg = config["train"].get("physics_loss", {})
    if not physics_cfg.get("enabled", False):
        return {}

    mode = str(physics_cfg.get("mode", "legacy_stefan"))
    if mode == "legacy_stefan":
        return {
            "ice_prev_m": str(physics_cfg.get("prev_ice_column", "ice_prev_m")),
            "ice_prev_gap_days": str(physics_cfg.get("gap_days_column", "ice_prev_gap_days")),
            "Air_Temperature_celsius": str(
                physics_cfg.get("temperature_column", "Air_Temperature_celsius")
            ),
            "ice_prev_available": str(
                physics_cfg.get("prev_available_column", "ice_prev_available")
            ),
        }
    if mode == "tc2020_curve":
        tc2020_cfg = resolve_tc2020_curve_preprocessing_config(config)
        required_columns = {
            "afdd": str(tc2020_cfg["afdd_column"]),
            "atdd": str(tc2020_cfg["atdd_column"]),
            "is_growth_phase": str(tc2020_cfg["growth_phase_column"]),
            "is_decay_phase": str(tc2020_cfg["decay_phase_column"]),
            "stable_ice_mask": str(tc2020_cfg["stable_ice_mask_column"]),
        }
        if bool(physics_cfg.get("enable_stefan_grow", False)):
            # TC2020-PLUS 的 Stefan 增量项还需要上一观测冰厚、间隔天数、
            # 当前气温和上一观测是否存在。只在 PLUS 开关打开时加入这些字段，
            # 避免原 tc2020_curve 缓存与训练上下文被无关字段污染。
            required_columns.update(
                {
                    "ice_prev_m": str(physics_cfg.get("prev_ice_column", "ice_prev_m")),
                    "ice_prev_gap_days": str(
                        physics_cfg.get("gap_days_column", "ice_prev_gap_days")
                    ),
                    "Air_Temperature_celsius": str(tc2020_cfg["temperature_column"]),
                    "ice_prev_available": str(
                        physics_cfg.get("prev_available_column", "ice_prev_available")
                    ),
                }
            )
        return required_columns
    raise ValueError(f"Unsupported physics loss mode: {mode}")


def _add_tc2020_curve_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    physics_cfg = config.get("train", {}).get("physics_loss", {})
    if str(physics_cfg.get("mode", "legacy_stefan")) != "tc2020_curve":
        return df

    data_cfg = config["data"]
    tc2020_cfg = resolve_tc2020_curve_preprocessing_config(config)
    lake_col = data_cfg["lake_column"]
    dt_col = data_cfg["datetime_column"]
    target_col = data_cfg["target_column"]
    temperature_col = str(tc2020_cfg["temperature_column"])
    required_columns = {lake_col, dt_col, target_col, temperature_col}
    missing_columns = sorted(column for column in required_columns if column not in df.columns)
    if missing_columns:
        raise ValueError(
            "TC2020 curve preprocessing requires the following dataframe columns, but they are missing: "
            f"{missing_columns}"
        )

    afdd_column = str(tc2020_cfg["afdd_column"])
    atdd_column = str(tc2020_cfg["atdd_column"])
    growth_phase_column = str(tc2020_cfg["growth_phase_column"])
    decay_phase_column = str(tc2020_cfg["decay_phase_column"])
    stable_ice_mask_column = str(tc2020_cfg["stable_ice_mask_column"])
    season_start_month = int(tc2020_cfg["season_start_month"])
    stable_ice_min_m = float(tc2020_cfg["stable_ice_min_m"])
    phase_tolerance_m = float(tc2020_cfg["phase_tolerance_m"])

    output = df.sort_values([lake_col, dt_col]).reset_index(drop=True).copy()
    output[target_col] = pd.to_numeric(output[target_col], errors="coerce").fillna(0.0)
    output[temperature_col] = (
        pd.to_numeric(output[temperature_col], errors="coerce").fillna(0.0)
    )

    previous_time = output.groupby(lake_col)[dt_col].shift(1)
    previous_target = output.groupby(lake_col)[target_col].shift(1)
    gap_days = (
        (output[dt_col] - previous_time).dt.total_seconds() / 86400.0
    ).fillna(0.0)
    positive_gap_days = gap_days.clip(lower=0.0)

    season_anchor_year = output[dt_col].dt.year - (
        output[dt_col].dt.month < season_start_month
    ).astype(int)
    season_id = season_anchor_year.astype(str)

    # AFDD/ATDD 以湖泊-冰季为单位累计，避免跨年后把不同冰季的积温错误地串在一起。
    delta_afdd = np.maximum(-output[temperature_col].to_numpy(dtype=float), 0.0) * positive_gap_days.to_numpy(dtype=float)
    delta_atdd = np.maximum(output[temperature_col].to_numpy(dtype=float), 0.0) * positive_gap_days.to_numpy(dtype=float)
    season_keys = pd.MultiIndex.from_arrays([output[lake_col].astype(str), season_id])
    output[afdd_column] = pd.Series(delta_afdd, index=output.index).groupby(season_keys).cumsum().astype(float)
    output[atdd_column] = pd.Series(delta_atdd, index=output.index).groupby(season_keys).cumsum().astype(float)

    prev_available = previous_target.notna()
    prev_ice = previous_target.fillna(0.0).astype(float)
    current_ice = output[target_col].astype(float)
    delta_ice = current_ice - prev_ice

    # growth/decay phase 用观测冰厚变化方向判定；stable mask 则要求前后两次观测都已有较稳定冰层，
    # 这样可以尽量避开初冻和融尽阶段的高噪声样本，让曲线约束只作用在更可解释的区间。
    output[growth_phase_column] = (
        prev_available & (delta_ice >= -phase_tolerance_m)
    ).astype(float)
    output[decay_phase_column] = (
        prev_available & (delta_ice < -phase_tolerance_m)
    ).astype(float)
    output[stable_ice_mask_column] = (
        prev_available
        & (prev_ice >= stable_ice_min_m)
        & (current_ice >= stable_ice_min_m)
    ).astype(float)
    return output


def _resolve_requested_lakes(available_lakes: list[str], requested_lakes: list[str]) -> list[str]:
    normalized_available = {lake_name: _normalize_lake_name(lake_name) for lake_name in available_lakes}
    resolved: list[str] = []
    for requested_lake in requested_lakes:
        requested_norm = _normalize_lake_name(requested_lake)
        exact_matches = [lake for lake, norm in normalized_available.items() if norm == requested_norm]
        if len(exact_matches) == 1:
            resolved.append(exact_matches[0])
            continue
        fuzzy_matches = [
            lake
            for lake, norm in normalized_available.items()
            if requested_norm and (requested_norm in norm or norm in requested_norm)
        ]
        if len(fuzzy_matches) == 1:
            resolved.append(fuzzy_matches[0])
            continue
        requested_tokens = _tokenize_lake_name(requested_lake)
        if not requested_tokens:
            continue
        overlap_scores = []
        for lake_name in available_lakes:
            available_tokens = _tokenize_lake_name(lake_name)
            overlap = len(requested_tokens & available_tokens)
            overlap_scores.append((overlap, lake_name))
        best_score = max(score for score, _ in overlap_scores)
        if best_score <= 0:
            continue
        best_matches = [lake_name for score, lake_name in overlap_scores if score == best_score]
        if len(best_matches) == 1:
            resolved.append(best_matches[0])
    return sorted(set(resolved))


def _normalize_lake_name(value: str) -> str:
    ascii_text = str(value).encode("ascii", errors="ignore").decode("ascii")
    return "".join(character.lower() for character in ascii_text if character.isalnum())


def _tokenize_lake_name(value: str) -> set[str]:
    ascii_text = str(value).encode("ascii", errors="ignore").decode("ascii").lower()
    return {token for token in re.findall(r"[a-z0-9]+", ascii_text) if len(token) >= 2}
