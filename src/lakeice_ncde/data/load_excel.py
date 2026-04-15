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
