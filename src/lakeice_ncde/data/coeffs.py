from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lakeice_ncde.utils.io import save_dataframe, save_yaml


def _require_torchcde():
    try:
        import torchcde  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchcde is required for coefficient precomputation. Install dependencies from requirements.txt in the SCI environment."
        ) from exc
    return torchcde


def compute_coefficients_for_windows(bundle_path: Path, interpolation: str, logger=None) -> dict[str, Any]:
    """Load windows and compute interpolation coefficients one sample at a time."""
    torchcde = _require_torchcde()
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    coeffs_list: list[Any] = []
    coeff_shapes: list[str] = []
    total_windows = len(bundle["windows"])
    if logger is not None:
        logger.info(
            "Computing %s coefficients for %d windows from %s",
            interpolation,
            total_windows,
            bundle_path,
        )

    for index, window in enumerate(bundle["windows"]):
        if window.numel() == 0:
            raise ValueError(f"Encountered an empty window at index {index}.")
        x = window.unsqueeze(0)
        if interpolation == "hermite":
            coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        elif interpolation == "linear":
            coeff = torchcde.linear_interpolation_coeffs(x)
        elif interpolation == "rectilinear":
            coeff = torchcde.linear_interpolation_coeffs(x, rectilinear=0)
        else:
            raise ValueError(f"Unsupported interpolation: {interpolation}")
        coeffs_list.append(coeff.squeeze(0))
        if isinstance(coeff, torch.Tensor):
            coeff_shapes.append(str(tuple(coeff.shape)))
        else:
            coeff_shapes.append("|".join(str(tuple(component.shape)) for component in coeff))
        if logger is not None and (
            index == 0
            or index + 1 == total_windows
            or (index + 1) % max(1, total_windows // 10) == 0
        ):
            logger.info(
                "Coefficient progress for %s: %d/%d windows processed.",
                bundle_path.name,
                index + 1,
                total_windows,
            )

    bundle["coeffs"] = coeffs_list
    bundle["interpolation"] = interpolation
    bundle["coeff_shapes"] = coeff_shapes
    return bundle


def save_coeff_bundle(
    coeff_bundle: dict[str, Any],
    coeff_root: Path,
    split_name: str,
    split: str,
) -> tuple[Path, Path, Path]:
    """Save the coefficient bundle and readable manifests."""
    split_dir = coeff_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    coeff_path = split_dir / f"{split}_{coeff_bundle['interpolation']}_coeffs.pt"
    metadata_path = split_dir / f"{split}_{coeff_bundle['interpolation']}_coeffs_metadata.csv"
    manifest_path = split_dir / f"{split}_{coeff_bundle['interpolation']}_coeffs_manifest.yaml"

    torch.save(coeff_bundle, coeff_path)
    metadata_df = pd.DataFrame(coeff_bundle["metadata"]).copy()
    metadata_df["coeff_shape"] = coeff_bundle["coeff_shapes"]
    save_dataframe(metadata_df, metadata_path)
    save_yaml(
        {
            "split_name": split_name,
            "split": split,
            "interpolation": coeff_bundle["interpolation"],
            "bundle_path": str(coeff_path),
            "metadata_path": str(metadata_path),
            "count": int(len(metadata_df)),
        },
        manifest_path,
    )
    return coeff_path, metadata_path, manifest_path
