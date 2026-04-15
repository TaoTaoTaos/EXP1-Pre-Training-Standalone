from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lakeice_ncde.data.datasets import Batch
from lakeice_ncde.data.scaling import inverse_transform_target
from lakeice_ncde.evaluation.metrics import compute_regression_metrics


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_transform: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run prediction over a dataloader and return prediction rows plus metrics."""
    model.eval()
    rows: list[dict[str, Any]] = []

    for batch in loader:
        y_pred_transformed = _predict_batch(model=model, batch=batch, device=device)
        y_pred = inverse_transform_target(y_pred_transformed, target_transform)
        y_true = batch.targets_raw.detach().cpu().numpy()

        for meta, pred_value, pred_transformed, true_value in zip(
            batch.metadata, y_pred, y_pred_transformed, y_true
        ):
            rows.append(
                {
                    "window_id": meta["window_id"],
                    "split": meta["split"],
                    "lake_name": meta["lake_name"],
                    "sample_datetime": meta["target_datetime"],
                    "length": meta["length"],
                    "y_true": float(true_value),
                    "y_pred": float(pred_value),
                    "y_pred_transformed": float(pred_transformed),
                }
            )

    predictions = pd.DataFrame(rows)
    metrics = compute_regression_metrics(
        y_true=predictions["y_true"].to_numpy(),
        y_pred=predictions["y_pred"].to_numpy(),
    )
    return predictions, metrics


def _predict_batch(model: torch.nn.Module, batch: Batch, device: torch.device) -> np.ndarray:
    """Run one forward pass per same-shape coefficient group and restore the original sample order."""
    predictions = torch.empty(len(batch.targets), device=device, dtype=batch.targets.dtype)
    for coeff_group in batch.coeff_groups:
        coeff = _move_coeff_to_device(coeff_group.coeffs, device)
        group_pred = model(coeff)
        if group_pred.ndim == 0:
            group_pred = group_pred.unsqueeze(0)
        if group_pred.ndim != 1:
            raise ValueError(f"Expected 1D batch predictions, received shape {tuple(group_pred.shape)}")
        if group_pred.shape[0] != len(coeff_group.indices):
            raise ValueError(
                f"Model returned {group_pred.shape[0]} predictions for {len(coeff_group.indices)} grouped samples."
            )
        predictions[coeff_group.indices.to(device)] = group_pred
    return predictions.detach().cpu().numpy()


def _move_coeff_to_device(coeff: Any, device: torch.device) -> Any:
    if isinstance(coeff, tuple):
        return tuple(component.to(device) for component in coeff)
    return coeff.to(device)
