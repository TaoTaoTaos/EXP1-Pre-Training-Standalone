from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PhysicsLossBreakdown:
    """Individual physics-loss components for logging and optimization."""

    stefan: torch.Tensor
    nonneg: torch.Tensor
    total: torch.Tensor
    kappa: torch.Tensor


def build_loss(config: dict) -> nn.Module:
    """Build the configured regression loss."""
    loss_name = config["train"]["loss"]
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=float(config["train"]["huber_delta"]))
    raise ValueError(f"Unsupported loss: {loss_name}")


def check_loss_is_finite(loss: torch.Tensor) -> None:
    """Fail fast on NaN or infinite losses."""
    if not torch.isfinite(loss):
        raise ValueError(f"Loss is not finite: {loss.item()}")


def compute_physics_loss(
    predictions_transformed: torch.Tensor,
    physics_context: dict[str, torch.Tensor] | None,
    config: dict,
    target_transform: str,
    theta_kappa: torch.Tensor | None,
) -> PhysicsLossBreakdown:
    """Compute the configured Stefan-style physics loss."""
    physics_cfg = config["train"].get("physics_loss", {})
    zero = predictions_transformed.new_zeros(())
    if not physics_cfg.get("enabled", False):
        return PhysicsLossBreakdown(stefan=zero, nonneg=zero, total=zero, kappa=zero)
    if physics_context is None:
        raise ValueError("Physics loss is enabled but physics_context is missing from the batch.")
    if theta_kappa is None:
        raise ValueError("Physics loss is enabled but theta_kappa is not initialized.")

    pred_ice = inverse_transform_target_tensor(predictions_transformed, target_transform)
    prev_ice = _get_physics_field(physics_context, "ice_prev_m", pred_ice.device)
    gap_days = _get_physics_field(physics_context, "ice_prev_gap_days", pred_ice.device)
    temp_c = _get_physics_field(physics_context, "Air_Temperature_celsius", pred_ice.device)
    prev_ok = _get_physics_field(physics_context, "ice_prev_available", pred_ice.device) > 0.5

    grow_mask = (
        prev_ok
        & (prev_ice > float(physics_cfg.get("min_prev_ice_m", 1.0e-3)))
        & (temp_c < float(physics_cfg.get("grow_temp_threshold_celsius", -0.5)))
    )

    delta_F = torch.relu(-temp_c) * gap_days
    kappa = F.softplus(theta_kappa)
    stefan_residual = (pred_ice.square() - prev_ice.square()) - kappa * delta_F

    grow_mask_float = grow_mask.float()
    loss_stefan = (grow_mask_float * stefan_residual.square()).sum() / (grow_mask_float.sum() + 1.0e-8)
    loss_nonneg = torch.relu(-pred_ice).square().mean()
    total = (
        float(physics_cfg.get("lambda_st", 1.0)) * loss_stefan
        + float(physics_cfg.get("lambda_nn", 1.0)) * loss_nonneg
    )
    return PhysicsLossBreakdown(stefan=loss_stefan, nonneg=loss_nonneg, total=total, kappa=kappa)


def inverse_transform_target_tensor(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Torch version of the target inverse transform for loss calculations."""
    if transform == "none":
        return values
    if transform == "log1p":
        return torch.expm1(values)
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_softplus(value: float) -> float:
    """Map a positive initialization target back to the unconstrained softplus parameter space."""
    if value <= 0:
        raise ValueError(f"Expected a positive kappa initialization, got {value}.")
    return float(torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32))).item())


def _get_physics_field(physics_context: dict[str, torch.Tensor], field_name: str, device: torch.device) -> torch.Tensor:
    if field_name not in physics_context:
        raise ValueError(f"Physics loss requires '{field_name}' in physics_context.")
    return physics_context[field_name].to(device)
