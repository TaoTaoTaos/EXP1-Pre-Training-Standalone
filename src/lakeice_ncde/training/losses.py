from __future__ import annotations

import torch
from torch import nn


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
    anchor_temperatures_celsius: torch.Tensor | None,
    config: dict,
    target_transform: str,
) -> torch.Tensor:
    """Penalize physically inconsistent predictions when configured."""
    physics_cfg = config["train"].get("physics_loss", {})
    if not physics_cfg.get("enabled", False):
        return predictions_transformed.new_zeros(())
    if anchor_temperatures_celsius is None:
        raise ValueError("Physics loss is enabled but anchor temperatures are missing from the batch.")

    rule_name = physics_cfg.get("rule", "no_ice_above_freezing")
    if rule_name != "no_ice_above_freezing":
        raise ValueError(f"Unsupported physics loss rule: {rule_name}")

    threshold_celsius = float(physics_cfg.get("threshold_celsius", 0.0))
    weight = float(physics_cfg.get("weight", 0.0))
    if weight <= 0:
        return predictions_transformed.new_zeros(())

    warm_mask = anchor_temperatures_celsius.to(predictions_transformed.device) > threshold_celsius
    if not torch.any(warm_mask):
        return predictions_transformed.new_zeros(())

    predictions_raw = inverse_transform_target_tensor(predictions_transformed, target_transform)
    violations = torch.relu(predictions_raw[warm_mask])
    return weight * torch.mean(violations.square())


def inverse_transform_target_tensor(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Torch version of the target inverse transform for loss calculations."""
    if transform == "none":
        return values
    if transform == "log1p":
        return torch.expm1(values)
    raise ValueError(f"Unsupported target transform: {transform}")
