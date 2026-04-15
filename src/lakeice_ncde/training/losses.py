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
