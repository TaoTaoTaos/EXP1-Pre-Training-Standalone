from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def build_scheduler(config: dict, optimizer: Optimizer):
    """Build the configured learning-rate scheduler."""
    scheduler_cfg = config["train"]["scheduler"]
    name = scheduler_cfg["name"]
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, int(config["train"]["max_epochs"])))
    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            factor=float(scheduler_cfg["factor"]),
            patience=int(scheduler_cfg["patience"]),
            min_lr=float(scheduler_cfg["min_lr"]),
        )
    raise ValueError(f"Unsupported scheduler: {name}")
