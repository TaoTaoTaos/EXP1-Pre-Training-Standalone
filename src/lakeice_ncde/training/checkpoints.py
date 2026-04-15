from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Save a checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint from disk."""
    return torch.load(path, map_location=map_location, weights_only=False)
