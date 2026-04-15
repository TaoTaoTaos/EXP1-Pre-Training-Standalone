from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class CoeffGroup:
    """A same-shape coefficient subgroup within one batch."""

    indices: torch.Tensor
    coeffs: Any


@dataclass
class Batch:
    """Batch object used by the trainer."""

    coeffs: list[Any]
    coeff_groups: list[CoeffGroup]
    targets: torch.Tensor
    targets_raw: torch.Tensor
    metadata: list[dict[str, Any]]
    physics_context: dict[str, torch.Tensor] | None = None


class CoeffDataset(Dataset):
    """Dataset that reads a coefficient bundle from disk."""

    def __init__(self, bundle_path: Path) -> None:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.bundle_path = bundle_path
        self.coeffs = bundle["coeffs"]
        self.targets = bundle["targets_transformed"].float()
        self.targets_raw = bundle["targets_raw"].float()
        self.metadata = bundle["metadata"]
        self.physics_context = bundle.get("physics_context")
        self.interpolation = bundle["interpolation"]
        self.input_channels = bundle["input_channels"]
        self.target_transform = bundle["target_transform"]
        self.target_column = bundle["target_column"]

        if len(self.coeffs) == 0:
            raise ValueError(f"No coeffs found in bundle: {bundle_path}")
        if len(self.coeffs) != len(self.targets):
            raise ValueError("Coefficient and target counts do not match.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "coeffs": self.coeffs[index],
            "target": self.targets[index],
            "target_raw": self.targets_raw[index],
            "metadata": self.metadata[index],
            "physics_context": None
            if self.physics_context is None
            else {name: values[index] for name, values in self.physics_context.items()},
        }


def _coeff_signature(coeff: Any) -> tuple[Any, ...]:
    """Build a hashable signature describing the coefficient structure."""
    if isinstance(coeff, tuple):
        return ("tuple", *(tuple(component.shape) for component in coeff))
    return ("tensor", tuple(coeff.shape))


def _stack_coeff_group(coeffs: list[Any]) -> Any:
    """Stack a same-shape coefficient group into a batched tensor/tuple."""
    first = coeffs[0]
    if isinstance(first, tuple):
        return tuple(torch.stack([coeff[i] for coeff in coeffs], dim=0) for i in range(len(first)))
    return torch.stack(coeffs, dim=0)


def collate_coeff_batch(items: list[dict[str, Any]], batch_parallel: bool = False) -> Batch:
    """Collate coefficient items and optionally group same-shape coeffs for batched forward passes."""
    if not items:
        raise ValueError("Received an empty batch from the DataLoader.")
    coeffs = [item["coeffs"] for item in items]
    targets = torch.stack([item["target"] for item in items]).float()
    targets_raw = torch.stack([item["target_raw"] for item in items]).float()
    metadata = [item["metadata"] for item in items]
    physics_context_items = [item.get("physics_context") for item in items]
    physics_context = None
    if physics_context_items and all(value is not None for value in physics_context_items):
        field_names = physics_context_items[0].keys()
        physics_context = {
            field_name: torch.stack([item[field_name] for item in physics_context_items]).float()
            for field_name in field_names
        }
    coeff_groups: list[CoeffGroup] = []
    if batch_parallel:
        grouped_indices: dict[tuple[Any, ...], list[int]] = {}
        grouped_coeffs: dict[tuple[Any, ...], list[Any]] = {}
        for index, coeff in enumerate(coeffs):
            signature = _coeff_signature(coeff)
            grouped_indices.setdefault(signature, []).append(index)
            grouped_coeffs.setdefault(signature, []).append(coeff)
        for signature, indices in grouped_indices.items():
            coeff_groups.append(
                CoeffGroup(
                    indices=torch.tensor(indices, dtype=torch.long),
                    coeffs=_stack_coeff_group(grouped_coeffs[signature]),
                )
            )
    else:
        for index, coeff in enumerate(coeffs):
            coeff_groups.append(CoeffGroup(indices=torch.tensor([index], dtype=torch.long), coeffs=coeff))
    return Batch(
        coeffs=coeffs,
        coeff_groups=coeff_groups,
        targets=targets,
        targets_raw=targets_raw,
        metadata=metadata,
        physics_context=physics_context,
    )


def create_dataloader(
    bundle_path: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    batch_parallel: bool = False,
) -> tuple[CoeffDataset, DataLoader]:
    """Create a DataLoader for a coefficient bundle."""
    dataset = CoeffDataset(bundle_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_coeff_batch, batch_parallel=batch_parallel),
    )
    return dataset, loader
