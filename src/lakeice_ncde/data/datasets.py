from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
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
    rollout_context: dict[str, Any] | None = None


class CoeffDataset(Dataset):
    """Dataset that reads a coefficient bundle from disk."""

    def __init__(self, bundle_path: Path) -> None:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.bundle_path = bundle_path
        self.coeffs = bundle["coeffs"]
        self.targets = bundle["targets_transformed"].float()
        self.targets_raw = bundle["targets_raw"].float()
        self.metadata = bundle["metadata"]
        self.windows = bundle.get("windows")
        self.feature_columns = bundle.get("feature_columns", [])
        self.feature_scaler = bundle.get("feature_scaler")
        self.physics_context = bundle.get("physics_context")
        self.interpolation = bundle["interpolation"]
        self.input_channels = bundle["input_channels"]
        self.target_transform = bundle["target_transform"]
        self.target_column = bundle["target_column"]
        self.rollout_next_indices = self._build_rollout_next_indices()

        if len(self.coeffs) == 0:
            raise ValueError(f"No coeffs found in bundle: {bundle_path}")
        if len(self.coeffs) != len(self.targets):
            raise ValueError("Coefficient and target counts do not match.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rollout_context = None
        next_index = self.rollout_next_indices[index]
        if next_index >= 0:
            rollout_context = {
                "next_window": self.windows[next_index].float(),
                "next_target_raw": self.targets_raw[next_index],
                "feature_columns": self.feature_columns,
                "feature_scaler": self.feature_scaler,
            }
        return {
            "coeffs": self.coeffs[index],
            "target": self.targets[index],
            "target_raw": self.targets_raw[index],
            "metadata": self.metadata[index],
            "physics_context": None
            if self.physics_context is None
            else {name: values[index] for name, values in self.physics_context.items()},
            "rollout_context": rollout_context,
        }

    def _build_rollout_next_indices(self) -> list[int]:
        """Find immediate same-lake next observations for one-step rollout training."""
        next_indices = [-1 for _ in self.metadata]
        if (
            self.windows is None
            or self.feature_scaler is None
            or self.physics_context is None
        ):
            return next_indices
        required_fields = {"ice_prev_m", "ice_prev_gap_days", "ice_prev_available"}
        if not required_fields.issubset(self.physics_context):
            return next_indices

        grouped_indices: dict[str, list[int]] = {}
        for index, row in enumerate(self.metadata):
            grouped_indices.setdefault(str(row.get("lake_name", "")), []).append(index)

        for indices in grouped_indices.values():
            dated_indices = [
                (index, pd.Timestamp(self.metadata[index]["target_datetime"]))
                for index in indices
                if "target_datetime" in self.metadata[index]
            ]
            dated_indices.sort(key=lambda item: item[1])
            for (current_index, current_time), (candidate_index, candidate_time) in zip(
                dated_indices,
                dated_indices[1:],
            ):
                gap_days = (candidate_time - current_time).total_seconds() / 86400.0
                candidate_gap = float(
                    self.physics_context["ice_prev_gap_days"][candidate_index].item()
                )
                candidate_prev_ice = float(
                    self.physics_context["ice_prev_m"][candidate_index].item()
                )
                candidate_available = float(
                    self.physics_context["ice_prev_available"][candidate_index].item()
                )
                current_target = float(self.targets_raw[current_index].item())
                # 只连接真实相邻观测：下一条样本的 prev_ice 必须确实等于当前样本目标值，
                # 且下一条样本记录的 gap_days 必须等于两条样本时间差。这样即使训练集做过
                # 子采样，也不会把跳过中间观测的样本误当成一步 rollout pair。
                if (
                    candidate_available > 0.5
                    and abs(candidate_gap - gap_days) <= 1.0e-4
                    and abs(candidate_prev_ice - current_target) <= 1.0e-4
                ):
                    next_indices[current_index] = candidate_index
        return next_indices


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
    rollout_items = [
        (index, item["rollout_context"])
        for index, item in enumerate(items)
        if item.get("rollout_context") is not None
    ]
    rollout_context = None
    if rollout_items:
        first_context = rollout_items[0][1]
        rollout_context = {
            "current_indices": torch.tensor(
                [index for index, _ in rollout_items],
                dtype=torch.long,
            ),
            "next_windows": [context["next_window"] for _, context in rollout_items],
            "next_targets_raw": torch.stack(
                [context["next_target_raw"] for _, context in rollout_items]
            ).float(),
            "feature_columns": first_context["feature_columns"],
            "feature_scaler": first_context["feature_scaler"],
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
        rollout_context=rollout_context,
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
