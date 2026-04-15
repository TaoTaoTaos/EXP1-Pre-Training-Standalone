from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lakeice_ncde.utils.io import save_dataframe, save_yaml


@dataclass
class SplitArtifacts:
    """Saved split outputs."""

    split_name: str
    split_seed: int | None
    manifest_path: Path
    assignments_path: Path


def _group_counts(df: pd.DataFrame, group_column: str) -> dict[str, int]:
    counts = df[group_column].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def greedy_group_split(
    df: pd.DataFrame,
    group_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    forced_assignments: dict[str, str] | None = None,
    allowed_splits: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Assign whole lakes to train, val, or test with a deterministic greedy strategy."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1.0e-8:
        raise ValueError("Split ratios must sum to 1.")

    forced_assignments = forced_assignments or {}
    allowed_splits = allowed_splits or {}
    group_counts = _group_counts(df, group_column)
    total = float(sum(group_counts.values()))
    targets = {
        "train": train_ratio * total,
        "val": val_ratio * total,
        "test": test_ratio * total,
    }
    current = {"train": 0.0, "val": 0.0, "test": 0.0}

    rng = random.Random(seed)
    groups = list(group_counts.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: item[1], reverse=True)

    assignments: dict[str, str] = {}
    for group_name, group_size in groups:
        best_split = None
        best_score = None
        if group_name in forced_assignments:
            candidate_splits = [forced_assignments[group_name]]
        else:
            candidate_splits = allowed_splits.get(group_name, ["train", "val", "test"])

        if not candidate_splits:
            raise ValueError(f"No candidate splits configured for group '{group_name}'.")

        for split_name in candidate_splits:
            if split_name not in current:
                raise ValueError(f"Unsupported split '{split_name}' configured for group '{group_name}'.")
            trial = current.copy()
            trial[split_name] += float(group_size)
            score = sum(abs(trial[name] - targets[name]) for name in trial)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assert best_split is not None
        assignments[group_name] = best_split
        current[best_split] += float(group_size)

    return assignments


def make_default_split(df: pd.DataFrame, config: dict) -> dict[str, str]:
    """Create the default group-aware split."""
    runtime = resolve_split_runtime(config)
    split_cfg = config["split"]
    data_cfg = config["data"]
    constraints = split_cfg.get("constraints", {})
    return greedy_group_split(
        df=df,
        group_column=data_cfg["lake_column"],
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(runtime["seed"]),
        forced_assignments={str(key): str(value) for key, value in constraints.get("forced_assignments", {}).items()},
        allowed_splits={
            str(key): [str(item) for item in value]
            for key, value in constraints.get("allowed_splits", {}).items()
        },
    )


def resolve_split_runtime(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve and cache the effective split seed/name for one run."""
    split_cfg = config["split"]
    runtime = split_cfg.setdefault("_runtime", {})
    if "seed" in runtime and "name" in runtime:
        return runtime

    configured_seed = split_cfg.get("seed")
    if configured_seed is None:
        seed = random.SystemRandom().randrange(0, 2**31 - 1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = str(split_cfg.get("name", "default_split"))
        name = f"{base_name}_{timestamp}_{seed}"
    else:
        seed = int(configured_seed)
        name = str(split_cfg["name"])

    runtime["seed"] = seed
    runtime["name"] = name
    return runtime


def save_split_assignments(
    assignments: dict[str, str],
    split_root: Path,
    split_name: str,
    split_seed: int | None = None,
) -> SplitArtifacts:
    """Save split assignments and a manifest for later pipeline stages."""
    split_dir = split_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    assignments_df = pd.DataFrame(
        [{"lake_name": lake_name, "split": split} for lake_name, split in sorted(assignments.items())]
    )
    assignments_path = split_dir / "lake_assignments.csv"
    save_dataframe(assignments_df, assignments_path)

    manifest = {
        "split_name": split_name,
        "split_seed": split_seed,
        "assignment_file": str(assignments_path),
        "splits": {
            split: sorted(assignments_df.loc[assignments_df["split"] == split, "lake_name"].tolist())
            for split in ("train", "val", "test")
        },
    }
    manifest_path = split_dir / "split_manifest.yaml"
    save_yaml(manifest, manifest_path)
    return SplitArtifacts(
        split_name=split_name,
        split_seed=split_seed,
        manifest_path=manifest_path,
        assignments_path=assignments_path,
    )


def build_lolo_assignments(
    df: pd.DataFrame,
    config: dict,
) -> list[dict[str, str]]:
    """Create leave-one-lake-out assignment dictionaries."""
    lake_column = config["data"]["lake_column"]
    split_cfg = config["split"]
    runtime = resolve_split_runtime(config)
    lakes = sorted(df[lake_column].dropna().astype(str).unique().tolist())
    assignments_per_fold: list[dict[str, str]] = []

    for fold_index, held_out_lake in enumerate(lakes):
        fold_df = df.loc[df[lake_column] != held_out_lake].copy()
        train_val_assignments = greedy_group_split(
            df=fold_df,
            group_column=lake_column,
            train_ratio=float(split_cfg["train_ratio"]) / (float(split_cfg["train_ratio"]) + float(split_cfg["val_ratio"])),
            val_ratio=float(split_cfg["val_ratio"]) / (float(split_cfg["train_ratio"]) + float(split_cfg["val_ratio"])),
            test_ratio=0.0,
            seed=int(runtime["seed"]) + fold_index,
        )
        assignments = {lake: "test" for lake in lakes if lake == held_out_lake}
        for lake, split in train_val_assignments.items():
            assignments[lake] = "train" if split == "train" else "val"
        assignments_per_fold.append(assignments)
    return assignments_per_fold


def save_lolo_folds(assignments_per_fold: list[dict[str, str]], split_root: Path) -> list[SplitArtifacts]:
    """Save all LOLO folds."""
    artifacts: list[SplitArtifacts] = []
    for index, assignments in enumerate(assignments_per_fold):
        split_name = f"lolo_fold_{index:02d}"
        artifacts.append(save_split_assignments(assignments, split_root, split_name))
    return artifacts
