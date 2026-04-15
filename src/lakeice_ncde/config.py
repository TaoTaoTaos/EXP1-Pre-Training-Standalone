from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


BASE_CONFIG_ORDER = [
    "data.yaml",
    "features.yaml",
    "split.yaml",
    "window.yaml",
    "coeffs.yaml",
    "model.yaml",
    "train.yaml",
    "eval.yaml",
    "experiment.yaml",
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary to YAML."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary to a JSON file."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_config(
    project_root: Path,
    config_path: Path,
    override_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Load base configuration plus experiment overrides."""
    base_dir = project_root / "configs" / "base"
    config: dict[str, Any] = {}
    for name in BASE_CONFIG_ORDER:
        config = deep_merge(config, load_yaml(base_dir / name))
    config = deep_merge(config, load_yaml(config_path))
    for override_path in override_paths or []:
        config = deep_merge(config, load_yaml(override_path))
    config["runtime"] = {
        "project_root": str(project_root),
        "config_path": str(config_path),
        "override_paths": [str(path) for path in (override_paths or [])],
    }
    return config


def apply_key_value_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply simple dotted-path key=value overrides to a config dictionary."""
    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        dotted_key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
    return updated
