from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths used across the experiment pipeline."""

    project_root: Path
    raw_excel: Path
    prepared_csv: Path
    validation_report_json: Path
    feature_schema_json: Path
    split_root: Path
    window_root: Path
    coeff_root: Path
    artifact_root: Path
    output_root: Path


def resolve_paths(config: dict, project_root: Path) -> ProjectPaths:
    """Resolve all configured paths relative to the project root."""
    path_cfg = config["paths"]
    return ProjectPaths(
        project_root=project_root,
        raw_excel=(project_root / path_cfg["raw_excel"]).resolve(),
        prepared_csv=(project_root / path_cfg["prepared_csv"]).resolve(),
        validation_report_json=(project_root / path_cfg["validation_report_json"]).resolve(),
        feature_schema_json=(project_root / path_cfg["feature_schema_json"]).resolve(),
        split_root=(project_root / path_cfg["split_root"]).resolve(),
        window_root=(project_root / path_cfg["window_root"]).resolve(),
        coeff_root=(project_root / path_cfg["coeff_root"]).resolve(),
        artifact_root=(project_root / path_cfg["artifact_root"]).resolve(),
        output_root=(project_root / path_cfg["output_root"]).resolve(),
    )


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def timestamp_run_name(prefix: str) -> str:
    """Create a timestamped run name."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
