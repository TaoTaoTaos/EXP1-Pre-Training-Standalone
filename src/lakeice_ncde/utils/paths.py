from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


RUN_SEQUENCE_PATTERN = re.compile(r"^\[(\d+)\]_")


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


def next_run_sequence(experiment_root: Path) -> int:
    """Return the next numeric run sequence for one experiment output directory."""
    if not experiment_root.exists():
        return 1

    max_sequence = 0
    for child in experiment_root.iterdir():
        if not child.is_dir():
            continue
        match = RUN_SEQUENCE_PATTERN.match(child.name)
        if match is None:
            continue
        max_sequence = max(max_sequence, int(match.group(1)))
    return max_sequence + 1


def build_sequential_run_name(
    experiment_root: Path,
    experiment_name: str,
    now: datetime | None = None,
) -> str:
    """Create a run directory name with a zero-padded sequence prefix."""
    sequence = next_run_sequence(experiment_root)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"[{sequence:02d}]_{experiment_name}_{timestamp}"


def build_prefixed_artifact_stem(run_name: str) -> str:
    """Create a filename stem that starts with the run sequence."""
    match = RUN_SEQUENCE_PATTERN.match(run_name)
    if match is None:
        return f"00_{run_name}"

    suffix = run_name[match.end() :]
    return f"{int(match.group(1)):02d}_{suffix}"


def build_prefixed_artifact_name(run_name: str, extension: str, suffix: str | None = None) -> str:
    """Create an artifact filename that starts with the run sequence."""
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    stem = build_prefixed_artifact_stem(run_name)
    if suffix:
        stem = f"{stem}_{suffix}"
    return f"{stem}{normalized_extension}"


def build_pdf_name(run_name: str) -> str:
    """Create a PDF filename that starts with the run sequence."""
    return build_prefixed_artifact_name(run_name, ".pdf")
