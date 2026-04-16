from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lakeice_ncde.config import save_yaml
from lakeice_ncde.utils.paths import build_sequential_run_name


@dataclass
class RunContext:
    """Paths associated with one experiment run."""

    run_name: str
    run_dir: Path
    artifacts_dir: Path
    log_path: Path
    config_path: Path


def create_run_context(output_root: Path, experiment_name: str, config: dict) -> RunContext:
    """Create the run directory structure and save the merged config."""
    experiment_root = output_root / experiment_name
    run_name = build_sequential_run_name(experiment_root, experiment_name)
    run_dir = experiment_root / run_name
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    config_path = run_dir / "config_merged.yaml"
    save_yaml(config, config_path)
    return RunContext(
        run_name=run_name,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        log_path=log_path,
        config_path=config_path,
    )
