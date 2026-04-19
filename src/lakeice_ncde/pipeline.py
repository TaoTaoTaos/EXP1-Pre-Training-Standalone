from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from lakeice_ncde.config import apply_key_value_overrides, load_config
from lakeice_ncde.data.coeffs import compute_coefficients_for_windows, save_coeff_bundle
from lakeice_ncde.data.datasets import create_dataloader
from lakeice_ncde.data.load_excel import (
    filter_include_lakes,
    load_raw_excel,
    resolve_required_physics_columns,
    resolve_tc2020_curve_preprocessing_config,
    standardize_dataframe,
)
from lakeice_ncde.data.split import (
    build_lolo_assignments,
    make_default_split,
    resolve_split_runtime,
    save_lolo_folds,
    save_split_assignments,
)
from lakeice_ncde.data.validate import validate_dataframe
from lakeice_ncde.data.windowing import build_window_bundles
from lakeice_ncde.experiment.registry import append_experiment_registry
from lakeice_ncde.experiment.tracker import RunContext, create_run_context
from lakeice_ncde.models.neural_cde import build_model
from lakeice_ncde.training.engine import Trainer
from lakeice_ncde.utils.io import load_dataframe, save_dataframe, save_json as io_save_json
from lakeice_ncde.utils.locking import PathLock
from lakeice_ncde.utils.logging import setup_logging
from lakeice_ncde.utils.paths import ProjectPaths, build_pdf_name, resolve_paths
from lakeice_ncde.utils.seed import set_seed
from lakeice_ncde.visualization.pdf_report import build_pdf_report


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Build the common CLI parser shared by scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Experiment override config path.")
    parser.add_argument("--override", type=str, action="append", default=[], help="Additional YAML override path.")
    parser.add_argument("--set", dest="set_values", type=str, action="append", default=[], help="Dotted key=value override.")
    return parser


def resolve_runtime(project_root: Path, config_path: str, override_paths: list[str], set_values: list[str]):
    """Load config, resolve paths, and return a logger."""
    config = load_config(
        project_root=project_root,
        config_path=(project_root / config_path).resolve(),
        override_paths=[(project_root / override).resolve() for override in override_paths],
    )
    config = apply_key_value_overrides(config, set_values)
    paths = resolve_paths(config, project_root)
    logger = setup_logging()
    return config, paths, logger


def validate_and_save(config: dict, paths: ProjectPaths, logger) -> dict[str, Any]:
    """Validate the raw Excel file and save a report."""
    if paths.validation_report_json.exists():
        report = json.loads(paths.validation_report_json.read_text(encoding="utf-8"))
        logger.info("Reusing cached validation report: %s", paths.validation_report_json)
        return report

    lock_path = paths.validation_report_json.with_name(f"{paths.validation_report_json.name}.lock")
    with PathLock(lock_path):
        if paths.validation_report_json.exists():
            report = json.loads(paths.validation_report_json.read_text(encoding="utf-8"))
            logger.info("Reusing cached validation report: %s", paths.validation_report_json)
            return report

        df = load_raw_excel(paths.raw_excel, sheet_name=config["data"].get("excel_sheet_name"))
        report = validate_dataframe(df, config, paths.raw_excel)
        io_save_json(report, paths.validation_report_json)
        logger.info("Validation report saved to %s", paths.validation_report_json)
        return report


def prepare_dataframe_artifact(config: dict, paths: ProjectPaths, logger) -> tuple[pd.DataFrame, dict]:
    """Prepare the canonical dataframe artifact."""
    raw_df = load_raw_excel(paths.raw_excel, sheet_name=config["data"].get("excel_sheet_name"))
    prepared_df, schema = standardize_dataframe(raw_df, config)
    save_dataframe(prepared_df, paths.prepared_csv)
    io_save_json(
        _build_prepared_dataframe_metadata(config, paths),
        _prepared_dataframe_metadata_path(paths.prepared_csv),
    )
    io_save_json(schema.to_dict(), paths.feature_schema_json)
    logger.info("Prepared dataframe saved to %s", paths.prepared_csv)
    return prepared_df, schema.to_dict()


def load_or_prepare_dataframe(config: dict, paths: ProjectPaths, logger) -> pd.DataFrame:
    """Load the prepared dataframe or create it on demand."""
    lock_path = paths.prepared_csv.with_name(f"{paths.prepared_csv.name}.lock")
    with PathLock(lock_path):
        if paths.prepared_csv.exists():
            prepared_df = load_dataframe(paths.prepared_csv, parse_dates=[config["data"]["datetime_column"]])
            required_physics_columns = resolve_required_physics_columns(config)
            expected_metadata = _build_prepared_dataframe_metadata(config, paths)
            physics_mode = expected_metadata["physics_loss_mode"]
            required_columns = {
                config["data"]["lake_column"],
                config["data"]["datetime_column"],
                config["data"]["target_column"],
                *config["features"]["feature_columns"],
                *required_physics_columns.values(),
            }
            if physics_mode == "tc2020_curve":
                tc2020_cfg = resolve_tc2020_curve_preprocessing_config(config)
                required_columns.add(str(tc2020_cfg["temperature_column"]))
            missing_columns = sorted(column for column in required_columns if column not in prepared_df.columns)
            metadata_path = _prepared_dataframe_metadata_path(paths.prepared_csv)
            cached_metadata = _load_prepared_dataframe_metadata(metadata_path)
            if not missing_columns and cached_metadata == expected_metadata:
                return filter_include_lakes(prepared_df, config)
            if missing_columns:
                logger.warning(
                    "Prepared dataframe is missing required columns %s. Rebuilding %s from the raw Excel source.",
                    missing_columns,
                    paths.prepared_csv,
                )
            else:
                logger.warning(
                    "Prepared dataframe preprocessing metadata is missing or stale at %s. Rebuilding %s from the raw Excel source.",
                    metadata_path,
                    paths.prepared_csv,
                )
        prepared_df, _ = prepare_dataframe_artifact(config, paths, logger)
        return prepared_df


def _prepared_dataframe_metadata_path(prepared_csv_path: Path) -> Path:
    return prepared_csv_path.with_name(f"{prepared_csv_path.stem}_metadata.json")


def _build_prepared_dataframe_metadata(
    config: dict,
    paths: ProjectPaths,
) -> dict[str, Any]:
    physics_cfg = config["train"].get("physics_loss", {})
    physics_mode = str(physics_cfg.get("mode", "legacy_stefan"))
    metadata: dict[str, Any] = {
        "format_version": 1,
        "raw_excel_path": str(paths.raw_excel),
        "excel_sheet_name": config["data"].get("excel_sheet_name"),
        "physics_loss_enabled": bool(physics_cfg.get("enabled", False)),
        "physics_loss_mode": physics_mode,
        "required_physics_columns": resolve_required_physics_columns(config),
    }
    if physics_mode == "tc2020_curve":
        metadata["tc2020_curve_preprocessing"] = resolve_tc2020_curve_preprocessing_config(
            config
        )
    return metadata


def _load_prepared_dataframe_metadata(metadata_path: Path) -> dict[str, Any] | None:
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def make_split_artifacts(config: dict, paths: ProjectPaths, logger) -> list[Path]:
    """Create split assignments for the configured strategy."""
    df = load_or_prepare_dataframe(config, paths, logger)
    strategy = config["split"]["strategy"]
    split_runtime = resolve_split_runtime(config)
    split_paths: list[Path] = []
    if strategy == "group":
        assignments = make_default_split(df, config)
        artifacts = save_split_assignments(
            assignments,
            paths.split_root,
            split_runtime["name"],
            split_seed=int(split_runtime["seed"]),
        )
        split_paths.append(artifacts.manifest_path)
        logger.info(
            "Created group split '%s' with effective seed=%s",
            split_runtime["name"],
            split_runtime["seed"],
        )
        _log_split_assignments(logger, assignments, split_runtime["name"])
    elif strategy == "lolo":
        assignments_per_fold = build_lolo_assignments(df, config)
        artifacts = save_lolo_folds(assignments_per_fold, paths.split_root)
        split_paths.extend(item.manifest_path for item in artifacts)
        for item, assignments in zip(artifacts, assignments_per_fold):
            _log_split_assignments(logger, assignments, item.split_name)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    logger.info("Saved %d split manifest(s) under %s", len(split_paths), paths.split_root)
    return split_paths


def _load_assignments(assignments_path: Path) -> dict[str, str]:
    assignments_df = load_dataframe(assignments_path)
    return dict(zip(assignments_df["lake_name"], assignments_df["split"]))


def _log_split_assignments(logger, assignments: dict[str, str], split_name: str) -> None:
    """Log the exact lake names assigned to each split."""
    for current_split in ("train", "val", "test"):
        lakes = sorted(
            lake_name
            for lake_name, assigned_split in assignments.items()
            if assigned_split == current_split
        )
        logger.info(
            "Split '%s' -> %s lakes (%d): %s",
            split_name,
            current_split,
            len(lakes),
            lakes if lakes else [],
        )


def resolve_split_manifest_path(paths: ProjectPaths, split_name: str) -> Path:
    """Resolve an existing split manifest from a split name."""
    manifest_path = paths.split_root / split_name / "split_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found for split '{split_name}': {manifest_path}")
    return manifest_path


def build_window_artifacts(
    config: dict,
    paths: ProjectPaths,
    logger,
    split_manifests: list[Path] | None = None,
) -> list[Path]:
    """Build windows for each configured split."""
    df = load_or_prepare_dataframe(config, paths, logger)
    split_manifests = split_manifests or make_split_artifacts(config, paths, logger)
    bundle_paths: list[Path] = []
    for manifest_path in split_manifests:
        manifest = json.loads(json.dumps(_read_yaml(manifest_path)))
        assignments = _load_assignments(Path(manifest["assignment_file"]))
        outputs = build_window_bundles(
            df=df,
            assignments=assignments,
            config=config,
            split_name=manifest["split_name"],
            window_root=paths.window_root,
            logger=logger,
        )
        bundle_paths.extend(item.bundle_path for item in outputs.values())
    logger.info("Saved %d window bundle(s) under %s", len(bundle_paths), paths.window_root)
    return bundle_paths


def build_coeff_artifacts(
    config: dict,
    paths: ProjectPaths,
    logger,
    split_manifests: list[Path] | None = None,
) -> list[Path]:
    """Compute interpolation coefficients for each window bundle."""
    window_bundle_paths = build_window_artifacts(config, paths, logger, split_manifests=split_manifests)
    coeff_paths: list[Path] = []
    interpolation = config["coeffs"]["interpolation"]
    for bundle_path in window_bundle_paths:
        coeff_bundle = compute_coefficients_for_windows(bundle_path, interpolation=interpolation, logger=logger)
        split_name = bundle_path.parent.name
        split = bundle_path.name.replace("_windows.pt", "")
        coeff_path, _, _ = save_coeff_bundle(coeff_bundle, paths.coeff_root, split_name=split_name, split=split)
        coeff_paths.append(coeff_path)
    logger.info("Saved %d coefficient bundle(s) under %s", len(coeff_paths), paths.coeff_root)
    return coeff_paths


def train_experiment(
    config: dict,
    paths: ProjectPaths,
    logger,
    split_name: str | None = None,
    output_root: Path | None = None,
    split_manifest_path: Path | None = None,
) -> RunContext:
    """Train one experiment run, save outputs, and return the run context."""
    set_seed(int(config["train"]["seed"]))
    split_runtime = resolve_split_runtime(config)
    if split_manifest_path is None and split_name is not None:
        split_manifest_path = resolve_split_manifest_path(paths, split_name)
    run_context = create_run_context(output_root or paths.output_root, config["experiment"]["name"], config)
    logger = setup_logging(run_context.log_path)
    logger.info("Starting experiment '%s'", config["experiment"]["name"])
    logger.info("Run directory: %s", run_context.run_dir)
    logger.info(
        "Config summary | interpolation=%s | method=%s | window_days=%s | batch_size=%s | target_transform=%s",
        config["coeffs"]["interpolation"],
        config["model"]["method"],
        config["window"]["window_days"],
        config["train"]["batch_size"],
        config["features"]["target_transform"],
    )
    coeff_paths = build_coeff_artifacts(
        config,
        paths,
        logger,
        split_manifests=None if split_manifest_path is None else [split_manifest_path],
    )
    target_split_name = split_name or (
        _read_yaml(split_manifest_path)["split_name"] if split_manifest_path is not None else split_runtime["name"]
    )
    logger.info("Using split '%s' for training.", target_split_name)

    split_dir = paths.coeff_root / target_split_name
    train_coeff = next(split_dir.glob("train_*_coeffs.pt"))
    val_coeff = next(split_dir.glob("val_*_coeffs.pt"))
    test_candidates = list(split_dir.glob("test_*_coeffs.pt"))
    test_coeff = test_candidates[0] if test_candidates else None

    train_dataset, train_loader = create_dataloader(
        train_coeff,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
        batch_parallel=bool(config["train"].get("batch_parallel", False)),
    )
    _, val_loader = create_dataloader(
        val_coeff,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        batch_parallel=bool(config["train"].get("batch_parallel", False)),
    )
    test_loader = None
    if test_coeff is not None and _bundle_count(test_coeff) > 0:
        _, test_loader = create_dataloader(
            test_coeff,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["train"]["num_workers"]),
            batch_parallel=bool(config["train"].get("batch_parallel", False)),
        )
    logger.info(
        "Data summary | train=%d | val=%d | test=%d | input_channels=%d | batch_parallel=%s",
        len(train_dataset),
        len(val_loader.dataset),
        0 if test_loader is None else len(test_loader.dataset),
        len(train_dataset.input_channels),
        bool(config["train"].get("batch_parallel", False)),
    )

    build_result = build_model(config, input_channels=len(train_dataset.input_channels))
    trainer = Trainer(build_result.model, config, run_context.run_dir, logger)
    artifacts = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_transform=train_dataset.target_transform,
    )

    manifest = {
        "split_name": target_split_name,
        "split_seed": split_runtime["seed"] if split_manifest_path is None else _read_yaml(split_manifest_path).get("split_seed"),
        "coeff_paths": [str(path) for path in coeff_paths if path.parent.name == target_split_name],
        "train_coeff_path": str(train_coeff),
        "val_coeff_path": str(val_coeff),
        "test_coeff_path": None if test_coeff is None else str(test_coeff),
        "train_window_path": str(paths.window_root / target_split_name / "train_windows.pt"),
        "project_root": str(paths.project_root),
    }
    io_save_json(manifest, run_context.artifacts_dir / "run_manifest.json")
    plot_from_run(run_context.run_dir, logger)

    summary = json.loads((artifacts.run_summary_path).read_text(encoding="utf-8"))
    registry_row = {
        "run_name": run_context.run_name,
        "experiment_name": config["experiment"]["name"],
        "split_name": target_split_name,
        "interpolation": config["coeffs"]["interpolation"],
        "method": config["model"]["method"],
        "window_days": config["window"]["window_days"],
        "target_transform": config["features"]["target_transform"],
        "batch_size": config["train"]["batch_size"],
        "learning_rate": config["train"]["learning_rate"],
        "weight_decay": config["train"]["weight_decay"],
        "best_epoch": summary["best_epoch"],
        "best_val_rmse": summary["best_val_rmse"],
        "duration_seconds": summary["duration_seconds"],
        "run_dir": str(run_context.run_dir),
    }
    if bool(config.get("experiment", {}).get("append_registry", True)):
        append_experiment_registry(output_root or paths.output_root, registry_row)
    return run_context


def evaluate_run(run_dir: Path, logger) -> dict[str, Any]:
    """Summarize a finished run."""
    metrics = load_dataframe(run_dir / "metrics.csv")
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    logger.info("Run %s | best_epoch=%s | best_val_rmse=%s", run_dir.name, summary["best_epoch"], summary["best_val_rmse"])
    return {"metrics": metrics.to_dict(orient="records"), "summary": summary}


def plot_from_run(run_dir: Path, logger) -> None:
    """Generate the consolidated PDF report for a run."""
    pdf_path = run_dir / build_pdf_name(run_dir.name)
    try:
        build_pdf_report(run_dir, pdf_path)
    except Exception:
        logger.exception("Failed to build PDF report for %s", run_dir)
        return
    logger.info("PDF report saved to %s", pdf_path)


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _bundle_count(bundle_path: Path) -> int:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return int(len(bundle["coeffs"]))
