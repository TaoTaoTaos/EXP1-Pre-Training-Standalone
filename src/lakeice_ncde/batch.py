from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.utils import get_column_letter

from lakeice_ncde.config import apply_key_value_overrides, load_config, load_yaml
from lakeice_ncde.experiment.registry import append_experiment_registry
from lakeice_ncde.experiment.tracker import RunContext, create_run_context
from lakeice_ncde.utils.io import load_dataframe, save_json
from lakeice_ncde.utils.logging import setup_logging
from lakeice_ncde.utils.paths import ProjectPaths, build_pdf_name, build_prefixed_artifact_name
from lakeice_ncde.visualization.batch_pdf_report import build_batch_pdf_report


@dataclass(frozen=True)
class BatchExperimentSpec:
    """One child experiment planned inside a batch run."""

    config_path: Path
    override_paths: tuple[Path, ...]
    set_values: tuple[str, ...]
    experiment_name: str


@dataclass(frozen=True)
class BatchExperimentResult:
    """Resolved output location for one finished child experiment."""

    experiment_name: str
    run_dir: Path
    config_path: Path
    override_paths: tuple[Path, ...]
    set_values: tuple[str, ...]


@dataclass(frozen=True)
class BatchRunSnapshot:
    """Collected artifacts and summary records for one child experiment."""

    experiment_name: str
    run_dir: Path
    pdf_path: Path
    config_path: Path
    override_paths: tuple[Path, ...]
    set_values: tuple[str, ...]
    records: dict[str, Any]
    registry_row: dict[str, Any]


def is_batch_config(config: dict[str, Any]) -> bool:
    """Return True when the merged config declares child experiments to run."""
    return bool((config.get("batch") or {}).get("experiments"))


def parse_batch_experiment_specs(config: dict[str, Any], project_root: Path) -> list[BatchExperimentSpec]:
    """Resolve child experiment entries from a batch config."""
    batch_cfg = config.get("batch") or {}
    entries = batch_cfg.get("experiments") or []
    if not entries:
        raise ValueError("Batch config must declare at least one entry under batch.experiments.")

    runtime_cfg = config.get("runtime") or {}
    batch_config_dir = Path(runtime_cfg.get("config_path", project_root)).resolve().parent
    specs: list[BatchExperimentSpec] = []
    for index, entry in enumerate(entries, start=1):
        config_ref, override_refs, set_values = _normalize_batch_entry(entry, index)
        config_path = _resolve_batch_path(batch_config_dir, config_ref)
        override_paths = tuple(_resolve_batch_path(batch_config_dir, item) for item in override_refs)
        child_config = load_config(
            project_root=project_root,
            config_path=config_path,
            override_paths=override_paths,
        )
        child_config = apply_key_value_overrides(child_config, set_values)
        experiment_name = str(child_config["experiment"]["name"])
        specs.append(
            BatchExperimentSpec(
                config_path=config_path,
                override_paths=override_paths,
                set_values=tuple(set_values),
                experiment_name=experiment_name,
            )
        )

    experiment_names = [item.experiment_name for item in specs]
    duplicates = sorted({name for name in experiment_names if experiment_names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Batch experiments must resolve to unique experiment names for parallel execution. "
            f"Duplicates: {duplicates}"
        )
    return specs


def run_batch_experiments(
    config: dict[str, Any],
    paths: ProjectPaths,
    project_root: Path,
    base_logger,
) -> dict[str, Any]:
    """Run multiple experiment configs in parallel and aggregate their outputs."""
    specs = parse_batch_experiment_specs(config, project_root)
    run_context = create_run_context(paths.output_root, str(config["experiment"]["name"]), config)
    logger = setup_logging(run_context.log_path)
    logger.info("Starting batch '%s' with %d experiment(s).", config["experiment"]["name"], len(specs))
    logger.info("Batch run directory: %s", run_context.run_dir)

    planned_experiments = [
        {
            "experiment_name": spec.experiment_name,
            "config_path": str(spec.config_path),
            "override_paths": [str(path) for path in spec.override_paths],
            "set_values": list(spec.set_values),
        }
        for spec in specs
    ]
    save_json({"planned_experiments": planned_experiments}, run_context.artifacts_dir / "batch_plan.json")

    max_workers = _resolve_max_workers(config, len(specs))
    logger.info("Launching %d parallel worker(s).", max_workers)
    results_by_name: dict[str, BatchExperimentResult] = {}
    future_to_spec = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as executor:
        for spec in specs:
            future = executor.submit(
                _run_batch_worker,
                str(project_root),
                str(spec.config_path),
                [str(path) for path in spec.override_paths],
                list(spec.set_values),
            )
            future_to_spec[future] = spec
            logger.info("Submitted %s -> %s", spec.experiment_name, spec.config_path)

        try:
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                payload = future.result()
                result = BatchExperimentResult(
                    experiment_name=str(payload["experiment_name"]),
                    run_dir=Path(payload["run_dir"]),
                    config_path=Path(payload["config_path"]),
                    override_paths=tuple(Path(path) for path in payload.get("override_paths", [])),
                    set_values=tuple(payload.get("set_values", [])),
                )
                results_by_name[result.experiment_name] = result
                logger.info("Finished %s -> %s", result.experiment_name, result.run_dir)
        except Exception:
            for future in future_to_spec:
                future.cancel()
            raise

    ordered_results = [results_by_name[spec.experiment_name] for spec in specs]
    snapshots = [collect_batch_run_snapshot(item) for item in ordered_results]
    for snapshot in snapshots:
        append_experiment_registry(paths.output_root, snapshot.registry_row)

    summary_excel_path, copied_pdf_paths, summary_pdf_path = write_batch_summary_artifacts(run_context, snapshots)
    manifest = {
        "batch_run_name": run_context.run_name,
        "batch_run_dir": str(run_context.run_dir),
        "summary_excel_path": str(summary_excel_path),
        "summary_pdf_path": str(summary_pdf_path),
        "copied_pdf_paths": [str(path) for path in copied_pdf_paths],
        "experiments": [
            {
                "experiment_name": snapshot.experiment_name,
                "run_dir": str(snapshot.run_dir),
                "pdf_path": str(snapshot.pdf_path),
                "config_path": str(snapshot.config_path),
                "override_paths": [str(path) for path in snapshot.override_paths],
                "set_values": list(snapshot.set_values),
            }
            for snapshot in snapshots
        ],
    }
    save_json(manifest, run_context.artifacts_dir / "batch_manifest.json")
    logger.info("Batch summary Excel saved to %s", summary_excel_path)
    logger.info("Batch summary PDF saved to %s", summary_pdf_path)
    base_logger.info("Finished batch '%s'. Run directory: %s", config["experiment"]["name"], run_context.run_dir)
    return manifest


def collect_batch_run_snapshot(result: BatchExperimentResult) -> BatchRunSnapshot:
    """Load one completed child run and flatten its config and metrics for batch reporting."""
    config_path = result.run_dir / "config_merged.yaml"
    metrics_path = result.run_dir / "metrics.csv"
    summary_path = result.run_dir / "run_summary.json"
    manifest_path = result.run_dir / "artifacts" / "run_manifest.json"
    pdf_path = result.run_dir / build_pdf_name(result.run_dir.name)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Batch child PDF not found: {pdf_path}")

    config = load_yaml(config_path)
    metrics_df = load_dataframe(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    records: dict[str, Any] = {
        "run.experiment_name": result.experiment_name,
        "run.run_name": result.run_dir.name,
        "run.run_dir": str(result.run_dir),
        "run.source_config_path": str(result.config_path),
        "run.config_merged_path": str(config_path),
        "run.pdf_path": str(pdf_path),
        "run.override_paths": " | ".join(str(path) for path in result.override_paths),
        "run.set_values": " | ".join(result.set_values),
    }
    records.update(_flatten_records(config, prefix="", exclude_keys={"runtime"}))
    records.update(_flatten_records(summary, prefix="summary"))
    records.update(_flatten_records(run_manifest, prefix="manifest"))
    for metric_row in metrics_df.to_dict(orient="records"):
        split_name = str(metric_row.get("split", "unknown"))
        for key, value in metric_row.items():
            if key == "split":
                continue
            records[f"metrics.{split_name}.{key}"] = value

    registry_row = {
        "run_name": result.run_dir.name,
        "experiment_name": result.experiment_name,
        "split_name": run_manifest.get("split_name", result.experiment_name),
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
        "run_dir": str(result.run_dir),
    }
    return BatchRunSnapshot(
        experiment_name=result.experiment_name,
        run_dir=result.run_dir,
        pdf_path=pdf_path,
        config_path=result.config_path,
        override_paths=result.override_paths,
        set_values=result.set_values,
        records=records,
        registry_row=registry_row,
    )


def build_batch_summary_dataframe(snapshots: list[BatchRunSnapshot]) -> pd.DataFrame:
    """Build the one-column-per-experiment Excel summary table."""
    row_order: list[str] = []
    for snapshot in snapshots:
        for key in snapshot.records:
            if key not in row_order:
                row_order.append(key)

    table: dict[str, list[Any]] = {"record": row_order}
    for snapshot in snapshots:
        table[snapshot.experiment_name] = [snapshot.records.get(key, "") for key in row_order]
    return pd.DataFrame(table)


def write_batch_summary_artifacts(run_context: RunContext, snapshots: list[BatchRunSnapshot]) -> tuple[Path, list[Path], Path]:
    """Copy child PDFs and write the aggregate Excel workbook."""
    copied_pdf_paths: list[Path] = []
    for snapshot in snapshots:
        destination = run_context.run_dir / snapshot.pdf_path.name
        shutil.copy2(snapshot.pdf_path, destination)
        copied_pdf_paths.append(destination)

    summary_df = build_batch_summary_dataframe(snapshots)
    excel_path = run_context.run_dir / build_prefixed_artifact_name(run_context.run_name, ".xlsx", suffix="summary")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        worksheet = writer.sheets["summary"]
        worksheet.freeze_panes = "B2"
        for index, column_cells in enumerate(worksheet.columns, start=1):
            values = ["" if cell.value is None else str(cell.value) for cell in column_cells]
            width = min(max(len(value) for value in values) + 2, 72)
            worksheet.column_dimensions[get_column_letter(index)].width = max(width, 14)

    summary_pdf_path = run_context.run_dir / build_pdf_name(run_context.run_name)
    build_batch_pdf_report(
        batch_run_name=run_context.run_name,
        batch_run_dir=run_context.run_dir,
        experiments=[
            {
                "experiment_name": snapshot.experiment_name,
                "run_dir": str(snapshot.run_dir),
            }
            for snapshot in snapshots
        ],
        pdf_path=summary_pdf_path,
    )
    return excel_path, copied_pdf_paths, summary_pdf_path


def _run_batch_worker(
    project_root: str,
    config_path: str,
    override_paths: list[str],
    set_values: list[str],
) -> dict[str, Any]:
    from lakeice_ncde.pipeline import resolve_runtime
    from lakeice_ncde.workflows.xiaoxingkai_transfer import run as run_xiaoxingkai_transfer

    config, paths, logger = resolve_runtime(Path(project_root), config_path, override_paths, set_values)
    config.setdefault("experiment", {})["append_registry"] = False
    fold_summary = run_xiaoxingkai_transfer(config, paths, logger)
    return {
        "experiment_name": config["experiment"]["name"],
        "run_dir": str(fold_summary["run_dir"]),
        "config_path": config_path,
        "override_paths": override_paths,
        "set_values": set_values,
    }


def _normalize_batch_entry(entry: Any, index: int) -> tuple[str, list[str], list[str]]:
    if isinstance(entry, (str, Path)):
        return str(entry), [], []
    if not isinstance(entry, dict):
        raise TypeError(
            "Each batch.experiments entry must be a config path string or a mapping with config/override/set."
        )

    config_ref = entry.get("config")
    if not config_ref:
        raise ValueError(f"Batch experiment entry #{index} is missing the required 'config' field.")

    override_refs = entry.get("override", entry.get("overrides", []))
    if isinstance(override_refs, (str, Path)):
        override_list = [str(override_refs)]
    else:
        override_list = [str(item) for item in (override_refs or [])]

    set_refs = entry.get("set", entry.get("set_values", []))
    if isinstance(set_refs, str):
        set_list = [set_refs]
    else:
        set_list = [str(item) for item in (set_refs or [])]
    return str(config_ref), override_list, set_list


def _resolve_batch_path(base_dir: Path, path_ref: str) -> Path:
    resolved_path = Path(path_ref)
    if not resolved_path.is_absolute():
        resolved_path = (base_dir / resolved_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Batch config reference does not exist: {resolved_path}")
    return resolved_path


def _resolve_max_workers(config: dict[str, Any], num_specs: int) -> int:
    batch_cfg = config.get("batch") or {}
    max_workers = int(batch_cfg.get("max_workers", num_specs))
    if max_workers < 1:
        raise ValueError("batch.max_workers must be at least 1.")
    return min(max_workers, num_specs)


def _flatten_records(
    value: Any,
    prefix: str,
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    exclude_keys = exclude_keys or set()
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            if not prefix and key in exclude_keys:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_records(child, child_prefix))
        return flattened
    if isinstance(value, list):
        return {prefix: " | ".join("" if item is None else str(item) for item in value)}
    return {prefix: value}
