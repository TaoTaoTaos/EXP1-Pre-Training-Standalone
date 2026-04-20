from __future__ import annotations

import copy
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lakeice_ncde.batch import BatchExperimentResult, collect_batch_run_snapshot, parse_batch_experiment_specs
from lakeice_ncde.config import load_config
from lakeice_ncde.search.config import SearchConfig, SearchObjectiveConfig
from lakeice_ncde.search.records import TrialResultRecord, write_trial_record
from lakeice_ncde.utils.io import save_yaml
from lakeice_ncde.utils.logging import setup_logging


FAILURE_SCORE = -1.0e12
EXPERIMENT_PATH_ALIASES = {
    "EXP0_pretrain_autoreg": "exp0",
    "EXP1_transfer_autoreg": "exp1",
    "EXP2_transfer_autoreg_stefan": "exp2",
}


class InvalidObjectiveMetricError(ValueError):
    """Raised when a completed run produced unusable objective metrics."""


@dataclass(frozen=True)
class TrialExecutionPlan:
    trial_number: int
    trial_dir: Path
    batch_output_root: Path
    batch_config: dict[str, Any]
    sampled_parameters: dict[str, Any]
    parameter_assignments: tuple[dict[str, Any], ...]
    experiment_overrides: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "trial_dir": str(self.trial_dir),
            "batch_output_root": str(self.batch_output_root),
            "sampled_parameters": self.sampled_parameters,
            "parameter_assignments": [dict(item) for item in self.parameter_assignments],
            "experiment_overrides": [dict(item) for item in self.experiment_overrides],
            "batch_config": self.batch_config,
        }


BatchRunner = Callable[[TrialExecutionPlan, Any], dict[str, Any]]


class SearchObjective:
    def __init__(
        self,
        project_root: Path,
        search_config: SearchConfig,
        batch_runner: BatchRunner,
    ) -> None:
        self.project_root = project_root
        self.search_config = search_config
        self.batch_runner = batch_runner

    def __call__(self, trial) -> float:
        trial_dir = self.search_config.output_root / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(trial_dir / "trial.log")
        worker_pid = os.getpid()
        start_time = time.time()
        sampled_parameters: dict[str, Any] = {}
        execution_plan: TrialExecutionPlan | None = None
        batch_manifest: dict[str, Any] = {}
        snapshots: list[Any] = []
        objective_metrics: dict[str, Any] = {}
        trial_status = "completed"
        threshold_met = False
        error_message: str | None = None
        traceback_text: str | None = None
        score = FAILURE_SCORE

        try:
            logger.info("Starting search trial %d on PID=%d", trial.number, worker_pid)
            for parameter in self.search_config.enabled_parameters:
                sampled_parameters[parameter.name] = parameter.suggest_value(trial)
            execution_plan = build_trial_execution_plan(
                project_root=self.project_root,
                search_config=self.search_config,
                trial_number=trial.number,
                trial_dir=trial_dir,
                sampled_parameters=sampled_parameters,
            )
            save_yaml(execution_plan.batch_config, trial_dir / "trial_config_resolved.yaml")
            save_yaml(
                {
                    "trial_number": trial.number,
                    "sampled_parameters": sampled_parameters,
                    "parameter_assignments": [dict(item) for item in execution_plan.parameter_assignments],
                    "experiment_overrides": [dict(item) for item in execution_plan.experiment_overrides],
                },
                trial_dir / "trial_overrides.yaml",
            )

            batch_manifest = self.batch_runner(execution_plan, logger)
            snapshots = _build_snapshots_from_manifest(batch_manifest)
            objective_snapshot = next(
                snapshot
                for snapshot in snapshots
                if snapshot.experiment_name == self.search_config.objective.experiment_name
            )
            objective_metrics = evaluate_objective_snapshot(
                objective_snapshot.records,
                self.search_config.objective,
            )
            score = float(objective_metrics["score"])
            threshold_met = bool(objective_metrics["threshold_met"])
            logger.info(
                "Completed search trial %d | score=%.6f | threshold_met=%s",
                trial.number,
                score,
                threshold_met,
            )
        except InvalidObjectiveMetricError as exc:
            trial_status = "invalid_metrics"
            threshold_met = False
            score = FAILURE_SCORE
            error_message = str(exc)
            logger.warning(
                "Search trial %d produced invalid objective metrics and will receive the failure score: %s",
                trial.number,
                error_message,
            )
        except Exception as exc:  # noqa: BLE001
            trial_status = "failed"
            threshold_met = False
            score = FAILURE_SCORE
            error_message = str(exc)
            traceback_text = traceback.format_exc()
            logger.exception("Search trial %d failed.", trial.number)

        duration_seconds = time.time() - start_time
        trial_record = build_trial_result_record(
            search_config=self.search_config,
            trial_number=trial.number,
            trial_status=trial_status,
            score=score,
            threshold_met=threshold_met,
            duration_seconds=duration_seconds,
            trial_dir=trial_dir,
            worker_pid=worker_pid,
            sampled_parameters=sampled_parameters,
            execution_plan=execution_plan,
            batch_manifest=batch_manifest,
            snapshots=snapshots,
            objective_metrics=objective_metrics,
            error_message=error_message,
            traceback_text=traceback_text,
        )
        write_trial_record(trial_record)

        trial.set_user_attr("trial_status", trial_status)
        trial.set_user_attr("threshold_met", threshold_met)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("score", score)
        trial.set_user_attr("worker_pid", worker_pid)
        if error_message is not None:
            trial.set_user_attr("error_message", error_message)
        return float(score)


def build_trial_execution_plan(
    project_root: Path,
    search_config: SearchConfig,
    trial_number: int,
    trial_dir: Path,
    sampled_parameters: dict[str, Any],
) -> TrialExecutionPlan:
    base_batch_config = load_config(
        project_root=project_root,
        config_path=search_config.base_batch_config,
        override_paths=[],
    )
    base_specs = parse_batch_experiment_specs(base_batch_config, project_root)
    experiment_names = [spec.experiment_name for spec in base_specs]

    generated_sets: dict[str, list[str]] = {
        spec.experiment_name: list(spec.set_values)
        for spec in base_specs
    }
    parameter_assignments: list[dict[str, Any]] = []
    for parameter in search_config.enabled_parameters:
        if parameter.name not in sampled_parameters:
            raise ValueError(f"Sampled parameters are missing '{parameter.name}'.")
        value = sampled_parameters[parameter.name]
        targets = experiment_names if parameter.applies_to_all else list(parameter.scope)
        unknown_targets = sorted(target for target in targets if target not in experiment_names)
        if unknown_targets:
            raise ValueError(
                f"Parameter '{parameter.name}' targets unknown experiment(s): {unknown_targets}. "
                f"Available experiments: {experiment_names}"
            )
        override = f"{parameter.key}={json.dumps(value, ensure_ascii=False)}"
        for target in targets:
            generated_sets[target].append(override)
        parameter_assignments.append(
            {
                "parameter_name": parameter.name,
                "parameter_key": parameter.key,
                "parameter_type": parameter.parameter_type,
                "scope": "all" if parameter.applies_to_all else " | ".join(parameter.scope),
                "value": value,
                "override": override,
                "applied_experiments": list(targets),
            }
        )

    batch_output_root = trial_dir / "runs"
    resolved_batch_config = copy.deepcopy(base_batch_config)
    resolved_batch_config["paths"]["output_root"] = str(batch_output_root)

    experiment_overrides: list[dict[str, Any]] = []
    batch_entries: list[dict[str, Any]] = []
    for spec in base_specs:
        entry: dict[str, Any] = {"config": str(spec.config_path)}
        if spec.override_paths:
            entry["override"] = [str(path) for path in spec.override_paths]
        set_values = list(generated_sets[spec.experiment_name])
        if set_values:
            entry["set"] = set_values
        batch_entries.append(entry)
        experiment_overrides.append(
            {
                "experiment_name": spec.experiment_name,
                "config_path": str(spec.config_path),
                "override_paths": [str(path) for path in spec.override_paths],
                "set_values": set_values,
            }
        )
    resolved_batch_config["batch"]["experiments"] = batch_entries

    return TrialExecutionPlan(
        trial_number=trial_number,
        trial_dir=trial_dir,
        batch_output_root=batch_output_root,
        batch_config=resolved_batch_config,
        sampled_parameters=dict(sampled_parameters),
        parameter_assignments=tuple(parameter_assignments),
        experiment_overrides=tuple(experiment_overrides),
    )


def evaluate_objective_snapshot(
    records: dict[str, Any],
    objective: SearchObjectiveConfig,
) -> dict[str, Any]:
    threshold_value = _require_metric(records, objective.split, objective.metric)
    test_r2 = _require_metric(records, objective.split, "r2")
    test_rmse = _require_metric(records, objective.split, "rmse")
    test_negative_count = _require_metric(records, objective.split, "negative_count")

    if objective.score_formula != "r2_dominant_composite":
        raise ValueError(f"Unsupported score formula '{objective.score_formula}'.")
    score = compute_r2_dominant_composite_score(
        test_r2=test_r2,
        test_rmse=test_rmse,
        test_negative_count=test_negative_count,
    )
    return {
        "objective_metric_value": threshold_value,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_negative_count": test_negative_count,
        "score": score,
        "threshold_met": threshold_value > objective.success_threshold,
    }


def compute_r2_dominant_composite_score(
    test_r2: float,
    test_rmse: float,
    test_negative_count: float,
) -> float:
    return float(test_r2 - 0.20 * test_rmse - 0.05 * math.log1p(test_negative_count))


def build_trial_result_record(
    search_config: SearchConfig,
    trial_number: int,
    trial_status: str,
    score: float,
    threshold_met: bool,
    duration_seconds: float,
    trial_dir: Path,
    worker_pid: int,
    sampled_parameters: dict[str, Any],
    execution_plan: TrialExecutionPlan | None,
    batch_manifest: dict[str, Any],
    snapshots: list[Any],
    objective_metrics: dict[str, Any],
    error_message: str | None,
    traceback_text: str | None,
) -> TrialResultRecord:
    master_row: dict[str, Any] = {
        "trial_number": trial_number,
        "trial_status": trial_status,
        "score": score,
        "threshold_met": threshold_met,
        "duration_seconds": duration_seconds,
        "worker_pid": worker_pid,
        "objective_experiment": search_config.objective.experiment_name,
        "objective_split": search_config.objective.split,
        "objective_metric": search_config.objective.metric,
        "objective_metric_value": objective_metrics.get("objective_metric_value"),
        "error_message": error_message,
        "path.batch_run_dir": batch_manifest.get("batch_run_dir", ""),
        "path.batch_pdf": batch_manifest.get("summary_pdf_path", ""),
        "path.batch_excel": batch_manifest.get("summary_excel_path", ""),
    }
    for parameter_name, parameter_value in sampled_parameters.items():
        master_row[f"param.{parameter_name}"] = parameter_value

    parameter_rows: list[dict[str, Any]] = []
    if execution_plan is not None:
        for assignment in execution_plan.parameter_assignments:
            parameter_rows.append(
                {
                    "trial_number": trial_number,
                    "trial_status": trial_status,
                    **assignment,
                }
            )

    experiment_rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        alias = EXPERIMENT_PATH_ALIASES.get(snapshot.experiment_name)
        master_row[f"path.{snapshot.experiment_name}.run_dir"] = str(snapshot.run_dir)
        master_row[f"path.{snapshot.experiment_name}.pdf"] = str(snapshot.pdf_path)
        if alias is not None:
            master_row[f"path.{alias}_run_dir"] = str(snapshot.run_dir)
        experiment_row = {
            "trial_number": trial_number,
            "trial_status": trial_status,
            "experiment_name": snapshot.experiment_name,
            "is_objective_experiment": snapshot.experiment_name == search_config.objective.experiment_name,
            "run_name": snapshot.records.get("run.run_name"),
            "run_dir": str(snapshot.run_dir),
            "pdf_path": str(snapshot.pdf_path),
        }
        for key, value in snapshot.records.items():
            if key.startswith("metrics.") or key.startswith("summary."):
                master_row[f"{snapshot.experiment_name}.{key}"] = value
                experiment_row[key] = value
        experiment_rows.append(experiment_row)

    return TrialResultRecord(
        trial_number=trial_number,
        trial_status=trial_status,
        score=score,
        threshold_met=threshold_met,
        duration_seconds=duration_seconds,
        trial_dir=trial_dir,
        worker_pid=worker_pid,
        master_row=master_row,
        parameter_rows=tuple(parameter_rows),
        experiment_rows=tuple(experiment_rows),
        error_message=error_message,
        traceback_text=traceback_text,
    )


def _build_snapshots_from_manifest(manifest: dict[str, Any]) -> list[Any]:
    snapshots: list[Any] = []
    for item in manifest.get("experiments", []):
        result = BatchExperimentResult(
            experiment_name=str(item["experiment_name"]),
            run_dir=Path(item["run_dir"]),
            config_path=Path(item["config_path"]),
            override_paths=tuple(Path(path) for path in item.get("override_paths", [])),
            set_values=tuple(str(value) for value in item.get("set_values", [])),
        )
        snapshots.append(collect_batch_run_snapshot(result))
    return snapshots


def _require_metric(records: dict[str, Any], split: str, metric_name: str) -> float:
    key = f"metrics.{split}.{metric_name}"
    if key not in records:
        raise KeyError(f"Objective metric '{key}' is missing from trial records.")
    value = float(records[key])
    if not math.isfinite(value):
        raise InvalidObjectiveMetricError(f"Objective metric '{key}' is not finite: {value}.")
    return value
