from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lakeice_ncde.utils.io import save_dataframe, save_json


TRIAL_METADATA_FILENAME = "trial_metadata.json"
TOP_TRIAL_COUNT = 10


@dataclass(frozen=True)
class TrialResultRecord:
    trial_number: int
    trial_status: str
    score: float
    threshold_met: bool
    duration_seconds: float
    trial_dir: Path
    worker_pid: int
    master_row: dict[str, Any]
    parameter_rows: tuple[dict[str, Any], ...]
    experiment_rows: tuple[dict[str, Any], ...]
    error_message: str | None = None
    traceback_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "trial_status": self.trial_status,
            "score": self.score,
            "threshold_met": self.threshold_met,
            "duration_seconds": self.duration_seconds,
            "trial_dir": str(self.trial_dir),
            "worker_pid": self.worker_pid,
            "error_message": self.error_message,
            "traceback_text": self.traceback_text,
            "master_row": _normalize_row(self.master_row),
            "parameter_rows": [_normalize_row(row) for row in self.parameter_rows],
            "experiment_rows": [_normalize_row(row) for row in self.experiment_rows],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialResultRecord":
        return cls(
            trial_number=int(data["trial_number"]),
            trial_status=str(data["trial_status"]),
            score=float(data["score"]),
            threshold_met=bool(data["threshold_met"]),
            duration_seconds=float(data["duration_seconds"]),
            trial_dir=Path(data["trial_dir"]),
            worker_pid=int(data.get("worker_pid", 0)),
            master_row=dict(data.get("master_row") or {}),
            parameter_rows=tuple(dict(row) for row in (data.get("parameter_rows") or [])),
            experiment_rows=tuple(dict(row) for row in (data.get("experiment_rows") or [])),
            error_message=data.get("error_message"),
            traceback_text=data.get("traceback_text"),
        )


def write_trial_record(record: TrialResultRecord) -> Path:
    metadata_path = record.trial_dir / TRIAL_METADATA_FILENAME
    save_json(record.to_dict(), metadata_path)
    return metadata_path


def load_trial_records(search_root: Path) -> list[TrialResultRecord]:
    records: list[TrialResultRecord] = []
    for metadata_path in sorted((search_root / "trials").glob("trial_*/trial_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            records.append(TrialResultRecord.from_dict(json.load(handle)))
    return sorted(records, key=lambda item: item.trial_number)


def write_search_outputs(
    search_root: Path,
    records: list[TrialResultRecord],
    study,
    search_name: str,
    configured_trial_count: int,
) -> dict[str, Any]:
    master_rows = [record.master_row for record in records]
    parameter_rows = [row for record in records for row in record.parameter_rows]
    experiment_rows = [row for record in records for row in record.experiment_rows]

    master_df = _build_dataframe(master_rows, leading_columns=[
        "trial_number",
        "trial_status",
        "score",
        "threshold_met",
        "duration_seconds",
        "worker_pid",
        "objective_experiment",
        "objective_split",
        "objective_metric",
        "objective_metric_value",
        "error_message",
        "path.batch_run_dir",
        "path.batch_pdf",
        "path.batch_excel",
        "path.exp0_run_dir",
        "path.exp1_run_dir",
        "path.exp2_run_dir",
    ])
    parameters_df = _build_dataframe(parameter_rows, leading_columns=[
        "trial_number",
        "trial_status",
        "parameter_name",
        "parameter_key",
        "scope",
        "parameter_type",
        "value",
    ])
    experiments_df = _build_dataframe(experiment_rows, leading_columns=[
        "trial_number",
        "trial_status",
        "experiment_name",
        "is_objective_experiment",
        "run_name",
        "run_dir",
        "pdf_path",
    ])

    top_trials_df = master_df.copy()
    if not top_trials_df.empty:
        top_trials_df = top_trials_df.sort_values(
            by=["threshold_met", "score", "trial_number"],
            ascending=[False, False, True],
        ).head(TOP_TRIAL_COUNT)

    save_dataframe(master_df, search_root / "trials_master.csv")
    save_dataframe(parameters_df, search_root / "trial_parameters.csv")
    save_dataframe(experiments_df, search_root / "trial_experiments.csv")
    save_dataframe(top_trials_df, search_root / "top_trials.csv")

    summary = _build_study_summary(
        search_name=search_name,
        configured_trial_count=configured_trial_count,
        records=records,
        master_df=master_df,
        study=study,
    )
    save_json(summary, search_root / "study_summary.json")
    return summary


def _build_study_summary(
    search_name: str,
    configured_trial_count: int,
    records: list[TrialResultRecord],
    master_df: pd.DataFrame,
    study,
) -> dict[str, Any]:
    trial_status_counts = (
        master_df["trial_status"].value_counts().to_dict()
        if "trial_status" in master_df.columns
        else {}
    )
    finished_trials = [trial for trial in study.trials if trial.state.is_finished()]
    summary: dict[str, Any] = {
        "search_name": search_name,
        "configured_trial_count": configured_trial_count,
        "recorded_trial_count": len(records),
        "optuna_trial_count": len(study.trials),
        "finished_optuna_trial_count": len(finished_trials),
        "threshold_met_count": int(master_df["threshold_met"].sum()) if "threshold_met" in master_df.columns else 0,
        "trial_status_counts": trial_status_counts,
    }
    if not master_df.empty:
        best_row = master_df.sort_values(
            by=["threshold_met", "score", "trial_number"],
            ascending=[False, False, True],
        ).iloc[0]
        summary.update(
            {
                "best_trial_number": int(best_row["trial_number"]),
                "best_score": float(best_row["score"]),
                "best_threshold_met": bool(best_row["threshold_met"]),
                "best_trial_status": str(best_row["trial_status"]),
                "best_batch_run_dir": best_row.get("path.batch_run_dir"),
            }
        )
        best_params = {
            column.removeprefix("param."): _normalize_value(best_row[column])
            for column in master_df.columns
            if column.startswith("param.")
        }
        summary["best_parameters"] = best_params
    return summary


def _build_dataframe(rows: list[dict[str, Any]], leading_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=leading_columns)
    df = pd.DataFrame(rows)
    leading = [column for column in leading_columns if column in df.columns]
    trailing = sorted(column for column in df.columns if column not in leading)
    return df.loc[:, [*leading, *trailing]]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in row.items()}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value
