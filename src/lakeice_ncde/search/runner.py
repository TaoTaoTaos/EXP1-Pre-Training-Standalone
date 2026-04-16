from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import optuna

from lakeice_ncde.batch import run_batch_experiments
from lakeice_ncde.search.config import SearchConfig, load_search_config
from lakeice_ncde.search.objective import BatchRunner, SearchObjective, TrialExecutionPlan
from lakeice_ncde.search.records import load_trial_records, write_search_outputs
from lakeice_ncde.utils.io import save_yaml
from lakeice_ncde.utils.logging import setup_logging
from lakeice_ncde.utils.paths import resolve_paths


def execute_trial_batch(plan: TrialExecutionPlan, logger) -> dict[str, Any]:
    """Run one dynamically resolved batch trial end to end."""
    project_root = Path(plan.batch_config["runtime"]["project_root"])
    paths = resolve_paths(plan.batch_config, project_root)
    return run_batch_experiments(plan.batch_config, paths, project_root, logger)


def run_search(
    project_root: Path,
    config_path: str | Path,
    batch_runner: BatchRunner | None = None,
) -> dict[str, Any]:
    """Run or resume a parameter search study."""
    search_config = load_search_config(project_root, config_path)
    search_root = search_config.output_root
    search_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(search_root / "search.log")
    save_yaml(search_config.to_dict(), search_root / "search_config_resolved.yaml")

    batch_runner = batch_runner or execute_trial_batch
    study = _create_or_load_study(search_config, worker_index=0)
    finished_trial_count = _count_finished_trials(study)
    remaining_trial_count = max(search_config.n_trials - finished_trial_count, 0)
    logger.info(
        "Search '%s' loaded from %s | configured trials=%d | finished=%d | remaining=%d",
        search_config.name,
        search_config.config_path,
        search_config.n_trials,
        finished_trial_count,
        remaining_trial_count,
    )

    worker_errors: list[str] = []
    if remaining_trial_count > 0:
        worker_count = min(search_config.execution.max_parallel_trials, remaining_trial_count)
        logger.info("Launching %d search worker(s).", worker_count)
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=get_context("spawn")) as executor:
            future_to_request = {
                executor.submit(
                    _worker_optimize,
                    str(project_root),
                    search_config,
                    trial_count,
                    worker_index,
                    batch_runner,
                ): (worker_index, trial_count)
                for worker_index, trial_count in enumerate(_distribute_trials(remaining_trial_count, worker_count))
                if trial_count > 0
            }
            for future in as_completed(future_to_request):
                worker_index, trial_count = future_to_request[future]
                try:
                    payload = future.result()
                    logger.info(
                        "Worker %d finished requested_trials=%d | seen_trials=%d",
                        worker_index,
                        trial_count,
                        payload["study_trial_count"],
                    )
                except Exception as exc:  # noqa: BLE001
                    worker_errors.append(f"worker_index={worker_index}: {exc}")
                    logger.exception("Search worker %d failed.", worker_index)
    else:
        logger.info("No remaining trials to schedule; rebuilding search outputs only.")

    study = _create_or_load_study(search_config, worker_index=0)
    records = load_trial_records(search_root)
    summary = write_search_outputs(
        search_root=search_root,
        records=records,
        study=study,
        search_name=search_config.name,
        configured_trial_count=search_config.n_trials,
    )
    logger.info(
        "Search '%s' wrote %d trial record(s) to %s",
        search_config.name,
        len(records),
        search_root,
    )

    if worker_errors:
        raise RuntimeError(
            "One or more search workers failed:\n" + "\n".join(worker_errors)
        )

    return {
        "search_root": str(search_root),
        "recorded_trial_count": len(records),
        "summary": summary,
    }


def _worker_optimize(
    project_root: str,
    search_config: SearchConfig,
    requested_trials: int,
    worker_index: int,
    batch_runner: BatchRunner,
) -> dict[str, Any]:
    if requested_trials < 1:
        return {
            "worker_index": worker_index,
            "requested_trials": requested_trials,
            "study_trial_count": 0,
        }

    study = _create_or_load_study(search_config, worker_index=worker_index)
    objective = SearchObjective(
        project_root=Path(project_root),
        search_config=search_config,
        batch_runner=batch_runner,
    )
    study.optimize(objective, n_trials=requested_trials, n_jobs=1, show_progress_bar=False)
    return {
        "worker_index": worker_index,
        "requested_trials": requested_trials,
        "study_trial_count": len(study.trials),
    }


def _create_or_load_study(search_config: SearchConfig, worker_index: int) -> optuna.study.Study:
    return optuna.create_study(
        study_name=search_config.study_name,
        direction="maximize",
        storage=search_config.storage.build_storage(),
        sampler=search_config.sampler.build_sampler(worker_index),
        load_if_exists=True,
    )


def _count_finished_trials(study: optuna.study.Study) -> int:
    return sum(1 for trial in study.trials if trial.state.is_finished())


def _distribute_trials(total_trials: int, worker_count: int) -> list[int]:
    if total_trials < 0:
        raise ValueError("total_trials must be non-negative.")
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1.")
    counts = [total_trials // worker_count for _ in range(worker_count)]
    for index in range(total_trials % worker_count):
        counts[index] += 1
    return counts
