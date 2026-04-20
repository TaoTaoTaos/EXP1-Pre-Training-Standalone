from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
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


_PREFIXED_SEQUENCE_PATTERN = re.compile(r"^(?P<index>\d+)(?P<suffix>(?:[_-].*)?)$")
_RUN_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.*?run)(?P<index>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SearchRootSequence:
    base_output_root: Path
    mode: str
    suffix: str
    width: int
    base_index: int

    def existing_roots(self) -> list[tuple[int, Path]]:
        parent = self.base_output_root.parent
        if not parent.exists():
            return []

        roots: list[tuple[int, Path]] = []
        if self.mode == "prefixed":
            for candidate in parent.iterdir():
                if not candidate.is_dir():
                    continue
                match = _PREFIXED_SEQUENCE_PATTERN.fullmatch(candidate.name)
                if match is None or match.group("suffix") != self.suffix:
                    continue
                roots.append((int(match.group("index")), candidate))
        else:
            if self.base_output_root.is_dir():
                roots.append((self.base_index, self.base_output_root))
            prefix = f"{self.base_output_root.name}_run"
            for candidate in parent.iterdir():
                if not candidate.is_dir():
                    continue
                if candidate == self.base_output_root or not candidate.name.startswith(prefix):
                    continue
                index_text = candidate.name.removeprefix(prefix)
                if index_text.isdigit():
                    roots.append((int(index_text), candidate))
        return sorted(roots, key=lambda item: item[0])

    def path_for_index(self, index: int) -> Path:
        if self.mode == "prefixed":
            return self.base_output_root.parent / f"{index:0{self.width}d}{self.suffix}"
        if index == self.base_index:
            return self.base_output_root
        return self.base_output_root.parent / f"{self.base_output_root.name}_run{index:0{self.width}d}"


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
    requested_search_config = load_search_config(project_root, config_path)
    search_config, search_resolution = _resolve_search_config_for_run(requested_search_config)
    search_root = search_config.output_root
    search_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(search_root / "search.log")
    save_yaml(search_config.to_dict(), search_root / "search_config_resolved.yaml")

    if search_resolution is not None and search_resolution["action"] == "resume_latest_root":
        logger.info(
            "Search base root %s auto-selected latest unfinished root %s with study name '%s'.",
            search_resolution["base_output_root"],
            search_resolution["selected_output_root"],
            search_config.name,
        )
    elif search_resolution is not None and search_resolution["action"] == "create_next_root":
        logger.info(
            "Search root %s already has %d finished trial(s); starting a fresh sequential root %s with study name '%s'.",
            search_resolution["completed_output_root"],
            search_resolution["finished_trial_count"],
            search_resolution["selected_output_root"],
            search_config.name,
        )

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


def _resolve_search_config_for_run(search_config: SearchConfig) -> tuple[SearchConfig, dict[str, Any] | None]:
    sequence = _discover_search_root_sequence(search_config.output_root)
    existing_roots = sequence.existing_roots()
    if not existing_roots:
        return search_config, None

    latest_index, latest_root = existing_roots[-1]
    latest_config = _replace_search_root(search_config, latest_root, latest_index)
    if not latest_config.storage.path.exists():
        if latest_root == search_config.output_root and latest_config.name == search_config.name:
            return latest_config, None
        return latest_config, {
            "action": "resume_latest_root",
            "base_output_root": search_config.output_root,
            "selected_output_root": latest_root,
        }

    latest_study = _create_or_load_study(latest_config, worker_index=0)
    finished_trial_count = _count_finished_trials(latest_study)
    if finished_trial_count < latest_config.n_trials:
        if latest_root == search_config.output_root and latest_config.name == search_config.name:
            return latest_config, None
        return latest_config, {
            "action": "resume_latest_root",
            "base_output_root": search_config.output_root,
            "selected_output_root": latest_root,
        }

    next_index = latest_index + 1
    next_root = sequence.path_for_index(next_index)
    next_config = _replace_search_root(search_config, next_root, next_index)
    return next_config, {
        "action": "create_next_root",
        "completed_output_root": latest_root,
        "selected_output_root": next_root,
        "finished_trial_count": finished_trial_count,
    }


def _discover_search_root_sequence(base_output_root: Path) -> SearchRootSequence:
    match = _PREFIXED_SEQUENCE_PATTERN.fullmatch(base_output_root.name)
    if match is not None:
        index_text = match.group("index")
        return SearchRootSequence(
            base_output_root=base_output_root,
            mode="prefixed",
            suffix=match.group("suffix"),
            width=len(index_text),
            base_index=int(index_text),
        )
    return SearchRootSequence(
        base_output_root=base_output_root,
        mode="appended",
        suffix="",
        width=2,
        base_index=1,
    )


def _replace_search_root(search_config: SearchConfig, search_root: Path, run_index: int) -> SearchConfig:
    storage_path = _replace_storage_path(
        search_config.storage.path,
        from_root=search_config.output_root,
        to_root=search_root,
    )
    return replace(
        search_config,
        name=_replace_search_name(search_config.name, run_index),
        output_root=search_root,
        storage=replace(search_config.storage, path=storage_path),
    )


def _replace_storage_path(storage_path: Path, from_root: Path, to_root: Path) -> Path:
    try:
        relative_path = storage_path.relative_to(from_root)
    except ValueError:
        return storage_path
    return (to_root / relative_path).resolve()


def _replace_search_name(base_name: str, run_index: int) -> str:
    match = _RUN_SUFFIX_PATTERN.fullmatch(base_name)
    if match is not None:
        index_width = len(match.group("index"))
        return f"{match.group('prefix')}{run_index:0{index_width}d}"
    if run_index == 1:
        return base_name
    return f"{base_name}_run{run_index:02d}"


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
        sampler=search_config.sampler.build_sampler(worker_index, search_config.enabled_parameters),
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
