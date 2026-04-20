from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from lakeice_ncde.config import load_yaml_with_extends


SUPPORTED_SCORE_FORMULAS = {"r2_dominant_composite"}
SUPPORTED_SAMPLERS = {"grid", "tpe"}
SUPPORTED_STORAGE_TYPES = {"journal"}
SUPPORTED_PARAMETER_TYPES = {"int", "float", "categorical", "bool"}


@dataclass(frozen=True)
class SearchSamplerConfig:
    name: str
    seed: int
    constant_liar: bool = False

    def build_sampler(
        self,
        worker_index: int = 0,
        parameters: tuple["SearchParameterSpec", ...] = (),
    ) -> optuna.samplers.BaseSampler:
        if self.name == "tpe":
            return optuna.samplers.TPESampler(
                seed=self.seed + worker_index,
                constant_liar=self.constant_liar,
            )
        if self.name == "grid":
            return optuna.samplers.GridSampler(
                _build_grid_search_space(parameters),
                seed=self.seed + worker_index,
            )
        raise ValueError(f"Unsupported sampler '{self.name}'. Supported samplers: {sorted(SUPPORTED_SAMPLERS)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "constant_liar": self.constant_liar,
        }


@dataclass(frozen=True)
class SearchStorageConfig:
    type: str
    path: Path

    def build_storage(self) -> JournalStorage:
        if self.type != "journal":
            raise ValueError(f"Unsupported storage type '{self.type}'. Supported types: {sorted(SUPPORTED_STORAGE_TYPES)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            backend = JournalFileBackend(str(self.path), lock_obj=JournalFileOpenLock(str(self.path)))
        else:
            backend = JournalFileBackend(str(self.path))
        return JournalStorage(backend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "path": str(self.path),
        }


@dataclass(frozen=True)
class SearchExecutionConfig:
    max_parallel_trials: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_parallel_trials": self.max_parallel_trials,
        }


@dataclass(frozen=True)
class SearchObjectiveConfig:
    experiment_name: str
    split: str
    metric: str
    success_threshold: float
    score_formula: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "split": self.split,
            "metric": self.metric,
            "success_threshold": self.success_threshold,
            "score_formula": self.score_formula,
        }


@dataclass(frozen=True)
class SearchParameterSpec:
    name: str
    key: str
    enabled: bool
    scope: tuple[str, ...]
    parameter_type: str
    low: int | float | None = None
    high: int | float | None = None
    step: int | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()

    @property
    def applies_to_all(self) -> bool:
        return self.scope == ("all",)

    def overlaps_with(self, other: "SearchParameterSpec") -> bool:
        if self.key != other.key:
            return False
        if self.applies_to_all or other.applies_to_all:
            return True
        return bool(set(self.scope) & set(other.scope))

    def suggest_value(self, trial: optuna.trial.Trial) -> Any:
        if not self.enabled:
            raise ValueError(f"Parameter '{self.name}' is disabled and cannot be sampled.")
        if self.parameter_type == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), step=int(self.step or 1), log=bool(self.log))
        if self.parameter_type == "float":
            return trial.suggest_float(self.name, float(self.low), float(self.high), log=bool(self.log))
        if self.parameter_type == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.parameter_type == "bool":
            return bool(trial.suggest_categorical(self.name, list(self.choices)))
        raise ValueError(f"Unsupported parameter type '{self.parameter_type}'.")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "key": self.key,
            "enabled": self.enabled,
            "scope": list(self.scope),
            "type": self.parameter_type,
        }
        if self.low is not None:
            data["low"] = self.low
        if self.high is not None:
            data["high"] = self.high
        if self.step is not None:
            data["step"] = self.step
        if self.log:
            data["log"] = self.log
        if self.choices:
            data["choices"] = list(self.choices)
        return data


@dataclass(frozen=True)
class SearchConfig:
    config_path: Path
    name: str
    base_batch_config: Path
    output_root: Path
    n_trials: int
    sampler: SearchSamplerConfig
    storage: SearchStorageConfig
    execution: SearchExecutionConfig
    objective: SearchObjectiveConfig
    parameters: tuple[SearchParameterSpec, ...]

    @property
    def study_name(self) -> str:
        return self.name

    @property
    def enabled_parameters(self) -> tuple[SearchParameterSpec, ...]:
        return tuple(parameter for parameter in self.parameters if parameter.enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "search": {
                "name": self.name,
                "base_batch_config": str(self.base_batch_config),
                "output_root": str(self.output_root),
                "n_trials": self.n_trials,
                "sampler": self.sampler.to_dict(),
                "storage": self.storage.to_dict(),
                "execution": self.execution.to_dict(),
                "objective": self.objective.to_dict(),
                "parameters": [parameter.to_dict() for parameter in self.parameters],
            }
        }


def load_search_config(project_root: Path, config_path: str | Path) -> SearchConfig:
    resolved_config_path = _resolve_repo_path(project_root, config_path)
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Search config not found: {resolved_config_path}")
    raw_config = load_yaml_with_extends(resolved_config_path)
    search_cfg = raw_config.get("search")
    if not isinstance(search_cfg, dict):
        raise ValueError(f"Search config must define a top-level 'search' mapping: {resolved_config_path}")

    name = str(search_cfg.get("name", "")).strip()
    if not name:
        raise ValueError("search.name is required.")

    raw_base_batch_config = str(search_cfg.get("base_batch_config", "")).strip()
    if not raw_base_batch_config:
        raise ValueError("search.base_batch_config is required.")
    base_batch_config = _resolve_repo_path(project_root, raw_base_batch_config)
    if not base_batch_config.exists():
        raise FileNotFoundError(f"search.base_batch_config not found: {base_batch_config}")

    raw_output_root = str(search_cfg.get("output_root", "")).strip()
    if not raw_output_root:
        raise ValueError("search.output_root is required.")
    output_root = _resolve_repo_path(project_root, raw_output_root)
    n_trials = int(search_cfg.get("n_trials", 0))
    if n_trials < 1:
        raise ValueError("search.n_trials must be at least 1.")

    sampler_cfg = search_cfg.get("sampler") or {}
    sampler_name = str(sampler_cfg.get("name", "")).strip().lower()
    if sampler_name not in SUPPORTED_SAMPLERS:
        raise ValueError(f"search.sampler.name must be one of {sorted(SUPPORTED_SAMPLERS)}.")
    sampler = SearchSamplerConfig(
        name=sampler_name,
        seed=int(sampler_cfg.get("seed", 42)),
        constant_liar=bool(sampler_cfg.get("constant_liar", False)),
    )

    storage_cfg = search_cfg.get("storage") or {}
    storage_type = str(storage_cfg.get("type", "")).strip().lower()
    if storage_type not in SUPPORTED_STORAGE_TYPES:
        raise ValueError(f"search.storage.type must be one of {sorted(SUPPORTED_STORAGE_TYPES)}.")
    raw_storage_path = storage_cfg.get("path", "")
    if not raw_storage_path:
        raise ValueError("search.storage.path is required.")
    storage_path = Path(raw_storage_path)
    if not storage_path.is_absolute():
        storage_path = (output_root / storage_path).resolve()
    storage = SearchStorageConfig(type=storage_type, path=storage_path)

    execution_cfg = search_cfg.get("execution") or {}
    max_parallel_trials = int(execution_cfg.get("max_parallel_trials", 0))
    if max_parallel_trials < 1:
        raise ValueError("search.execution.max_parallel_trials must be at least 1.")
    execution = SearchExecutionConfig(max_parallel_trials=max_parallel_trials)

    objective_cfg = search_cfg.get("objective") or {}
    score_formula = str(objective_cfg.get("score_formula", "")).strip()
    if score_formula not in SUPPORTED_SCORE_FORMULAS:
        raise ValueError(
            f"search.objective.score_formula must be one of {sorted(SUPPORTED_SCORE_FORMULAS)}."
        )
    objective = SearchObjectiveConfig(
        experiment_name=str(objective_cfg.get("experiment_name", "")).strip(),
        split=str(objective_cfg.get("split", "")).strip(),
        metric=str(objective_cfg.get("metric", "")).strip(),
        success_threshold=float(objective_cfg.get("success_threshold", 0.0)),
        score_formula=score_formula,
    )
    if not objective.experiment_name:
        raise ValueError("search.objective.experiment_name is required.")
    if not objective.split:
        raise ValueError("search.objective.split is required.")
    if not objective.metric:
        raise ValueError("search.objective.metric is required.")

    raw_parameters = search_cfg.get("parameters") or []
    parameters = tuple(_build_parameter_spec(item) for item in raw_parameters)
    enabled_parameters = [parameter for parameter in parameters if parameter.enabled]
    if not enabled_parameters:
        raise ValueError("search.parameters must include at least one enabled parameter.")
    _validate_parameter_conflicts(enabled_parameters)
    _validate_sampler_parameters(sampler, execution, enabled_parameters)

    return SearchConfig(
        config_path=resolved_config_path,
        name=name,
        base_batch_config=base_batch_config,
        output_root=output_root,
        n_trials=n_trials,
        sampler=sampler,
        storage=storage,
        execution=execution,
        objective=objective,
        parameters=parameters,
    )


def _build_parameter_spec(raw: dict[str, Any]) -> SearchParameterSpec:
    if not isinstance(raw, dict):
        raise TypeError("Each search.parameters entry must be a mapping.")
    parameter_name = str(raw.get("name", "")).strip()
    if not parameter_name:
        raise ValueError("Each search.parameters entry must define a non-empty 'name'.")
    parameter_key = str(raw.get("key", "")).strip()
    if not parameter_key:
        raise ValueError(f"search.parameters[{parameter_name}].key is required.")
    parameter_type = str(raw.get("type", "")).strip().lower()
    if parameter_type not in SUPPORTED_PARAMETER_TYPES:
        raise ValueError(
            f"search.parameters[{parameter_name}].type must be one of {sorted(SUPPORTED_PARAMETER_TYPES)}."
        )
    scope = _normalize_scope(parameter_name, raw.get("scope"))
    enabled = bool(raw.get("enabled", False))

    if parameter_type == "int":
        low = int(raw["low"])
        high = int(raw["high"])
        step = int(raw.get("step", 1))
        if high < low:
            raise ValueError(f"search.parameters[{parameter_name}] requires high >= low.")
        if step < 1:
            raise ValueError(f"search.parameters[{parameter_name}].step must be at least 1.")
        return SearchParameterSpec(
            name=parameter_name,
            key=parameter_key,
            enabled=enabled,
            scope=scope,
            parameter_type=parameter_type,
            low=low,
            high=high,
            step=step,
            log=bool(raw.get("log", False)),
        )

    if parameter_type == "float":
        low = float(raw["low"])
        high = float(raw["high"])
        if high <= low:
            raise ValueError(f"search.parameters[{parameter_name}] requires high > low.")
        return SearchParameterSpec(
            name=parameter_name,
            key=parameter_key,
            enabled=enabled,
            scope=scope,
            parameter_type=parameter_type,
            low=low,
            high=high,
            log=bool(raw.get("log", False)),
        )

    raw_choices = raw.get("choices")
    if raw_choices is None and parameter_type == "bool":
        raw_choices = [True, False]
    if not isinstance(raw_choices, list) or not raw_choices:
        raise ValueError(f"search.parameters[{parameter_name}].choices must be a non-empty list.")
    return SearchParameterSpec(
        name=parameter_name,
        key=parameter_key,
        enabled=enabled,
        scope=scope,
        parameter_type=parameter_type,
        choices=tuple(raw_choices),
    )


def _normalize_scope(parameter_name: str, raw_scope: Any) -> tuple[str, ...]:
    if not isinstance(raw_scope, list) or not raw_scope:
        raise ValueError(f"search.parameters[{parameter_name}].scope must be a non-empty list.")
    scope = tuple(str(item).strip() for item in raw_scope if str(item).strip())
    if not scope:
        raise ValueError(f"search.parameters[{parameter_name}].scope must not be empty.")
    if "all" in scope and len(scope) > 1:
        raise ValueError(
            f"search.parameters[{parameter_name}].scope cannot mix 'all' with explicit experiment names."
        )
    return scope


def _validate_parameter_conflicts(parameters: list[SearchParameterSpec]) -> None:
    seen_names: set[str] = set()
    for parameter in parameters:
        if parameter.name in seen_names:
            raise ValueError(f"Duplicate enabled parameter name detected: {parameter.name}")
        seen_names.add(parameter.name)

    for index, parameter in enumerate(parameters):
        for other in parameters[index + 1 :]:
            if parameter.overlaps_with(other):
                raise ValueError(
                    "Enabled search parameters cannot target the same key with overlapping scopes: "
                    f"{parameter.name} ({parameter.key}) and {other.name} ({other.key})"
                )


def _validate_sampler_parameters(
    sampler: SearchSamplerConfig,
    execution: SearchExecutionConfig,
    parameters: list[SearchParameterSpec],
) -> None:
    if sampler.name != "grid":
        return
    _build_grid_search_space(tuple(parameters))


def _build_grid_search_space(parameters: tuple[SearchParameterSpec, ...]) -> dict[str, list[Any]]:
    search_space: dict[str, list[Any]] = {}
    for parameter in parameters:
        if not parameter.enabled:
            continue
        if parameter.parameter_type not in {"categorical", "bool"}:
            raise ValueError(
                "Grid search requires enabled parameters to use type 'categorical' or 'bool' with explicit choices: "
                f"{parameter.name} uses type '{parameter.parameter_type}'."
            )
        if not parameter.choices:
            raise ValueError(f"Grid search parameter '{parameter.name}' must define non-empty choices.")
        search_space[parameter.name] = list(parameter.choices)
    if not search_space:
        raise ValueError("Grid search requires at least one enabled parameter.")
    return search_space


def _resolve_repo_path(project_root: Path, path_ref: str | Path) -> Path:
    candidate = Path(path_ref)
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate
