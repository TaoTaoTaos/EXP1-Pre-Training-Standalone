from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import pytest

from lakeice_ncde.search.config import load_search_config
from lakeice_ncde.search.objective import FAILURE_SCORE, build_trial_execution_plan, compute_r2_dominant_composite_score
from lakeice_ncde.search.runner import run_search
from lakeice_ncde.utils.io import load_dataframe, save_dataframe, save_json, save_yaml
from lakeice_ncde.utils.paths import build_pdf_name, build_prefixed_artifact_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAMES = [
    "EXP0_pretrain_autoreg",
    "EXP1_transfer_autoreg",
    "EXP2_transfer_autoreg_stefan",
]


def _stub_batch_runner(plan, logger) -> dict:
    time.sleep(0.15)
    if plan.sampled_parameters.get("force_failure_all"):
        raise RuntimeError("Simulated batch runner failure.")

    batch_run_name = "[01]_Run-ALL_20260416_120000"
    batch_run_dir = plan.batch_output_root / "Run-ALL" / batch_run_name
    batch_run_dir.mkdir(parents=True, exist_ok=True)

    summary_excel_path = batch_run_dir / build_prefixed_artifact_name(batch_run_name, ".xlsx", suffix="summary")
    pd.DataFrame([{"trial_number": plan.trial_number}]).to_excel(summary_excel_path, index=False)

    summary_pdf_path = batch_run_dir / build_pdf_name(batch_run_name)
    summary_pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    experiments: list[dict] = []
    copied_pdf_paths: list[str] = []
    for sequence, experiment_override in enumerate(plan.experiment_overrides, start=1):
        run_dir, pdf_path = _create_fake_search_run(
            plan=plan,
            experiment_name=experiment_override["experiment_name"],
            sequence=sequence,
        )
        copied_destination = batch_run_dir / pdf_path.name
        copied_destination.write_bytes(pdf_path.read_bytes())
        copied_pdf_paths.append(str(copied_destination))
        experiments.append(
            {
                "experiment_name": experiment_override["experiment_name"],
                "run_dir": str(run_dir),
                "pdf_path": str(pdf_path),
                "config_path": experiment_override["config_path"],
                "override_paths": experiment_override["override_paths"],
                "set_values": experiment_override["set_values"],
            }
        )

    return {
        "batch_run_name": batch_run_name,
        "batch_run_dir": str(batch_run_dir),
        "summary_excel_path": str(summary_excel_path),
        "summary_pdf_path": str(summary_pdf_path),
        "copied_pdf_paths": copied_pdf_paths,
        "experiments": experiments,
    }


def _create_fake_search_run(plan, experiment_name: str, sequence: int) -> tuple[Path, Path]:
    run_name = f"[{sequence:02d}]_{experiment_name}_20260416_120000"
    run_dir = plan.batch_output_root / experiment_name / run_name
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    learning_rate = float(plan.sampled_parameters.get("learning_rate_all", 2.0e-4))
    weight_decay = float(plan.sampled_parameters.get("weight_decay_all", 5.0e-4))
    window_days = int(plan.sampled_parameters.get("window_days_all", 7))
    hidden_channels = int(plan.sampled_parameters.get("hidden_channels_all", 56))
    hidden_hidden_channels = int(plan.sampled_parameters.get("hidden_hidden_channels_all", 112))
    num_hidden_layers = int(plan.sampled_parameters.get("num_hidden_layers_all", 5))
    huber_delta = float(plan.sampled_parameters.get("huber_delta_all", 0.1))
    lambda_st = float(plan.sampled_parameters.get("lambda_st_exp2", 0.02))
    init_kappa = float(plan.sampled_parameters.get("init_kappa_exp2", 0.0085))
    grow_temp_threshold = float(plan.sampled_parameters.get("grow_temp_threshold_celsius_exp2", -0.5))

    metrics = _compute_fake_metrics(
        experiment_name=experiment_name,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        hidden_hidden_channels=hidden_hidden_channels,
        num_hidden_layers=num_hidden_layers,
        lambda_st=lambda_st,
        init_kappa=init_kappa,
        grow_temp_threshold=grow_temp_threshold,
    )

    save_yaml(
        {
            "experiment": {"name": experiment_name},
            "custom_split": {"target_lake_test_start": "2026-01-01"},
            "coeffs": {"interpolation": "hermite"},
            "model": {
                "method": "rk4",
                "hidden_channels": hidden_channels,
                "hidden_hidden_channels": hidden_hidden_channels,
                "num_hidden_layers": num_hidden_layers,
                "dropout": 0.05,
            },
            "window": {"window_days": window_days},
            "features": {
                "target_transform": "none",
                "feature_columns": ["Air_Temperature_celsius"],
            },
            "train": {
                "device": "cpu",
                "batch_parallel": True,
                "batch_size": 16,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_epochs": 200,
                "loss": "huber",
                "huber_delta": huber_delta,
                "physics_loss": {
                    "enabled": experiment_name == "EXP2_transfer_autoreg_stefan",
                    "rule": "stefan_growth_residual",
                    "lambda_st": lambda_st,
                    "lambda_nn": 1.0,
                    "init_kappa": init_kappa,
                    "grow_temp_threshold_celsius": grow_temp_threshold,
                    "min_prev_ice_m": 1.0e-3,
                    "prev_ice_column": "ice_prev_m",
                    "gap_days_column": "ice_prev_gap_days",
                    "temperature_column": "Air_Temperature_celsius",
                    "prev_available_column": "ice_prev_available",
                },
            },
        },
        run_dir / "config_merged.yaml",
    )
    save_dataframe(
        pd.DataFrame(
            [
                {
                    "split": "val",
                    "loss": round(metrics["test_rmse"] * 0.7, 6),
                    "rmse": metrics["val_rmse"],
                    "mae": round(metrics["val_rmse"] * 0.7, 6),
                    "r2": round(metrics["test_r2"] - 0.03, 6),
                    "bias": 0.0,
                    "negative_count": max(metrics["negative_count"] - 1, 0),
                },
                {
                    "split": "test",
                    "loss": round(metrics["test_rmse"] * 0.8, 6),
                    "rmse": metrics["test_rmse"],
                    "mae": round(metrics["test_rmse"] * 0.75, 6),
                    "r2": metrics["test_r2"],
                    "bias": 0.0,
                    "negative_count": metrics["negative_count"],
                },
            ]
        ),
        run_dir / "metrics.csv",
    )
    save_json(
        {
            "best_epoch": 12,
            "best_val_rmse": metrics["val_rmse"],
            "duration_seconds": 8.5 + sequence,
            "final_val_loss": round(metrics["val_rmse"] * 0.7, 6),
            "final_test_loss": round(metrics["test_rmse"] * 0.8, 6),
        },
        run_dir / "run_summary.json",
    )
    save_json(
        {"split_name": experiment_name, "target_lake": "Xiaoxingkai"},
        artifacts_dir / "run_manifest.json",
    )
    pdf_path = run_dir / build_pdf_name(run_name)
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    return run_dir, pdf_path


def _compute_fake_metrics(
    experiment_name: str,
    learning_rate: float,
    hidden_channels: int,
    hidden_hidden_channels: int,
    num_hidden_layers: int,
    lambda_st: float,
    init_kappa: float,
    grow_temp_threshold: float,
) -> dict[str, float]:
    lr_score = max(0.0, 1.0 - abs(math.log10(learning_rate) + 3.7))
    hidden_score = min(hidden_channels / 96.0, 1.0)
    depth_penalty = abs(num_hidden_layers - 5) * 0.02
    wide_bonus = min(hidden_hidden_channels / 160.0, 1.0) * 0.10
    physics_bonus = 0.0
    if experiment_name == "EXP2_transfer_autoreg_stefan":
        physics_bonus = (
            lambda_st * 3.0
            + init_kappa * 4.0
            - abs(grow_temp_threshold + 0.5) * 0.03
        )
    experiment_bonus = {
        "EXP0_pretrain_autoreg": 0.00,
        "EXP1_transfer_autoreg": 0.04,
        "EXP2_transfer_autoreg_stefan": 0.08,
    }[experiment_name]
    test_r2 = round(
        -0.08 + 0.18 * lr_score + 0.16 * hidden_score + wide_bonus - depth_penalty + physics_bonus + experiment_bonus,
        6,
    )
    test_rmse = round(max(0.05, 0.36 - max(test_r2, -0.2) * 0.25), 6)
    negative_count = int(max(0, round(4 - test_r2 * 12)))
    val_rmse = round(test_rmse * 0.9, 6)
    return {
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "negative_count": negative_count,
        "val_rmse": val_rmse,
    }


def _write_search_config(
    tmp_path: Path,
    *,
    config_name: str,
    output_root_name: str,
    n_trials: int,
    max_parallel_trials: int,
    parameters: list[dict],
) -> Path:
    config_path = tmp_path / config_name
    save_yaml(
        {
            "search": {
                "name": "test_search_study",
                "base_batch_config": str(PROJECT_ROOT / "configs" / "experiments" / "Run-ALL.yaml"),
                "output_root": str(tmp_path / output_root_name),
                "n_trials": n_trials,
                "sampler": {"name": "tpe", "seed": 7},
                "storage": {"type": "journal", "path": "artifacts/study.journal"},
                "execution": {"max_parallel_trials": max_parallel_trials},
                "objective": {
                    "experiment_name": "EXP2_transfer_autoreg_stefan",
                    "split": "test",
                    "metric": "r2",
                    "success_threshold": 0.0,
                    "score_formula": "r2_dominant_composite",
                },
                "parameters": parameters,
            }
        },
        config_path,
    )
    return config_path


def _default_parameters() -> list[dict]:
    return [
        {
            "name": "learning_rate_all",
            "key": "train.learning_rate",
            "enabled": True,
            "scope": ["all"],
            "type": "float",
            "low": 1.0e-5,
            "high": 1.0e-3,
            "log": True,
        },
        {
            "name": "hidden_channels_all",
            "key": "model.hidden_channels",
            "enabled": True,
            "scope": ["all"],
            "type": "int",
            "low": 32,
            "high": 96,
            "step": 32,
        },
        {
            "name": "lambda_st_exp2",
            "key": "train.physics_loss.lambda_st",
            "enabled": True,
            "scope": ["EXP2_transfer_autoreg_stefan"],
            "type": "float",
            "low": 0.01,
            "high": 0.05,
            "log": False,
        },
    ]


def test_example_search_config_loads_expected_defaults() -> None:
    config_path = next((PROJECT_ROOT / "configs" / "search").glob("*.yaml"))
    config = load_search_config(PROJECT_ROOT, config_path)

    assert config.name == "EXP2_test_r2_search"
    assert config.n_trials == 64
    assert config.execution.max_parallel_trials == 4
    assert config.objective.experiment_name == "EXP2_transfer_autoreg_stefan"
    assert config.objective.metric == "r2"
    assert config.storage.path.name == "study.journal"
    assert any(parameter.name == "lambda_st_exp2" and parameter.enabled for parameter in config.parameters)


def test_search_config_rejects_mixed_all_and_explicit_scope(tmp_path: Path) -> None:
    config_path = _write_search_config(
        tmp_path,
        config_name="invalid_search.yaml",
        output_root_name="search-output",
        n_trials=2,
        max_parallel_trials=1,
        parameters=[
            {
                "name": "bad_scope",
                "key": "train.learning_rate",
                "enabled": True,
                "scope": ["all", "EXP2_transfer_autoreg_stefan"],
                "type": "float",
                "low": 1.0e-5,
                "high": 1.0e-3,
                "log": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="cannot mix 'all'"):
        load_search_config(PROJECT_ROOT, config_path)


def test_trial_execution_plan_applies_global_and_exp2_only_overrides_without_mutating_source_configs(tmp_path: Path) -> None:
    config_path = _write_search_config(
        tmp_path,
        config_name="search.yaml",
        output_root_name="search-output",
        n_trials=2,
        max_parallel_trials=1,
        parameters=_default_parameters(),
    )
    search_config = load_search_config(PROJECT_ROOT, config_path)
    source_paths = [
        PROJECT_ROOT / "configs" / "experiments" / name
        for name in [
            "EXP0_pretrain_autoreg.yaml",
            "EXP1_transfer_autoreg.yaml",
            "EXP2_transfer_autoreg_stefan.yaml",
        ]
    ]
    before_text = {path: path.read_text(encoding="utf-8") for path in source_paths}

    plan = build_trial_execution_plan(
        project_root=PROJECT_ROOT,
        search_config=search_config,
        trial_number=0,
        trial_dir=tmp_path / "trial_0000",
        sampled_parameters={
            "learning_rate_all": 3.0e-4,
            "hidden_channels_all": 64,
            "lambda_st_exp2": 0.03,
        },
    )

    overrides_by_experiment = {
        item["experiment_name"]: item["set_values"]
        for item in plan.experiment_overrides
    }
    assert "train.learning_rate=0.0003" in overrides_by_experiment["EXP0_pretrain_autoreg"]
    assert "model.hidden_channels=64" in overrides_by_experiment["EXP1_transfer_autoreg"]
    assert "train.physics_loss.lambda_st=0.03" not in overrides_by_experiment["EXP0_pretrain_autoreg"]
    assert "train.physics_loss.lambda_st=0.03" in overrides_by_experiment["EXP2_transfer_autoreg_stefan"]

    after_text = {path: path.read_text(encoding="utf-8") for path in source_paths}
    assert after_text == before_text


def test_compute_r2_dominant_composite_score_matches_formula() -> None:
    score = compute_r2_dominant_composite_score(
        test_r2=0.12,
        test_rmse=0.30,
        test_negative_count=4,
    )

    assert score == pytest.approx(0.12 - 0.20 * 0.30 - 0.05 * math.log1p(4))


def test_run_search_writes_trial_records_and_supports_resume(tmp_path: Path) -> None:
    first_config = _write_search_config(
        tmp_path,
        config_name="search_first.yaml",
        output_root_name="resume-search",
        n_trials=3,
        max_parallel_trials=2,
        parameters=_default_parameters(),
    )
    second_config = _write_search_config(
        tmp_path,
        config_name="search_second.yaml",
        output_root_name="resume-search",
        n_trials=5,
        max_parallel_trials=2,
        parameters=_default_parameters(),
    )

    first_result = run_search(PROJECT_ROOT, first_config, batch_runner=_stub_batch_runner)
    assert first_result["recorded_trial_count"] == 3

    second_result = run_search(PROJECT_ROOT, second_config, batch_runner=_stub_batch_runner)
    search_root = Path(second_result["search_root"])
    master_df = load_dataframe(search_root / "trials_master.csv")
    parameter_df = load_dataframe(search_root / "trial_parameters.csv")
    experiment_df = load_dataframe(search_root / "trial_experiments.csv")
    top_trials_df = load_dataframe(search_root / "top_trials.csv")

    assert len(master_df) == 5
    assert sorted(master_df["trial_number"].tolist()) == [0, 1, 2, 3, 4]
    assert parameter_df["trial_number"].nunique() == 5
    assert len(experiment_df) == 15
    expected_top = top_trials_df.sort_values(
        by=["threshold_met", "score", "trial_number"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(top_trials_df.reset_index(drop=True), expected_top)
    assert master_df["worker_pid"].nunique() >= 2
    assert (search_root / "study_summary.json").exists()
    assert (search_root / "artifacts" / "study.journal").exists()
    assert (search_root / "trials" / "trial_0004" / "trial_config_resolved.yaml").exists()
    assert (search_root / "trials" / "trial_0004" / "trial_overrides.yaml").exists()
    assert (search_root / "trials" / "trial_0004" / "trial_metadata.json").exists()


def test_run_search_records_failed_trials_in_master_csv(tmp_path: Path) -> None:
    parameters = _default_parameters() + [
        {
            "name": "force_failure_all",
            "key": "custom.force_failure",
            "enabled": True,
            "scope": ["all"],
            "type": "bool",
            "choices": [True],
        }
    ]
    config_path = _write_search_config(
        tmp_path,
        config_name="search_failure.yaml",
        output_root_name="failed-search",
        n_trials=1,
        max_parallel_trials=1,
        parameters=parameters,
    )

    result = run_search(PROJECT_ROOT, config_path, batch_runner=_stub_batch_runner)
    search_root = Path(result["search_root"])
    master_df = load_dataframe(search_root / "trials_master.csv")

    assert len(master_df) == 1
    assert master_df.loc[0, "trial_status"] == "failed"
    assert master_df.loc[0, "threshold_met"] in (False, 0)
    assert float(master_df.loc[0, "score"]) == FAILURE_SCORE
    assert "Simulated batch runner failure." in str(master_df.loc[0, "error_message"])
