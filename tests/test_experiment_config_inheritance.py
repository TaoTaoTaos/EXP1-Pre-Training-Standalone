from __future__ import annotations

from pathlib import Path

from lakeice_ncde.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_experiment_config(name: str) -> dict:
    return load_config(
        project_root=PROJECT_ROOT,
        config_path=(PROJECT_ROOT / "configs" / "experiments" / Path(name)).resolve(),
        override_paths=[],
    )


def test_exp0_keeps_target_lake_fully_held_out() -> None:
    config = _load_experiment_config("EXP0_pretrain_autoreg.yaml")

    assert config["experiment"]["name"] == "EXP0_pretrain_autoreg"
    assert config["custom_split"]["target_lake_test_start"] is None
    assert config["paths"]["prepared_csv"].endswith("prepared_data_EXP0_pretrain_autoreg.csv")


def test_exp1_inherits_from_exp0_and_restores_target_cutoff() -> None:
    config = _load_experiment_config("EXP1_transfer_autoreg.yaml")

    assert config["experiment"]["name"] == "EXP1_transfer_autoreg"
    assert config["custom_split"]["target_lake_test_start"] == "2026-01-01"
    assert config["train"]["physics_loss"]["enabled"] is False
    assert config["paths"]["prepared_csv"].endswith("prepared_data_EXP1_transfer_autoreg.csv")


def test_exp2_inherits_from_exp0_and_enables_physics() -> None:
    config = _load_experiment_config("EXP2_transfer_autoreg_stefan.yaml")

    assert config["experiment"]["name"] == "EXP2_transfer_autoreg_stefan"
    assert config["custom_split"]["target_lake_test_start"] == "2026-01-01"
    assert config["train"]["physics_loss"]["enabled"] is True
    assert config["train"]["physics_loss"]["lambda_st"] == 0.02
    assert config["paths"]["prepared_csv"].endswith("prepared_data_EXP2_transfer_autoreg_stefan.csv")


def test_old_aliases_resolve_to_new_experiment_names() -> None:
    exp0 = _load_experiment_config("Olds/EXP0_transfer_autoreg.yaml")
    exp1 = _load_experiment_config("Olds/EXP1_history_autoreg.yaml")
    exp2 = _load_experiment_config("Olds/EXP2_history_autoreg_stefan.yaml")

    assert exp0["experiment"]["name"] == "EXP0_pretrain_autoreg"
    assert exp1["experiment"]["name"] == "EXP1_transfer_autoreg"
    assert exp2["experiment"]["name"] == "EXP2_transfer_autoreg_stefan"
