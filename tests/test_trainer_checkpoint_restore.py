from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from lakeice_ncde.data.datasets import Batch
from lakeice_ncde.training.engine import Trainer


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, coeffs):  # pragma: no cover - forward is unused after monkeypatching _run_epoch
        del coeffs
        return self.bias.unsqueeze(0)


class DummyLoader:
    def __init__(self, dataset_size: int = 1) -> None:
        self.dataset = list(range(dataset_size))

    def __len__(self) -> int:
        return 1


def _base_config(
    *,
    physics_enabled: bool,
    physics_mode: str = "legacy_stefan",
    enable_decay: bool = False,
) -> dict:
    return {
        "train": {
            "device": "cpu",
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "gradient_clip_norm": None,
            "loss": "mse",
            "huber_delta": 1.0,
            "early_stopping": {"patience": 10, "min_delta": 1.0e-8},
            "scheduler": {"name": "none", "factor": 0.5, "patience": 2, "min_lr": 1.0e-6},
            "physics_loss": {
                "enabled": physics_enabled,
                "mode": physics_mode,
                "rule": "stefan_growth_residual",
                "lambda_st": 1.0,
                "lambda_curve_grow": 1.0,
                "lambda_curve_decay": 0.2,
                "lambda_nn": 1.0,
                "init_kappa": 1.0,
                "init_alpha": 1.0,
                "init_alpha_decay": 1.0,
                "enable_decay": enable_decay,
                "min_prev_ice_m": 1.0e-3,
                "grow_temp_threshold_celsius": -0.5,
                "prev_ice_column": "ice_prev_m",
                "gap_days_column": "ice_prev_gap_days",
                "temperature_column": "Air_Temperature_celsius",
                "prev_available_column": "ice_prev_available",
            },
        },
        "coeffs": {"interpolation": "hermite"},
        "window": {"window_days": 7},
        "features": {"target_transform": "none"},
        "debug": {"enabled": False},
    }


@pytest.fixture
def dummy_logger():
    class _Logger:
        def info(self, *args, **kwargs) -> None:
            return None

        def warning(self, *args, **kwargs) -> None:
            return None

    return _Logger()


def test_fit_reports_best_checkpoint_val_loss(tmp_path, monkeypatch, dummy_logger) -> None:
    config = _base_config(physics_enabled=False)
    trainer = Trainer(DummyModel(), config, tmp_path, dummy_logger)

    def fake_run_epoch(self, loader, train, epoch, max_epochs) -> float:
        del self, loader, train, epoch, max_epochs
        return 0.0

    prediction_sequence = [
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.4]}), {"rmse": 0.4, "mae": 0.4, "r2": 0.0, "bias": 0.4, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
    ]

    def fake_predict_loader(model, loader, device, target_transform):
        del model, loader, device, target_transform
        return prediction_sequence.pop(0)

    monkeypatch.setattr(Trainer, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr("lakeice_ncde.training.engine.predict_loader", fake_predict_loader)

    trainer.fit(
        train_loader=DummyLoader(),
        val_loader=DummyLoader(),
        test_loader=None,
        target_transform="none",
    )

    run_summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert run_summary["best_epoch"] == 1
    assert run_summary["final_val_loss"] == pytest.approx(0.01)
    assert float(metrics.loc[metrics["split"] == "val", "loss"].iloc[0]) == pytest.approx(0.01)


def test_fit_restores_best_theta_kappa_before_reporting(tmp_path, monkeypatch, dummy_logger) -> None:
    config = _base_config(physics_enabled=True)
    trainer = Trainer(DummyModel(), config, tmp_path, dummy_logger)

    best_theta = torch.tensor(-2.0, dtype=torch.float32)
    latest_theta = torch.tensor(3.0, dtype=torch.float32)

    def fake_run_epoch(self, loader, train, epoch, max_epochs) -> float:
        del loader, train, max_epochs
        with torch.no_grad():
            assert self.theta_kappa is not None
            self.theta_kappa.copy_(best_theta if epoch == 1 else latest_theta)
        return 0.0

    prediction_sequence = [
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.4]}), {"rmse": 0.4, "mae": 0.4, "r2": 0.0, "bias": 0.4, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
    ]

    def fake_predict_loader(model, loader, device, target_transform):
        del model, loader, device, target_transform
        return prediction_sequence.pop(0)

    monkeypatch.setattr(Trainer, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr("lakeice_ncde.training.engine.predict_loader", fake_predict_loader)

    trainer.fit(
        train_loader=DummyLoader(),
        val_loader=DummyLoader(),
        test_loader=None,
        target_transform="none",
    )

    run_summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["best_epoch"] == 1
    assert run_summary["physics_kappa"] == pytest.approx(torch.nn.functional.softplus(best_theta).item())


def test_fit_restores_best_theta_alpha_before_reporting(tmp_path, monkeypatch, dummy_logger) -> None:
    config = _base_config(
        physics_enabled=True,
        physics_mode="tc2020_curve",
        enable_decay=True,
    )
    trainer = Trainer(DummyModel(), config, tmp_path, dummy_logger)

    best_theta_alpha = torch.tensor(-1.5, dtype=torch.float32)
    latest_theta_alpha = torch.tensor(2.5, dtype=torch.float32)
    best_theta_alpha_decay = torch.tensor(-2.2, dtype=torch.float32)
    latest_theta_alpha_decay = torch.tensor(1.8, dtype=torch.float32)

    def fake_run_epoch(self, loader, train, epoch, max_epochs) -> float:
        del loader, train, max_epochs
        with torch.no_grad():
            assert self.theta_alpha is not None
            assert self.theta_alpha_decay is not None
            self.theta_alpha.copy_(best_theta_alpha if epoch == 1 else latest_theta_alpha)
            self.theta_alpha_decay.copy_(
                best_theta_alpha_decay if epoch == 1 else latest_theta_alpha_decay
            )
        return 0.0

    prediction_sequence = [
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.4]}), {"rmse": 0.4, "mae": 0.4, "r2": 0.0, "bias": 0.4, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
    ]

    def fake_predict_loader(model, loader, device, target_transform):
        del model, loader, device, target_transform
        return prediction_sequence.pop(0)

    monkeypatch.setattr(Trainer, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr("lakeice_ncde.training.engine.predict_loader", fake_predict_loader)

    trainer.fit(
        train_loader=DummyLoader(),
        val_loader=DummyLoader(),
        test_loader=None,
        target_transform="none",
    )

    run_summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["best_epoch"] == 1
    assert run_summary["physics_loss_mode"] == "tc2020_curve"
    assert run_summary["physics_lambda_curve_grow"] == pytest.approx(1.0)
    assert run_summary["physics_lambda_curve_decay"] == pytest.approx(0.2)
    assert run_summary["physics_enable_decay"] is True
    assert run_summary["physics_alpha"] == pytest.approx(
        torch.nn.functional.softplus(best_theta_alpha).item()
    )
    assert run_summary["physics_alpha_decay"] == pytest.approx(
        torch.nn.functional.softplus(best_theta_alpha_decay).item()
    )


def test_fit_restores_best_tc2020_plus_theta_kappa_before_reporting(
    tmp_path,
    monkeypatch,
    dummy_logger,
) -> None:
    config = _base_config(
        physics_enabled=True,
        physics_mode="tc2020_curve",
        enable_decay=False,
    )
    config["train"]["physics_loss"]["enable_stefan_grow"] = True
    trainer = Trainer(DummyModel(), config, tmp_path, dummy_logger)

    best_theta_alpha = torch.tensor(-1.5, dtype=torch.float32)
    latest_theta_alpha = torch.tensor(2.5, dtype=torch.float32)
    best_theta_kappa = torch.tensor(-2.0, dtype=torch.float32)
    latest_theta_kappa = torch.tensor(3.0, dtype=torch.float32)

    def fake_run_epoch(self, loader, train, epoch, max_epochs) -> float:
        del loader, train, max_epochs
        with torch.no_grad():
            assert self.theta_alpha is not None
            assert self.theta_kappa is not None
            self.theta_alpha.copy_(best_theta_alpha if epoch == 1 else latest_theta_alpha)
            self.theta_kappa.copy_(best_theta_kappa if epoch == 1 else latest_theta_kappa)
        return 0.0

    prediction_sequence = [
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.4]}), {"rmse": 0.4, "mae": 0.4, "r2": 0.0, "bias": 0.4, "negative_count": 0.0}),
        (pd.DataFrame({"y_true": [1.0], "y_pred": [1.1]}), {"rmse": 0.1, "mae": 0.1, "r2": 0.0, "bias": 0.1, "negative_count": 0.0}),
    ]

    def fake_predict_loader(model, loader, device, target_transform):
        del model, loader, device, target_transform
        return prediction_sequence.pop(0)

    monkeypatch.setattr(Trainer, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr("lakeice_ncde.training.engine.predict_loader", fake_predict_loader)

    trainer.fit(
        train_loader=DummyLoader(),
        val_loader=DummyLoader(),
        test_loader=None,
        target_transform="none",
    )

    run_summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["best_epoch"] == 1
    assert run_summary["physics_enable_stefan_grow"] is True
    assert run_summary["physics_kappa"] == pytest.approx(
        torch.nn.functional.softplus(best_theta_kappa).item()
    )


def test_rollout_stability_loss_replaces_next_prev_ice(
    tmp_path,
    monkeypatch,
    dummy_logger,
) -> None:
    config = _base_config(physics_enabled=False)
    config["train"]["physics_loss"]["enable_rollout_stability"] = True
    config["train"]["physics_loss"]["rollout_stability_huber_beta"] = 0.1
    trainer = Trainer(DummyModel(), config, tmp_path, dummy_logger)

    next_window = torch.zeros((2, 4), dtype=torch.float32)
    batch = Batch(
        coeffs=[],
        coeff_groups=[],
        targets=torch.tensor([0.4], dtype=torch.float32),
        targets_raw=torch.tensor([0.4], dtype=torch.float32),
        metadata=[{}],
        rollout_context={
            "current_indices": torch.tensor([0], dtype=torch.long),
            "next_windows": [next_window],
            "next_targets_raw": torch.tensor([0.5], dtype=torch.float32),
            "feature_columns": [
                "ice_prev_m",
                "ice_prev_gap_days",
                "ice_prev_available",
            ],
            "feature_scaler": {
                "mean": {
                    "ice_prev_m": 0.0,
                    "ice_prev_gap_days": 0.0,
                    "ice_prev_available": 0.0,
                },
                "std": {
                    "ice_prev_m": 0.2,
                    "ice_prev_gap_days": 1.0,
                    "ice_prev_available": 1.0,
                },
            },
        },
    )

    def fake_predict_paths(self, paths):
        del self
        torch.testing.assert_close(paths[0][-1, 1], torch.tensor(2.0))
        torch.testing.assert_close(paths[0][-1, 3], torch.tensor(1.0))
        return torch.tensor([0.6], dtype=torch.float32)

    monkeypatch.setattr(Trainer, "_predict_paths", fake_predict_paths)

    loss = trainer._compute_rollout_stability_loss(
        batch,
        torch.tensor([0.4], dtype=torch.float32),
    )

    expected = F.smooth_l1_loss(
        torch.tensor([0.6], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        beta=0.1,
    )
    torch.testing.assert_close(loss, expected)


def test_tc2020_trainer_requires_explicit_init_alpha(tmp_path, dummy_logger) -> None:
    config = _base_config(
        physics_enabled=True,
        physics_mode="tc2020_curve",
        enable_decay=False,
    )
    del config["train"]["physics_loss"]["init_alpha"]

    with pytest.raises(ValueError, match="init_alpha"):
        Trainer(DummyModel(), config, tmp_path, dummy_logger)


def test_tc2020_trainer_requires_explicit_init_alpha_decay_when_decay_enabled(
    tmp_path,
    dummy_logger,
) -> None:
    config = _base_config(
        physics_enabled=True,
        physics_mode="tc2020_curve",
        enable_decay=True,
    )
    del config["train"]["physics_loss"]["init_alpha_decay"]

    with pytest.raises(ValueError, match="init_alpha_decay"):
        Trainer(DummyModel(), config, tmp_path, dummy_logger)
