from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lakeice_ncde.data.datasets import Batch
from lakeice_ncde.evaluation.predict import predict_loader
from lakeice_ncde.training.checkpoints import load_checkpoint, save_checkpoint
from lakeice_ncde.training.history import HistoryLogger
from lakeice_ncde.training.losses import (
    build_loss,
    check_loss_is_finite,
    compute_physics_loss,
    inverse_softplus,
)
from lakeice_ncde.training.schedulers import build_scheduler
from lakeice_ncde.utils.io import save_dataframe, save_json


@dataclass
class TrainArtifacts:
    """Returned training artifacts."""

    run_dir: Path
    metrics_path: Path
    history_path: Path
    best_ckpt_path: Path
    latest_ckpt_path: Path
    val_predictions_path: Path
    test_predictions_path: Path
    per_lake_metrics_path: Path
    run_summary_path: Path


class Trainer:
    """Manual trainer for NeuralCDE experiments."""

    def __init__(self, model: torch.nn.Module, config: dict, run_dir: Path, logger) -> None:
        self.model = model
        self.config = config
        self.run_dir = run_dir
        self.logger = logger
        self.device = self._resolve_device(config["train"]["device"])
        self.model.to(self.device)
        self.criterion = build_loss(config)
        physics_cfg = config["train"].get("physics_loss", {})
        self.physics_loss_enabled = bool(physics_cfg.get("enabled", False))
        self.physics_loss_mode = str(physics_cfg.get("mode", "legacy_stefan"))
        self.theta_kappa: nn.Parameter | None = None
        self.theta_alpha: nn.Parameter | None = None
        self.theta_alpha_decay: nn.Parameter | None = None
        optimizer_parameters = list(self.model.parameters())
        if self.physics_loss_enabled:
            if self.physics_loss_mode == "legacy_stefan":
                init_kappa = float(physics_cfg.get("init_kappa", 1.0))
                self.theta_kappa = nn.Parameter(
                    torch.tensor(
                        inverse_softplus(init_kappa),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                optimizer_parameters.append(self.theta_kappa)
            elif self.physics_loss_mode == "tc2020_curve":
                init_alpha = float(_require_physics_config_value(physics_cfg, "init_alpha"))
                self.theta_alpha = nn.Parameter(
                    torch.tensor(
                        inverse_softplus(init_alpha),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                optimizer_parameters.append(self.theta_alpha)
                enable_decay = bool(_require_physics_config_value(physics_cfg, "enable_decay"))
                if enable_decay:
                    init_alpha_decay = float(
                        _require_physics_config_value(physics_cfg, "init_alpha_decay")
                    )
                    self.theta_alpha_decay = nn.Parameter(
                        torch.tensor(
                            inverse_softplus(init_alpha_decay),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                    optimizer_parameters.append(self.theta_alpha_decay)
            else:
                raise ValueError(
                    f"Unsupported physics loss mode: {self.physics_loss_mode}"
                )
        self.optimizer = AdamW(
            optimizer_parameters,
            lr=float(config["train"]["learning_rate"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
        self.scheduler = build_scheduler(config, self.optimizer)
        self.history = HistoryLogger()
        self.best_state: dict[str, Any] | None = None
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
        target_transform: str,
    ) -> TrainArtifacts:
        """Train the model, save checkpoints, and evaluate the best checkpoint."""
        max_epochs = int(self.config["train"]["max_epochs"])
        if self.config.get("debug", {}).get("enabled") and self.config["debug"].get("max_epochs") is not None:
            max_epochs = int(self.config["debug"]["max_epochs"])

        start_time = time.time()
        latest_ckpt_path = self.run_dir / "latest.ckpt"
        best_ckpt_path = self.run_dir / "best.ckpt"
        self.logger.info(
            "Training on %s | train_batches=%d | val_batches=%d | test_batches=%d",
            self.device,
            len(train_loader),
            len(val_loader),
            0 if test_loader is None else len(test_loader),
        )

        for epoch in range(1, max_epochs + 1):
            self.logger.info("Epoch %d/%d started.", epoch, max_epochs)
            train_loss = self._run_epoch(train_loader, train=True, epoch=epoch, max_epochs=max_epochs)
            val_predictions, val_metrics = predict_loader(
                model=self.model,
                loader=val_loader,
                device=self.device,
                target_transform=target_transform,
            )
            val_loss = float(np.mean(np.square(val_predictions["y_pred"] - val_predictions["y_true"])))
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.log_epoch(row)
            self.logger.info(
                "Epoch %d complete | train_loss=%.8f | val_loss=%.8f | val_rmse=%.8f | val_mae=%.8f | val_r2=%.8f",
                epoch,
                train_loss,
                val_loss,
                val_metrics["rmse"],
                val_metrics["mae"],
                val_metrics["r2"],
            )

            checkpoint_state = self._build_checkpoint_state(epoch)

            current_metric = val_metrics["rmse"]
            min_delta = float(self.config["train"]["early_stopping"]["min_delta"])
            if current_metric < (self.best_metric - min_delta):
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.bad_epochs = 0
                checkpoint_state["best_metric"] = self.best_metric
                self.best_state = copy.deepcopy(checkpoint_state)
                save_checkpoint(best_ckpt_path, self.best_state)
                save_dataframe(val_predictions, self.run_dir / "val_predictions.csv")
            else:
                self.bad_epochs += 1

            save_checkpoint(latest_ckpt_path, checkpoint_state)

            if self.scheduler is not None:
                if self.config["train"]["scheduler"]["name"] == "reduce_on_plateau":
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

            patience = int(self.config["train"]["early_stopping"]["patience"])
            if self.bad_epochs >= patience:
                self.logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        if self.best_state is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        best_checkpoint = load_checkpoint(best_ckpt_path, map_location=self.device)
        self._restore_checkpoint_state(best_checkpoint)

        val_predictions, val_metrics = predict_loader(
            model=self.model,
            loader=val_loader,
            device=self.device,
            target_transform=target_transform,
        )
        val_loss = float(np.mean(np.square(val_predictions["y_pred"] - val_predictions["y_true"])))
        save_dataframe(val_predictions, self.run_dir / "val_predictions.csv")

        if test_loader is not None:
            test_predictions, test_metrics = predict_loader(
                model=self.model,
                loader=test_loader,
                device=self.device,
                target_transform=target_transform,
            )
            save_dataframe(test_predictions, self.run_dir / "test_predictions.csv")
            test_loss = float(np.mean(np.square(test_predictions["y_pred"] - test_predictions["y_true"])))
        else:
            test_predictions = pd.DataFrame(columns=["lake_name", "sample_datetime", "y_true", "y_pred"])
            test_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "negative_count": np.nan}
            test_loss = np.nan

        metrics_rows = [
            {"split": "val", "loss": val_loss, **val_metrics},
            {"split": "test", "loss": test_loss, **test_metrics},
        ]
        metrics_path = self.run_dir / "metrics.csv"
        history_path = self.run_dir / "epoch_summary.csv"
        per_lake_metrics_path = self.run_dir / "per_lake_metrics.csv"
        run_summary_path = self.run_dir / "run_summary.json"

        self.history.save(history_path)
        save_dataframe(pd.DataFrame(metrics_rows), metrics_path)

        if not test_predictions.empty:
            from lakeice_ncde.evaluation.per_lake_summary import compute_per_lake_metrics

            per_lake_metrics = compute_per_lake_metrics(test_predictions)
            save_dataframe(per_lake_metrics, per_lake_metrics_path)
        else:
            save_dataframe(pd.DataFrame(), per_lake_metrics_path)

        duration_seconds = time.time() - start_time
        self.logger.info(
            "Final evaluation | val_loss=%.8f | test_loss=%.8f | best_val_rmse=%.8f | best_epoch=%d",
            val_loss,
            test_loss,
            self.best_metric,
            self.best_epoch,
        )
        save_json(
            {
                "best_epoch": self.best_epoch,
                "best_val_rmse": self.best_metric,
                "final_val_loss": val_loss,
                "final_test_loss": test_loss,
                "duration_seconds": duration_seconds,
                "device": str(self.device),
                "train_loss_name": self.config["train"]["loss"],
                "physics_loss_enabled": bool(self.config["train"].get("physics_loss", {}).get("enabled", False)),
                "physics_loss_mode": self.physics_loss_mode,
                "physics_loss_rule": self.config["train"].get("physics_loss", {}).get("rule"),
                "physics_lambda_st": self.config["train"].get("physics_loss", {}).get("lambda_st"),
                "physics_lambda_nn": self.config["train"].get("physics_loss", {}).get("lambda_nn"),
                "physics_kappa": None if self.theta_kappa is None else float(torch.nn.functional.softplus(self.theta_kappa).item()),
                "physics_alpha": None if self.theta_alpha is None else float(torch.nn.functional.softplus(self.theta_alpha).item()),
                "physics_alpha_decay": (
                    1.0
                    if self.physics_loss_mode == "tc2020_curve" and self.theta_alpha_decay is None
                    else None if self.theta_alpha_decay is None else float(torch.nn.functional.softplus(self.theta_alpha_decay).item())
                ),
                "interpolation": self.config["coeffs"]["interpolation"],
                "window_days": self.config["window"]["window_days"],
                "target_transform": target_transform,
            },
            run_summary_path,
        )

        return TrainArtifacts(
            run_dir=self.run_dir,
            metrics_path=metrics_path,
            history_path=history_path,
            best_ckpt_path=best_ckpt_path,
            latest_ckpt_path=latest_ckpt_path,
            val_predictions_path=self.run_dir / "val_predictions.csv",
            test_predictions_path=self.run_dir / "test_predictions.csv",
            per_lake_metrics_path=per_lake_metrics_path,
            run_summary_path=run_summary_path,
        )

    def _build_checkpoint_state(self, epoch: int) -> dict[str, Any]:
        """Build a checkpoint payload including physics-only trainable state."""
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_metric": self.best_metric,
            "theta_kappa_state": None if self.theta_kappa is None else self.theta_kappa.detach().cpu().clone(),
            "theta_alpha_state": None if self.theta_alpha is None else self.theta_alpha.detach().cpu().clone(),
            "theta_alpha_decay_state": None if self.theta_alpha_decay is None else self.theta_alpha_decay.detach().cpu().clone(),
        }

    def _restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore the model and any extra trainable physics parameters from a checkpoint."""
        self.model.load_state_dict(checkpoint["model_state_dict"])
        theta_kappa_state = checkpoint.get("theta_kappa_state")
        with torch.no_grad():
            if self.theta_kappa is not None and theta_kappa_state is not None:
                self.theta_kappa.copy_(theta_kappa_state.to(self.device))
            theta_alpha_state = checkpoint.get("theta_alpha_state")
            if self.theta_alpha is not None and theta_alpha_state is not None:
                self.theta_alpha.copy_(theta_alpha_state.to(self.device))
            theta_alpha_decay_state = checkpoint.get("theta_alpha_decay_state")
            if self.theta_alpha_decay is not None and theta_alpha_decay_state is not None:
                self.theta_alpha_decay.copy_(theta_alpha_decay_state.to(self.device))

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int, max_epochs: int) -> float:
        self.model.train(mode=train)
        losses: list[float] = []
        total_batches = len(loader)
        phase = "train" if train else "eval"
        for batch_index, batch in enumerate(loader, start=1):
            if len(batch.coeff_groups) == 0:
                raise ValueError("Encountered a batch with no windows.")
            pred_tensor = self._predict_batch(batch)
            targets = batch.targets.to(self.device)
            base_loss = self.criterion(pred_tensor, targets)
            physics_breakdown = compute_physics_loss(
                predictions_transformed=pred_tensor,
                physics_context=batch.physics_context,
                config=self.config,
                target_transform=self.config["features"]["target_transform"],
                theta_kappa=self.theta_kappa,
                theta_alpha=self.theta_alpha,
                theta_alpha_decay=self.theta_alpha_decay,
            )
            loss = base_loss + physics_breakdown.total
            check_loss_is_finite(loss)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_norm = self.config["train"]["gradient_clip_norm"]
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip_norm))
                self.optimizer.step()

            batch_loss = float(loss.item())
            losses.append(batch_loss)
            running_loss = float(np.mean(losses))
            if self.physics_loss_enabled:
                if self.physics_loss_mode == "legacy_stefan":
                    self.logger.info(
                        "Epoch %d/%d | %s batch %d/%d | batch_loss=%.8f | base_loss=%.8f | loss_stefan=%.8f | loss_nonneg=%.8f | physics_total=%.8f | kappa=%.8f | running_loss=%.8f",
                        epoch,
                        max_epochs,
                        phase,
                        batch_index,
                        total_batches,
                        batch_loss,
                        float(base_loss.item()),
                        float(physics_breakdown.stefan.item()),
                        float(physics_breakdown.nonneg.item()),
                        float(physics_breakdown.total.item()),
                        float(physics_breakdown.kappa.item()),
                        running_loss,
                    )
                elif self.physics_loss_mode == "tc2020_curve":
                    self.logger.info(
                        "Epoch %d/%d | %s batch %d/%d | batch_loss=%.8f | base_loss=%.8f | loss_curve_grow=%.8f | loss_curve_decay=%.8f | loss_nonneg=%.8f | physics_total=%.8f | alpha=%.8f | alpha_decay=%.8f | running_loss=%.8f",
                        epoch,
                        max_epochs,
                        phase,
                        batch_index,
                        total_batches,
                        batch_loss,
                        float(base_loss.item()),
                        float(physics_breakdown.curve_grow.item()),
                        float(physics_breakdown.curve_decay.item()),
                        float(physics_breakdown.nonneg.item()),
                        float(physics_breakdown.total.item()),
                        float(physics_breakdown.alpha.item()),
                        float(physics_breakdown.alpha_decay.item()),
                        running_loss,
                    )
                else:
                    raise ValueError(
                        f"Unsupported physics loss mode: {self.physics_loss_mode}"
                    )
            else:
                self.logger.info(
                    "Epoch %d/%d | %s batch %d/%d | batch_loss=%.8f | running_loss=%.8f",
                    epoch,
                    max_epochs,
                    phase,
                    batch_index,
                    total_batches,
                    batch_loss,
                    running_loss,
                )
        return float(np.mean(losses))

    def _predict_batch(self, batch: Batch) -> torch.Tensor:
        """Run one forward pass per same-shape coefficient group and restore the original sample order."""
        predictions = torch.empty(len(batch.targets), device=self.device, dtype=batch.targets.dtype)
        for coeff_group in batch.coeff_groups:
            coeff = self._move_coeff_to_device(coeff_group.coeffs)
            group_pred = self.model(coeff)
            if group_pred.ndim == 0:
                group_pred = group_pred.unsqueeze(0)
            if group_pred.ndim != 1:
                raise ValueError(f"Forward pass returned unexpected shape: {tuple(group_pred.shape)}")
            if group_pred.shape[0] != len(coeff_group.indices):
                raise ValueError(
                    f"Forward pass returned {group_pred.shape[0]} predictions for {len(coeff_group.indices)} samples."
                )
            predictions[coeff_group.indices.to(self.device)] = group_pred
        return predictions

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _move_coeff_to_device(self, coeff: Any) -> Any:
        if isinstance(coeff, tuple):
            return tuple(component.to(self.device) for component in coeff)
        return coeff.to(self.device)


def _require_physics_config_value(physics_cfg: dict, field_name: str) -> object:
    if field_name not in physics_cfg:
        raise ValueError(
            f"Physics loss mode '{physics_cfg.get('mode', 'legacy_stefan')}' requires config field '{field_name}'."
        )
    return physics_cfg[field_name]
