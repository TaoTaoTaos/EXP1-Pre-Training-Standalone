from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def _finalize_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_loss_curves_figure(history: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_title("Training And Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_loss_curves(history: pd.DataFrame, path: Path) -> None:
    fig = create_loss_curves_figure(history)
    _finalize_figure(fig, path)


def create_metric_curves_figure(history: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_map = [("val_rmse", "RMSE"), ("val_mae", "MAE"), ("val_r2", "R2")]
    for ax, (column, label) in zip(axes, metric_map):
        ax.plot(history["epoch"], history[column], linewidth=2)
        ax.set_title(f"Validation {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    return fig


def plot_metric_curves(history: pd.DataFrame, path: Path) -> None:
    fig = create_metric_curves_figure(history)
    _finalize_figure(fig, path)


def create_pred_vs_obs_figure(predictions: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predictions["y_true"], predictions["y_pred"], alpha=0.6, s=18)
    min_value = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
    max_value = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    ax.set_title("Predicted Vs Observed Ice Thickness")
    ax.set_xlabel("Observed total_ice_m")
    ax.set_ylabel("Predicted total_ice_m")
    ax.grid(True, alpha=0.3)
    return fig


def plot_pred_vs_obs(predictions: pd.DataFrame, path: Path) -> None:
    fig = create_pred_vs_obs_figure(predictions)
    _finalize_figure(fig, path)


def plot_residual_histogram(predictions: pd.DataFrame, path: Path) -> None:
    residuals = predictions["y_pred"] - predictions["y_true"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Prediction - Observation")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _finalize_figure(fig, path)


def create_lake_timeseries_figure(
    predictions: pd.DataFrame,
    lake_name: str,
    start_datetime: str | pd.Timestamp | None = None,
    end_datetime: str | pd.Timestamp | None = None,
    title: str | None = None,
) -> plt.Figure:
    lake_df = predictions.loc[predictions["lake_name"] == lake_name].sort_values("sample_datetime")
    if start_datetime is not None:
        lake_df = lake_df.loc[lake_df["sample_datetime"] >= pd.Timestamp(start_datetime)]
    if end_datetime is not None:
        lake_df = lake_df.loc[lake_df["sample_datetime"] <= pd.Timestamp(end_datetime)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lake_df["sample_datetime"], lake_df["y_true"], label="Observed", linewidth=2, color="#1f77b4")
    ax.scatter(
        lake_df["sample_datetime"],
        lake_df["y_pred"],
        label="Predicted",
        s=24,
        alpha=0.85,
        color="#d62728",
        edgecolors="white",
        linewidths=0.3,
    )
    ax.set_title(title or f"Lake Time Series | {_display_lake_name(lake_name)}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Ice thickness (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_per_lake_timeseries(predictions: pd.DataFrame, output_dir: Path) -> None:
    for lake_name, lake_df in predictions.groupby("lake_name"):
        fig = create_lake_timeseries_figure(predictions, lake_name)
        safe_name = _slugify(lake_name)
        _finalize_figure(fig, output_dir / f"05_lake_timeseries_{safe_name}.png")


def plot_per_lake_metrics(per_lake_metrics: pd.DataFrame, path: Path) -> None:
    if per_lake_metrics.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metric_columns = [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2")]
    x = np.arange(len(per_lake_metrics))
    for ax, (column, title) in zip(axes, metric_columns):
        ax.bar(x, per_lake_metrics[column], alpha=0.85)
        ax.set_title(f"Per-Lake {title}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(per_lake_metrics["lake_name"], rotation=45, ha="right")
    _finalize_figure(fig, path)


def plot_input_paths(window_bundle: dict, path: Path, sample_count: int) -> None:
    windows = window_bundle["windows"][:sample_count]
    input_channels = window_bundle["input_channels"]
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 3 * max(1, len(windows))), sharex=False)
    if len(windows) == 1:
        axes = [axes]
    for ax, window in zip(axes, windows):
        time_values = window[:, 0].numpy()
        for channel_index, channel_name in enumerate(input_channels[1:], start=1):
            ax.plot(time_values, window[:, channel_index].numpy(), label=channel_name, alpha=0.8)
        ax.set_title("Input Path Example")
        ax.set_xlabel("Relative Time")
        ax.set_ylabel("Scaled Value")
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)
    _finalize_figure(fig, path)


def plot_interpolation_debug(window_bundle: dict, coeff_bundle: dict, path: Path, sample_count: int, points: int) -> None:
    try:
        import torchcde  # type: ignore
    except ModuleNotFoundError:
        return

    fig, axes = plt.subplots(sample_count, 1, figsize=(12, 3 * max(1, sample_count)))
    if sample_count == 1:
        axes = [axes]

    for index in range(sample_count):
        window = window_bundle["windows"][index]
        coeff = coeff_bundle["coeffs"][index]
        if coeff_bundle["interpolation"] == "hermite":
            interp = torchcde.CubicSpline(coeff.unsqueeze(0))
        else:
            interp = torchcde.LinearInterpolation(coeff.unsqueeze(0))
        ts = torch.linspace(float(interp.interval[0]), float(interp.interval[1]), points)
        evaluated = interp.evaluate(ts)
        if evaluated.ndim == 3:
            if evaluated.shape[0] == 1:
                evaluated = evaluated.squeeze(0)
            elif evaluated.shape[1] == 1:
                evaluated = evaluated.squeeze(1)
        evaluated_np = evaluated.detach().cpu().numpy()
        ax = axes[index]
        observed_feature = window[:, 1].numpy() if window.shape[1] > 1 else window[:, 0].numpy()
        interpolated_feature = evaluated_np[:, 1] if evaluated_np.shape[1] > 1 else evaluated_np[:, 0]
        ax.plot(window[:, 0].numpy(), observed_feature, "o", label="Observed first feature")
        ax.plot(ts.detach().cpu().numpy(), interpolated_feature, "-", label="Interpolated first feature")
        ax.set_title(f"Interpolation Debug Example {index}")
        ax.set_xlabel("Relative Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
    _finalize_figure(fig, path)


def create_prediction_distribution_figure(predictions: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(predictions["y_true"], bins=30, alpha=0.6, label="Observed", edgecolor="black")
    ax.hist(predictions["y_pred"], bins=30, alpha=0.6, label="Predicted", edgecolor="black")
    ax.set_title("Prediction Distribution Vs Observation Distribution")
    ax.set_xlabel("Ice thickness (m)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_prediction_distribution(predictions: pd.DataFrame, path: Path) -> None:
    fig = create_prediction_distribution_figure(predictions)
    _finalize_figure(fig, path)


def _slugify(value: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in value)
    return sanitized[:80]


def _display_lake_name(value: str) -> str:
    return str(value).replace("【", "[").replace("】", "] ")
