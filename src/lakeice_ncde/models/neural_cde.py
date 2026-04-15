from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _require_torchcde():
    try:
        import torchcde  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchcde is required for NeuralCDE training. Install dependencies from requirements.txt in the SCI environment."
        ) from exc
    return torchcde


class CDEFunc(nn.Module):
    """Vector field for the NeuralCDE hidden state."""

    def __init__(
        self,
        hidden_channels: int,
        input_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = hidden_channels
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_hidden_channels))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_hidden_channels
        layers.append(nn.Linear(in_features, hidden_channels * input_channels))
        self.network = nn.Sequential(*layers)
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        del t
        output = self.network(z)
        return output.view(z.shape[0], self.hidden_channels, self.input_channels)


@dataclass
class ModelBuildResult:
    """Convenience dataclass for model construction."""

    model: nn.Module
    input_channels: int


class NeuralCDERegressor(nn.Module):
    """NeuralCDE regressor for irregular lake-ice windows."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
        dropout: float,
        interpolation: str,
        method: str,
        use_adjoint: bool,
        nonnegative_output: bool,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.interpolation = interpolation
        self.method = method
        self.use_adjoint = use_adjoint
        self.nonnegative_output = nonnegative_output

        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(
            hidden_channels=hidden_channels,
            input_channels=input_channels,
            hidden_hidden_channels=hidden_hidden_channels,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        self.readout = nn.Linear(hidden_channels, 1)
        self.softplus = nn.Softplus()

    def _build_interpolation(self, coeffs: Any):
        torchcde = _require_torchcde()
        if isinstance(coeffs, tuple):
            first_component = coeffs[0]
            if first_component.ndim < 3:
                coeffs = tuple(component.unsqueeze(0) for component in coeffs)
        else:
            if coeffs.ndim < 3:
                coeffs = coeffs.unsqueeze(0)

        if self.interpolation == "hermite":
            return torchcde.CubicSpline(coeffs)
        if self.interpolation in {"linear", "rectilinear"}:
            return torchcde.LinearInterpolation(coeffs)
        raise ValueError(f"Unsupported interpolation: {self.interpolation}")

    def forward(self, coeffs: Any) -> torch.Tensor:
        torchcde = _require_torchcde()
        interpolation = self._build_interpolation(coeffs)
        x0 = interpolation.evaluate(interpolation.interval[0])
        if x0.ndim != 2 or x0.shape[-1] != self.input_channels:
            raise ValueError(f"Unexpected x0 shape: {tuple(x0.shape)}")

        z0 = self.initial(x0)
        options: dict[str, Any] = {}
        grid_points = getattr(interpolation, "grid_points", None)
        if self.method == "rk4" and grid_points is not None and len(grid_points) > 1:
            options["step_size"] = float(torch.diff(grid_points).min().item())
        elif self.method != "rk4" and grid_points is not None and len(grid_points) > 2:
            options["jump_t"] = grid_points[1:-1]

        z_t = torchcde.cdeint(
            X=interpolation,
            z0=z0,
            func=self.func,
            t=interpolation.interval,
            method=self.method,
            options=options,
            adjoint=self.use_adjoint,
        )
        if z_t.ndim != 3:
            raise ValueError(f"Unexpected cdeint output shape: {tuple(z_t.shape)}")
        if z_t.shape[0] == 2:
            z_last = z_t[-1]
        elif z_t.shape[1] == 2:
            z_last = z_t[:, -1, :]
        else:
            raise ValueError(f"Unable to infer last hidden state from shape: {tuple(z_t.shape)}")

        prediction = self.readout(z_last).squeeze(-1)
        if prediction.ndim != 1:
            raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
        if self.nonnegative_output:
            prediction = self.softplus(prediction)
        return prediction


def build_model(config: dict, input_channels: int) -> ModelBuildResult:
    """Construct the NeuralCDE model."""
    model_cfg = config["model"]
    model = NeuralCDERegressor(
        input_channels=input_channels,
        hidden_channels=int(model_cfg["hidden_channels"]),
        hidden_hidden_channels=int(model_cfg["hidden_hidden_channels"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        dropout=float(model_cfg["dropout"]),
        interpolation=config["coeffs"]["interpolation"],
        method=model_cfg["method"],
        use_adjoint=bool(model_cfg["use_adjoint"]),
        nonnegative_output=bool(model_cfg["nonnegative_output"]),
    )
    return ModelBuildResult(model=model, input_channels=input_channels)
