from __future__ import annotations

import pytest
import torch

from lakeice_ncde.training.losses import (
    compute_physics_loss,
    compute_tc2020_curve_thickness,
    inverse_softplus,
)


def _tc2020_config(*, enable_decay: bool) -> dict:
    return {
        "train": {
            "physics_loss": {
                "enabled": True,
                "mode": "tc2020_curve",
                "lambda_curve_grow": 1.0,
                "lambda_curve_decay": 0.2,
                "lambda_nn": 1.0,
                "enable_decay": enable_decay,
            }
        }
    }


def test_compute_tc2020_curve_thickness_defaults_alpha_decay_to_one() -> None:
    afdd = torch.tensor([0.0, 4.0], dtype=torch.float32)
    atdd = torch.tensor([1.0, 2.0], dtype=torch.float32)
    theta_alpha = torch.tensor(inverse_softplus(1.5), dtype=torch.float32)

    h_curve, alpha, alpha_decay = compute_tc2020_curve_thickness(
        afdd=afdd,
        atdd=atdd,
        theta_alpha=theta_alpha,
        theta_alpha_decay=None,
        enable_decay=False,
    )

    expected_alpha = torch.tensor(1.5, dtype=torch.float32)
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(alpha_decay, torch.tensor(1.0, dtype=torch.float32))
    torch.testing.assert_close(
        h_curve,
        expected_alpha * torch.sqrt(torch.clamp(afdd, min=0.0)),
    )


def test_compute_physics_loss_tc2020_curve_respects_masks() -> None:
    predictions = torch.tensor([2.2, 0.7, -0.3], dtype=torch.float32)
    physics_context = {
        "afdd": torch.tensor([4.0, 9.0, 16.0], dtype=torch.float32),
        "atdd": torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32),
        "is_growth_phase": torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        "is_decay_phase": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        "stable_ice_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
    }
    theta_alpha = torch.tensor(inverse_softplus(1.0), dtype=torch.float32)
    theta_alpha_decay = torch.tensor(inverse_softplus(0.5), dtype=torch.float32)

    breakdown = compute_physics_loss(
        predictions_transformed=predictions,
        physics_context=physics_context,
        config=_tc2020_config(enable_decay=True),
        target_transform="none",
        theta_kappa=None,
        theta_alpha=theta_alpha,
        theta_alpha_decay=theta_alpha_decay,
    )

    expected_curve = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    expected_curve_grow = (predictions[0] - expected_curve[0]).pow(2)
    expected_curve_decay = (predictions[1] - expected_curve[1]).pow(2)
    expected_nonneg = torch.relu(-predictions).square().mean()
    expected_total = (
        expected_curve_grow
        + 0.2 * expected_curve_decay
        + expected_nonneg
    )

    torch.testing.assert_close(breakdown.curve_grow, expected_curve_grow)
    torch.testing.assert_close(breakdown.curve_decay, expected_curve_decay)
    torch.testing.assert_close(breakdown.nonneg, expected_nonneg)
    torch.testing.assert_close(breakdown.total, expected_total)
    torch.testing.assert_close(breakdown.alpha, torch.tensor(1.0, dtype=torch.float32))
    torch.testing.assert_close(
        breakdown.alpha_decay,
        torch.tensor(0.5, dtype=torch.float32),
    )
    torch.testing.assert_close(breakdown.stefan, torch.tensor(0.0, dtype=torch.float32))
    torch.testing.assert_close(
        breakdown.stefan_grow,
        torch.tensor(0.0, dtype=torch.float32),
    )
    torch.testing.assert_close(breakdown.kappa, torch.tensor(0.0, dtype=torch.float32))


def test_compute_physics_loss_tc2020_curve_plus_adds_stefan_growth_term() -> None:
    predictions = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    physics_context = {
        "afdd": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        "atdd": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        "is_growth_phase": torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
        "is_decay_phase": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        "stable_ice_mask": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        "ice_prev_m": torch.tensor([0.4, 0.02, 0.4, 0.4], dtype=torch.float32),
        "ice_prev_gap_days": torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float32),
        "Air_Temperature_celsius": torch.tensor([-1.0, -1.0, 0.0, -1.0], dtype=torch.float32),
        "ice_prev_available": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    }
    config = _tc2020_config(enable_decay=False)
    physics_cfg = config["train"]["physics_loss"]
    physics_cfg.update(
        {
            "enable_stefan_grow": True,
            "lambda_curve_grow": 0.0,
            "lambda_curve_decay": 0.0,
            "lambda_st": 0.5,
            "lambda_nn": 0.0,
            "min_prev_ice_m": 0.05,
            "grow_temp_threshold_celsius": -0.5,
        }
    )
    theta_alpha = torch.tensor(inverse_softplus(1.0), dtype=torch.float32)
    theta_kappa = torch.tensor(inverse_softplus(0.1), dtype=torch.float32)

    breakdown = compute_physics_loss(
        predictions_transformed=predictions,
        physics_context=physics_context,
        config=config,
        target_transform="none",
        theta_kappa=theta_kappa,
        theta_alpha=theta_alpha,
    )

    # 只有第 0 个样本同时满足 grow_mask、prev 可用、prev_ice > 0.05 且气温 < -0.5。
    expected_residual = (predictions[0].square() - physics_context["ice_prev_m"][0].square()) - 0.1 * 2.0
    expected_stefan = expected_residual.square()
    torch.testing.assert_close(breakdown.stefan_grow, expected_stefan)
    torch.testing.assert_close(breakdown.stefan, expected_stefan)
    torch.testing.assert_close(breakdown.kappa, torch.tensor(0.1, dtype=torch.float32))
    torch.testing.assert_close(breakdown.total, 0.5 * expected_stefan)


def test_compute_physics_loss_tc2020_curve_plus_requires_theta_kappa() -> None:
    config = _tc2020_config(enable_decay=False)
    config["train"]["physics_loss"]["enable_stefan_grow"] = True

    with pytest.raises(ValueError, match="theta_kappa"):
        compute_physics_loss(
            predictions_transformed=torch.tensor([0.2], dtype=torch.float32),
            physics_context={
                "afdd": torch.tensor([1.0], dtype=torch.float32),
                "atdd": torch.tensor([0.0], dtype=torch.float32),
                "is_growth_phase": torch.tensor([1.0], dtype=torch.float32),
                "is_decay_phase": torch.tensor([0.0], dtype=torch.float32),
                "stable_ice_mask": torch.tensor([1.0], dtype=torch.float32),
            },
            config=config,
            target_transform="none",
            theta_kappa=None,
            theta_alpha=torch.tensor(inverse_softplus(1.0), dtype=torch.float32),
        )


def test_compute_physics_loss_tc2020_curve_requires_stable_mask() -> None:
    with pytest.raises(ValueError, match="stable_ice_mask"):
        compute_physics_loss(
            predictions_transformed=torch.tensor([0.2], dtype=torch.float32),
            physics_context={
                "afdd": torch.tensor([1.0], dtype=torch.float32),
                "atdd": torch.tensor([0.0], dtype=torch.float32),
                "is_growth_phase": torch.tensor([1.0], dtype=torch.float32),
                "is_decay_phase": torch.tensor([0.0], dtype=torch.float32),
            },
            config=_tc2020_config(enable_decay=False),
            target_transform="none",
            theta_kappa=None,
            theta_alpha=torch.tensor(inverse_softplus(1.0), dtype=torch.float32),
        )
