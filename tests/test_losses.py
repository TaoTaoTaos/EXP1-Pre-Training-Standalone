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
    torch.testing.assert_close(breakdown.kappa, torch.tensor(0.0, dtype=torch.float32))


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
