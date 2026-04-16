from __future__ import annotations

import torch

from lakeice_ncde.workflows.xiaoxingkai_transfer import _subsample_bundle_by_lake


def test_subsample_bundle_keeps_physics_context_aligned(tmp_path) -> None:
    bundle_path = tmp_path / "train_windows.pt"
    bundle = {
        "split_name": "EXP2_transfer_autoreg_stefan",
        "split": "train",
        "windows": [torch.tensor([[float(index)]], dtype=torch.float32) for index in range(4)],
        "metadata": [
            {
                "window_id": f"train_{index:06d}",
                "split": "train",
                "lake_name": "LakeA" if index < 2 else "LakeB",
                "target_datetime": f"2026-01-0{index + 1} 12:00:00",
                "length": 1,
                "window_days": 7,
                "target_raw": float(index),
                "target_transformed": float(index),
            }
            for index in range(4)
        ],
        "targets_raw": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        "targets_transformed": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
        "physics_context": {
            "ice_prev_m": torch.tensor([100.0, 101.0, 102.0, 103.0], dtype=torch.float32),
            "ice_prev_gap_days": torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float32),
            "Air_Temperature_celsius": torch.tensor([-10.0, -11.0, -12.0, -13.0], dtype=torch.float32),
            "ice_prev_available": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        },
    }
    torch.save(bundle, bundle_path)

    _subsample_bundle_by_lake(bundle_path, max_windows_per_lake=1, seed=0)

    result = torch.load(bundle_path, map_location="cpu", weights_only=False)
    kept_targets = result["targets_raw"].tolist()
    kept_prev_ice = result["physics_context"]["ice_prev_m"].tolist()
    kept_gap_days = result["physics_context"]["ice_prev_gap_days"].tolist()
    assert len(kept_targets) == 2
    assert len(kept_prev_ice) == len(kept_targets)
    assert len(kept_gap_days) == len(kept_targets)

    for target_raw, prev_ice, gap_days in zip(kept_targets, kept_prev_ice, kept_gap_days):
        assert prev_ice == 100.0 + target_raw
        assert gap_days == 10.0 + target_raw
