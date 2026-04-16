from __future__ import annotations

from datetime import datetime

from lakeice_ncde.experiment.tracker import create_run_context
from lakeice_ncde.utils.paths import build_pdf_name, build_sequential_run_name, next_run_sequence


def test_next_run_sequence_ignores_non_run_directories(tmp_path) -> None:
    experiment_root = tmp_path / "EXP1"
    (experiment_root / "[01]_EXP1_20260416_010101").mkdir(parents=True)
    (experiment_root / "summary").mkdir()

    assert next_run_sequence(experiment_root) == 2


def test_build_sequential_run_name_uses_zero_padded_prefix(tmp_path) -> None:
    experiment_root = tmp_path / "EXP1"
    (experiment_root / "[01]_EXP1_20260416_010101").mkdir(parents=True)

    run_name = build_sequential_run_name(
        experiment_root,
        "EXP1",
        now=datetime(2026, 4, 16, 12, 34, 56),
    )

    assert run_name == "[02]_EXP1_20260416_123456"


def test_create_run_context_and_pdf_name_share_same_sequence(tmp_path) -> None:
    run_context = create_run_context(tmp_path, "EXP1", {"experiment": {"name": "EXP1"}})

    assert run_context.run_name.startswith("[01]_EXP1_")
    assert run_context.run_dir.name == run_context.run_name
    assert build_pdf_name(run_context.run_name).startswith("01_")
    assert not (run_context.run_dir / "figures").exists()
