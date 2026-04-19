# Xiaoxingkai Transfer Training

This repository is a standalone Neural CDE training project for one task: train on source lakes and evaluate transfer to `【Li】Lake Xiaoxingkai`.

## Main Entry

There is one supported entry point:

```powershell
python -m lakeice_ncde run --config configs/experiments/EXP0_pretrain_autoreg.yaml
```

After editable install, the equivalent command is:

```powershell
lakeice-ncde run --config configs/experiments/EXP0_pretrain_autoreg.yaml
```

To launch the current three experiments in parallel and collect their PDFs plus one Excel summary:

```powershell
python -m lakeice_ncde run --config configs/experiments/Run-ALL.yaml
```

To launch or resume the parameter-search workflow that wraps `Run-ALL.yaml`:

```powershell
python -m lakeice_ncde search --config configs/search/参数搜索.yaml
```

## Available Configs

- `configs/experiments/EXP0_pretrain_autoreg.yaml`
  Recommended EXP0 config. Autoregressive pretraining on source lakes only; Xiaoxingkai is fully held out as test.
- `configs/experiments/EXP1_transfer_autoreg.yaml`
  Recommended EXP1 config. Transfer experiment: starts from EXP0 and allows Xiaoxingkai history before `2026-01-01` to participate in train/val.
- `configs/experiments/EXP2_transfer_autoreg_stefan.yaml`
  Recommended EXP2 config. EXP1-style transfer setup plus Stefan-style physics loss.
- `configs/experiments/Run-ALL.yaml`
  Batch config. Runs EXP0/EXP1/EXP2 in parallel and gathers the three PDFs plus one one-column-per-experiment Excel summary into one run folder.
- `configs/experiments/Olds/EXP0_transfer_autoreg.yaml`
  Compatibility alias for `EXP0_pretrain_autoreg.yaml`.
- `configs/experiments/Olds/EXP1_history_autoreg.yaml`
  Compatibility alias for `EXP1_transfer_autoreg.yaml`.
- `configs/experiments/Olds/EXP2_history_autoreg_stefan.yaml`
  Compatibility alias for `EXP2_transfer_autoreg_stefan.yaml`.
- `configs/experiments/Olds/xiaoxingkai_transfer.yaml`
  Legacy baseline. Uses the normalized `less60` dataset and tests on Xiaoxingkai.
- `configs/experiments/Olds/xiaoxingkai_transfer_daily12.yaml`
  Legacy baseline with Xiaoxingkai test rows restricted to 12:00 observations.
- `configs/experiments/Olds/xiaoxingkai_transfer_daily12_history_autoreg.yaml`
  Legacy long-name version of EXP1.
- `configs/experiments/Olds/xiaoxingkai_transfer_daily12_history_autoreg_physics.yaml`
  Legacy long-name version of EXP2.

## Recommended Workflow

Use the short experiment names for the current three-stage workflow:

1. `EXP0_pretrain_autoreg`
   Source-lake autoregressive pretraining. Xiaoxingkai stays fully isolated as test.
1. `EXP1_transfer_autoreg`
   Transfer experiment. Xiaoxingkai history before `2026-01-01` enters train/val.
1. `EXP2_transfer_autoreg_stefan`
   EXP1 transfer setup plus Stefan-style physics regularization.

## Workflow

The single CLI command runs the full pipeline:

1. Load and standardize the Excel data.
2. Split source lakes into temporal `train` and `val`.
3. Depending on the experiment, either keep Xiaoxingkai fully isolated as `test` (EXP0) or split its pre-cutoff history into `train`/`val` and reserve the later segment as `test` (EXP1/EXP2).
4. Build irregular windows.
5. Precompute CDE interpolation coefficients.
6. Train the Neural CDE model.
7. Export metrics, predictions, and the PDF report.

`Run-ALL.yaml` reuses the same `run` command, but treats its child configs as a batch:

1. Resolve the child experiment config list under `batch.experiments`.
2. Run those experiments in parallel.
3. Copy each child PDF into `outputs/runs/Run-ALL/[NN]_Run-ALL_.../`.
4. Write one Excel workbook where each experiment is a column and each row is one parameter or metric record.

## Install

```powershell
conda activate SCI
cd S:\STU-Papers\My_Papers\Neural-CDE-based-IceTransfer
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Output

Single-experiment artifacts are written under `outputs/runs/<experiment_name>/...`.

Batch artifacts are written under `outputs/runs/Run-ALL/[NN]_Run-ALL_.../`, including:

- the copied child PDFs
- one batch-level comparison PDF
- one `_summary.xlsx` workbook
- the merged batch config and batch manifest under `artifacts/`

Search artifacts are written under the configured `search.output_root`, including:

- `trials_master.csv` with one row per trial
- `trial_parameters.csv` with one row per trial x parameter
- `trial_experiments.csv` with one row per trial x experiment
- `top_trials.csv` sorted by the objective score
- `study_summary.json` plus `artifacts/study.journal` for resume
- one `trials/trial_000N/` folder per trial with resolved config, overrides, metadata, and the full `Run-ALL` outputs
