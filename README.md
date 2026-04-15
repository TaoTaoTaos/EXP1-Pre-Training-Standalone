# Xiaoxingkai Transfer Training

This repository is a standalone Neural CDE training project for one task: train on source lakes and evaluate transfer to `【Li】Lake Xiaoxingkai`.

## Main Entry

There is one supported entry point:

```powershell
python -m lakeice_ncde run --config configs/experiments/EXP1_history_autoreg.yaml
```

After editable install, the equivalent command is:

```powershell
lakeice-ncde run --config configs/experiments/EXP1_history_autoreg.yaml
```

## Available Configs

- `configs/experiments/EXP1_history_autoreg.yaml`
  Recommended EXP1 config. Adds lagged ice-history features and uses autoregressive seasonal rollout.
- `configs/experiments/EXP2_history_autoreg_stefan.yaml`
  Recommended EXP2 config. Builds on EXP1 and adds a Stefan-style physics loss.
- `configs/experiments/xiaoxingkai_transfer.yaml`
  Legacy baseline. Uses the normalized `less60` dataset and tests on Xiaoxingkai.
- `configs/experiments/xiaoxingkai_transfer_daily12.yaml`
  Legacy baseline with Xiaoxingkai test rows restricted to 12:00 observations.
- `configs/experiments/xiaoxingkai_transfer_daily12_history_autoreg.yaml`
  Legacy long-name version of EXP1.
- `configs/experiments/xiaoxingkai_transfer_daily12_history_autoreg_physics.yaml`
  Legacy long-name version of EXP2.

## Recommended Workflow

Use the short experiment names for the current two-stage workflow:

1. `EXP1_history_autoreg`
   History-feature baseline with autoregressive rollout.
2. `EXP2_history_autoreg_stefan`
   EXP1 plus Stefan-style physics regularization.

## Workflow

The single CLI command runs the full pipeline:

1. Load and standardize the Excel data.
2. Split source lakes into temporal `train` and `val`.
3. Keep Xiaoxingkai fully isolated as `test`.
4. Build irregular windows.
5. Precompute CDE interpolation coefficients.
6. Train the Neural CDE model.
7. Export metrics, predictions, plots, and the PDF report.

## Install

```powershell
conda activate SCI
cd S:\STU-Papers\My_Papers\EXP1-Pre-Training-Standalone
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Output

Run artifacts are written under `outputs/runs/<experiment_name>/...`.
