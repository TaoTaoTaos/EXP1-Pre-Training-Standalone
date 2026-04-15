# Xiaoxingkai Transfer Training

This repository is a standalone Neural CDE training project for one task: train on source lakes and evaluate transfer to `【Li】Lake Xiaoxingkai`.

## Main Entry

There is one supported entry point:

```powershell
python -m lakeice_ncde run --config configs/experiments/xiaoxingkai_transfer.yaml
```

After editable install, the equivalent command is:

```powershell
lakeice-ncde run --config configs/experiments/xiaoxingkai_transfer.yaml
```

## Available Configs

- `configs/experiments/xiaoxingkai_transfer.yaml`
  Uses the normalized `less60` dataset and tests on Xiaoxingkai.
- `configs/experiments/xiaoxingkai_transfer_daily12.yaml`
  Same workflow, but the Xiaoxingkai test rows are restricted to 12:00 observations.

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
