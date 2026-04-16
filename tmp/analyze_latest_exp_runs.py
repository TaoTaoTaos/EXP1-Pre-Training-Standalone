from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"S:\STU-Papers\My_Papers\EXP1-Pre-Training-Standalone")
RUNS = {
    "EXP1": PROJECT_ROOT / "outputs" / "runs" / "EXP1_history_autoreg" / "EXP1_history_autoreg_20260416_011753",
    "EXP2": PROJECT_ROOT
    / "outputs"
    / "runs"
    / "EXP2_history_autoreg_stefan"
    / "EXP2_history_autoreg_stefan_20260416_010825",
}
FORCING_CSV = PROJECT_ROOT / "data" / "raw" / "XXKH-St-UTC+8-ERA5-1980-202603-noNA.csv"


def load_run(name: str, run_dir: Path) -> dict:
    df = pd.read_csv(run_dir / "seasonal_rollout_overlap_predictions.csv")
    df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
    df = df.sort_values("sample_datetime").reset_index(drop=True)
    df["residual"] = df["y_pred"] - df["y_true"]
    df["pred_diff"] = df["y_pred"].diff()
    df["true_diff"] = df["y_true"].diff()
    df["abs_residual"] = df["residual"].abs()

    with (run_dir / "run_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return {"name": name, "run_dir": run_dir, "df": df, "summary": summary}


def summarize_run(run: dict) -> dict:
    df = run["df"]
    peak_idx = int(df["y_pred"].idxmax())
    peak = df.loc[peak_idx]
    worst_high_idx = int(df["residual"].idxmax())
    worst_high = df.loc[worst_high_idx]
    largest_drop_idx = int(df["pred_diff"].idxmin())
    largest_drop = df.loc[largest_drop_idx]
    last = df.iloc[-1]

    return {
        "test_rmse": run["summary"]["final_test_loss"] ** 0.5,
        "test_r2": run["summary"].get("physics_loss_rule", None),
        "peak": {
            "date": peak["sample_datetime"],
            "y_pred": float(peak["y_pred"]),
            "y_true": float(peak["y_true"]),
            "residual": float(peak["residual"]),
            "pred_diff": None if pd.isna(peak["pred_diff"]) else float(peak["pred_diff"]),
            "true_diff": None if pd.isna(peak["true_diff"]) else float(peak["true_diff"]),
        },
        "worst_high": {
            "date": worst_high["sample_datetime"],
            "y_pred": float(worst_high["y_pred"]),
            "y_true": float(worst_high["y_true"]),
            "residual": float(worst_high["residual"]),
        },
        "largest_drop": {
            "date": largest_drop["sample_datetime"],
            "y_pred": float(largest_drop["y_pred"]),
            "y_true": float(largest_drop["y_true"]),
            "pred_diff": float(largest_drop["pred_diff"]),
            "true_diff": float(largest_drop["true_diff"]),
            "residual": float(largest_drop["residual"]),
        },
        "last": {
            "date": last["sample_datetime"],
            "y_pred": float(last["y_pred"]),
            "y_true": float(last["y_true"]),
            "residual": float(last["residual"]),
        },
        "mean_bias": float(df["residual"].mean()),
        "mean_abs_error": float(df["abs_residual"].mean()),
    }


def load_forcing() -> pd.DataFrame:
    forcing = pd.read_csv(FORCING_CSV)
    datetime_column = "sample_datetime"
    if datetime_column not in forcing.columns:
        for candidate in ("datetime", "era5_datetime", "time", "date"):
            if candidate in forcing.columns:
                datetime_column = candidate
                break
    forcing["sample_datetime"] = pd.to_datetime(forcing[datetime_column])
    return forcing.sort_values("sample_datetime").reset_index(drop=True)


def window_with_forcing(run: dict, center_date: pd.Timestamp, days: int = 3) -> pd.DataFrame:
    forcing = load_forcing()
    df = run["df"]
    start = center_date - pd.Timedelta(days=days)
    end = center_date + pd.Timedelta(days=days)
    window = df.loc[(df["sample_datetime"] >= start) & (df["sample_datetime"] <= end)].copy()
    forcing_window = forcing.loc[(forcing["sample_datetime"] >= start) & (forcing["sample_datetime"] <= end)].copy()
    merged = window.merge(forcing_window, on="sample_datetime", how="left", suffixes=("", "_forcing"))
    keep = [
        "sample_datetime",
        "y_true",
        "y_pred",
        "residual",
        "pred_diff",
        "true_diff",
        "Air_Temperature_celsius",
        "Precipitation_millimeterPerDay",
        "Snowfall_millimeterPerDay",
        "Relative_Humidity_percent",
        "Shortwave_Radiation_Downwelling_wattPerMeterSquared",
        "Longwave_Radiation_Downwelling_wattPerMeterSquared",
    ]
    available = [column for column in keep if column in merged.columns]
    return merged[available]


def compare_runs(run_a: dict, run_b: dict) -> pd.DataFrame:
    a = run_a["df"][["sample_datetime", "y_true", "y_pred", "residual", "pred_diff"]].rename(
        columns={
            "y_pred": f"y_pred_{run_a['name']}",
            "residual": f"residual_{run_a['name']}",
            "pred_diff": f"pred_diff_{run_a['name']}",
        }
    )
    b = run_b["df"][["sample_datetime", "y_pred", "residual", "pred_diff"]].rename(
        columns={
            "y_pred": f"y_pred_{run_b['name']}",
            "residual": f"residual_{run_b['name']}",
            "pred_diff": f"pred_diff_{run_b['name']}",
        }
    )
    merged = a.merge(b, on="sample_datetime", how="inner")
    merged["pred_gap_exp2_minus_exp1"] = merged[f"y_pred_{run_b['name']}"] - merged[f"y_pred_{run_a['name']}"]
    return merged


def main() -> None:
    runs = [load_run(name, run_dir) for name, run_dir in RUNS.items()]
    summaries = {run["name"]: summarize_run(run) for run in runs}

    print("=== Run Summary ===")
    for run in runs:
        name = run["name"]
        summary = run["summary"]
        analysis = summaries[name]
        metrics_df = pd.read_csv(run["run_dir"] / "metrics.csv")
        test_r2 = float(metrics_df.loc[metrics_df["split"] == "test", "r2"].iloc[0])
        print(
            f"{name}: best_epoch={summary['best_epoch']}, "
            f"test_rmse={summary.get('final_test_loss', 0.0) ** 0.5:.6f}, "
            f"test_r2={test_r2:.6f}, "
            f"mean_bias={analysis['mean_bias']:.6f}, mae={analysis['mean_abs_error']:.6f}"
        )
        print(
            f"  peak={analysis['peak']['date']} pred={analysis['peak']['y_pred']:.6f} "
            f"true={analysis['peak']['y_true']:.6f} residual={analysis['peak']['residual']:.6f}"
        )
        print(
            f"  worst_high={analysis['worst_high']['date']} pred={analysis['worst_high']['y_pred']:.6f} "
            f"true={analysis['worst_high']['y_true']:.6f} residual={analysis['worst_high']['residual']:.6f}"
        )
        print(
            f"  largest_drop={analysis['largest_drop']['date']} pred_diff={analysis['largest_drop']['pred_diff']:.6f} "
            f"true_diff={analysis['largest_drop']['true_diff']:.6f} residual={analysis['largest_drop']['residual']:.6f}"
        )
        print(
            f"  last={analysis['last']['date']} pred={analysis['last']['y_pred']:.6f} "
            f"true={analysis['last']['y_true']:.6f} residual={analysis['last']['residual']:.6f}"
        )

    comparison = compare_runs(runs[0], runs[1])
    print("\n=== Largest EXP2 - EXP1 Prediction Gains ===")
    gains = comparison.sort_values("pred_gap_exp2_minus_exp1", ascending=False).head(10)
    print(gains[["sample_datetime", "y_true", "y_pred_EXP1", "y_pred_EXP2", "pred_gap_exp2_minus_exp1"]].to_string(index=False))

    print("\n=== Most Negative EXP2 Residuals ===")
    exp2_df = runs[1]["df"].sort_values("residual").head(10)
    print(exp2_df[["sample_datetime", "y_true", "y_pred", "residual", "pred_diff", "true_diff"]].to_string(index=False))

    peak_date = pd.Timestamp(summaries["EXP2"]["peak"]["date"])
    drop_date = pd.Timestamp(summaries["EXP2"]["largest_drop"]["date"])
    print("\n=== EXP2 Peak Window With Forcing ===")
    print(window_with_forcing(runs[1], peak_date, days=4).to_string(index=False))
    print("\n=== EXP2 Largest Drop Window With Forcing ===")
    print(window_with_forcing(runs[1], drop_date, days=4).to_string(index=False))


if __name__ == "__main__":
    main()
