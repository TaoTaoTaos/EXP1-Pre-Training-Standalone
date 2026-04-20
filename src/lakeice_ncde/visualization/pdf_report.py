from __future__ import annotations

import html
import json
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from lakeice_ncde.data.load_excel import filter_include_lakes, load_raw_excel, standardize_dataframe
from lakeice_ncde.data.windowing import _build_single_window
from lakeice_ncde.visualization.plots import (
    create_lake_timeseries_figure,
    create_loss_curves_figure,
    create_metric_curves_figure,
    create_pred_vs_obs_figure,
    create_prediction_distribution_figure,
)


XIAOXINGKAI_NAME = "【Li】Lake Xiaoxingkai"

XIAOXINGKAI_NAME = "[Li] Lake Xiaoxingkai"


def build_pdf_report(run_dir: Path, pdf_path: Path) -> None:
    """Build a single self-contained PDF report for one finished run."""
    report_data = _collect_report_data(run_dir)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"{report_data['experiment_name']} report",
    )
    styles = _build_styles()
    story: list[Any] = []

    story.extend(_build_cover_section(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_setup_section(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_figures_section(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_data_selection_section(report_data, styles))

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#17324d"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=12,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCompact",
            parent=styles["BodyText"],
            fontSize=8.6,
            leading=10.2,
            spaceAfter=0,
        )
    )
    return styles


def _build_cover_section(report_data: dict[str, Any], styles) -> list[Any]:
    metrics_df = report_data["metrics"]
    per_lake_df = report_data["per_lake_metrics"]
    summary = report_data["run_summary"]
    diagnostics = report_data["diagnostics"]
    seasonal_rollout_overview = report_data["seasonal_rollout_overview"]

    story: list[Any] = [
        Paragraph(f"{report_data['experiment_name']} Result Report", styles["ReportTitle"]),
        Paragraph(
            f"Run directory: {report_data['run_dir']}<br/>"
            f"Experiment: {report_data['experiment_name']}<br/>"
            f"Source file: {report_data['raw_excel']}<br/>"
            f"Target lake: {_safe_lake_label(report_data['target_lake_label'])}",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Front Summary", styles["SectionHeading"]),
    ]

    overall_table = _styled_table(
        [["Split", "Loss", "RMSE", "MAE", "R2", "Bias", "Negative Preds"]]
        + [
            [
                row["split"],
                f"{row['loss']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mae']:.4f}",
                f"{row['r2']:.4f}",
                f"{row['bias']:.4f}",
                str(int(row["negative_count"])),
            ]
            for _, row in metrics_df.iterrows()
        ],
        col_widths=[1.1, 1.0, 1.0, 1.0, 1.2, 1.0, 1.2],
    )
    story.append(overall_table)
    story.append(Spacer(1, 10))

    if len(per_lake_df) > 1:
        story.append(Paragraph("Per-Lake Evaluation Summary", styles["SectionHeading"]))
        lake_rows = [["Lake", "Count", "RMSE", "MAE", "R2", "Bias"]]
        for _, row in per_lake_df.iterrows():
            lake_rows.append(
                [
                    _safe_lake_label(row["lake_name"]),
                    str(int(row["count"])),
                    f"{row['rmse']:.4f}",
                    f"{row['mae']:.4f}",
                    f"{row['r2']:.4f}",
                    f"{row['bias']:.4f}",
                ]
            )
        story.append(_styled_table(lake_rows, col_widths=[3.4, 0.8, 1.0, 1.0, 1.2, 1.0]))
        story.append(Spacer(1, 10))

    story.append(Paragraph("Key Diagnostic Notes", styles["SectionHeading"]))
    diagnostic_pairs = [
        ("Best epoch", str(summary["best_epoch"])),
        ("Validation windows", str(diagnostics["val_window_count"])),
        (
            "Observed evaluation mean/std",
            f"{diagnostics['eval_true_mean']:.4f} / {diagnostics['eval_true_std']:.4f}",
        ),
        (
            "Predicted evaluation mean/std",
            f"{diagnostics['eval_pred_mean']:.4f} / {diagnostics['eval_pred_std']:.4f}",
        ),
    ]
    if seasonal_rollout_overview:
        diagnostic_pairs.extend(
            [
                ("Seasonal-rollout overlap points", str(seasonal_rollout_overview["overlap_rows"])),
                (
                    "Seasonal-rollout span",
                    f"{seasonal_rollout_overview['rollout_start']} to {seasonal_rollout_overview['rollout_end']}",
                ),
                (
                    "Observed overlap span",
                    f"{seasonal_rollout_overview['overlap_start']} to {seasonal_rollout_overview['overlap_end']}",
                ),
            ]
        )
    story.append(
        _build_key_value_pairs_table(
            diagnostic_pairs,
            styles=styles,
            col_widths=[2.0, 3.0, 2.0, 3.0],
            font_size=8.4,
            left_columns=[0, 1, 2, 3],
            center_columns=[],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Interpretation: if predicted std is much smaller than observed std, the model is behaving too conservatively and is under-expressing peak-to-trough variation.",
            styles["BodySmall"],
        )
    )
    return story


def _build_setup_section(report_data: dict[str, Any], styles) -> list[Any]:
    config = report_data["config"]
    run_summary = report_data["run_summary"]
    feature_columns = [config["features"]["time_channel_name"], *config["features"]["feature_columns"]]

    story: list[Any] = [
        Paragraph("0. Training Inputs And Hyperparameters", styles["SectionHeading"]),
        Paragraph(
            "The model consumes one irregular path per valid anchor row. Each path contains relative time plus meteorological and seasonal channels; the target total_ice_m is predicted, not fed back as an input feature.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Input Channels", styles["Heading3"]),
    ]

    channel_lines = [f"{index + 1}. {column}" for index, column in enumerate(feature_columns)]
    story.append(_build_two_column_text_table(channel_lines, styles=styles, col_widths=[4.9, 4.9]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Core Training Configuration", styles["Heading3"]))
    config_pairs = [
        ("window.window_days", str(config["window"]["window_days"])),
        ("coeffs.interpolation", str(config["coeffs"]["interpolation"])),
        ("model.method", str(config["model"]["method"])),
        ("model.hidden_channels", str(config["model"]["hidden_channels"])),
        ("model.hidden_hidden_channels", str(config["model"]["hidden_hidden_channels"])),
        ("model.num_hidden_layers", str(config["model"]["num_hidden_layers"])),
        ("train.batch_size", str(config["train"]["batch_size"])),
        ("train.learning_rate", str(config["train"]["learning_rate"])),
        ("train.weight_decay", str(config["train"]["weight_decay"])),
        ("train.loss", str(config["train"]["loss"])),
        ("train.huber_delta", str(config["train"]["huber_delta"])),
        ("train.max_epochs", str(config["train"]["max_epochs"])),
        ("train.gradient_clip_norm", str(config["train"]["gradient_clip_norm"])),
        ("custom_split.val_fraction", str(config["custom_split"]["val_fraction"])),
        ("custom_split.min_train_rows_per_lake", str(config["custom_split"]["min_train_rows_per_lake"])),
        ("custom_split.min_val_rows_per_lake", str(config["custom_split"]["min_val_rows_per_lake"])),
        ("custom_split.max_train_windows_per_lake", str(config["custom_split"]["max_train_windows_per_lake"])),
        ("custom_split.max_val_windows_per_lake", str(config["custom_split"]["max_val_windows_per_lake"])),
        ("custom_split.target_lake_test_start", str(config["custom_split"].get("target_lake_test_start", "none"))),
    ]
    story.append(
        _build_key_value_pairs_table(
            config_pairs,
            styles=styles,
            col_widths=[2.6, 1.7, 2.6, 1.7],
            font_size=8.2,
            left_columns=[0, 2],
            center_columns=[1, 3],
        )
    )

    physics_cfg = config["train"].get("physics_loss", {})
    if physics_cfg.get("enabled", False):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Physics Loss Configuration", styles["Heading3"]))
        physics_pairs = _build_physics_pairs(physics_cfg, run_summary)
        story.append(
            _build_key_value_pairs_table(
                physics_pairs,
                styles=styles,
                col_widths=[2.6, 1.7, 2.6, 1.7],
                font_size=8.2,
                left_columns=[0, 2],
                center_columns=[1, 3],
            )
        )
    return story


def _build_physics_pairs(
    physics_cfg: dict[str, Any],
    run_summary: dict[str, Any],
) -> list[tuple[str, str]]:
    mode = str(physics_cfg.get("mode", "legacy_stefan"))
    common_pairs = [
        ("train.physics_loss.mode", mode),
        ("train.physics_loss.lambda_nn", str(physics_cfg.get("lambda_nn", "n/a"))),
    ]
    if mode == "tc2020_curve":
        return common_pairs + [
            ("train.physics_loss.lambda_curve_grow", str(physics_cfg.get("lambda_curve_grow", "n/a"))),
            ("run_summary.physics_lambda_curve_grow", str(run_summary.get("physics_lambda_curve_grow", "n/a"))),
            ("train.physics_loss.lambda_curve_decay", str(physics_cfg.get("lambda_curve_decay", "n/a"))),
            ("run_summary.physics_lambda_curve_decay", str(run_summary.get("physics_lambda_curve_decay", "n/a"))),
            ("train.physics_loss.enable_decay", str(physics_cfg.get("enable_decay", "n/a"))),
            ("run_summary.physics_enable_decay", str(run_summary.get("physics_enable_decay", "n/a"))),
            ("train.physics_loss.init_alpha", str(physics_cfg.get("init_alpha", "n/a"))),
            ("run_summary.physics_alpha", str(run_summary.get("physics_alpha", "n/a"))),
            ("train.physics_loss.init_alpha_decay", str(physics_cfg.get("init_alpha_decay", "n/a"))),
            ("run_summary.physics_alpha_decay", str(run_summary.get("physics_alpha_decay", "n/a"))),
            ("train.physics_loss.temperature_column", str(physics_cfg.get("temperature_column", "n/a"))),
            ("train.physics_loss.afdd_column", str(physics_cfg.get("afdd_column", "n/a"))),
            ("train.physics_loss.atdd_column", str(physics_cfg.get("atdd_column", "n/a"))),
            ("train.physics_loss.growth_phase_column", str(physics_cfg.get("growth_phase_column", "n/a"))),
            ("train.physics_loss.decay_phase_column", str(physics_cfg.get("decay_phase_column", "n/a"))),
            ("train.physics_loss.stable_ice_mask_column", str(physics_cfg.get("stable_ice_mask_column", "n/a"))),
            ("train.physics_loss.season_start_month", str(physics_cfg.get("season_start_month", "n/a"))),
            ("train.physics_loss.stable_ice_min_m", str(physics_cfg.get("stable_ice_min_m", "n/a"))),
            ("train.physics_loss.phase_tolerance_m", str(physics_cfg.get("phase_tolerance_m", "n/a"))),
        ]
    return common_pairs + [
        ("train.physics_loss.rule", str(physics_cfg.get("rule", "none"))),
        ("train.physics_loss.lambda_st", str(physics_cfg.get("lambda_st", "n/a"))),
        ("run_summary.physics_lambda_st", str(run_summary.get("physics_lambda_st", "n/a"))),
        ("train.physics_loss.init_kappa", str(physics_cfg.get("init_kappa", "n/a"))),
        ("run_summary.physics_kappa", str(run_summary.get("physics_kappa", "n/a"))),
        ("train.physics_loss.min_prev_ice_m", str(physics_cfg.get("min_prev_ice_m", "n/a"))),
        (
            "train.physics_loss.grow_temp_threshold_celsius",
            str(physics_cfg.get("grow_temp_threshold_celsius", "n/a")),
        ),
        ("train.physics_loss.prev_ice_column", str(physics_cfg.get("prev_ice_column", "n/a"))),
        ("train.physics_loss.gap_days_column", str(physics_cfg.get("gap_days_column", "n/a"))),
        ("train.physics_loss.temperature_column", str(physics_cfg.get("temperature_column", "n/a"))),
        ("train.physics_loss.prev_available_column", str(physics_cfg.get("prev_available_column", "n/a"))),
    ]


def _build_data_selection_section(report_data: dict[str, Any], styles) -> list[Any]:
    row_stats = report_data["row_stats"]
    window_stats = report_data["window_stats"]
    totals = report_data["totals"]

    story: list[Any] = [
        Paragraph("2. Lake Selection, Windowing, And Balance Effects", styles["SectionHeading"]),
        Paragraph(
            f"Each source lake is first split by time into train/val. The target lake {_safe_lake_label(report_data['target_lake_label'])} is split so that rows before the cutoff are further divided into train/val, while rows from the cutoff onward are withheld from training and only used later as observation checkpoints for seasonal-rollout testing. A row only becomes a valid training or validation window if there are at least two observations inside the {report_data['config']['window']['window_days']}-day lookback. After that, train and val windows are capped per lake for balance.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("Lake Selection Counts From The Source File", styles["Heading3"]),
    ]

    row_table = [["Lake", "Rows In Source", "Train Rows", "Val Rows", "Withheld Test Rows"]]
    for _, row in row_stats.iterrows():
        row_table.append(
            [
                _safe_lake_label(row["lake_name"]),
                str(int(row["source_rows"])),
                str(int(row["train_rows"])),
                str(int(row["val_rows"])),
                str(int(row["test_rows"])),
            ]
        )
    story.append(_styled_table(row_table, col_widths=[3.5, 1.1, 1.0, 1.0, 1.0], font_size=8.3))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Window Eligibility And Balance Effects", styles["Heading3"]))
    window_table = [
        [
            "Lake",
            "Train Valid",
            "Train Kept",
            "Val Valid",
            "Val Kept",
            "Test Kept",
            "Median Points",
        ]
    ]
    for _, row in window_stats.iterrows():
        window_table.append(
            [
                _safe_lake_label(row["lake_name"]),
                str(int(row["train_pre_balance"])),
                str(int(row["train_post_balance"])),
                str(int(row["val_pre_balance"])),
                str(int(row["val_post_balance"])),
                str(int(row["test_post_balance"])),
                f"{row['median_window_points']:.1f}",
            ]
        )
    story.append(
        _styled_table(
            window_table,
            col_widths=[3.0, 0.95, 0.85, 0.95, 0.85, 0.85, 1.0],
            font_size=8.0,
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("Total Counts", styles["Heading3"]))
    total_pairs = [
        ("Total included source rows", str(totals["source_rows"])),
        ("Train / val / test rows", f"{totals['train_rows']} / {totals['val_rows']} / {totals['test_rows']}"),
        ("Train windows before / after balance", f"{totals['train_pre_balance']} / {totals['train_post_balance']}"),
        ("Val windows before / after balance", f"{totals['val_pre_balance']} / {totals['val_post_balance']}"),
        ("Held-out test windows", str(totals["test_post_balance"])),
    ]
    story.append(
        _build_key_value_pairs_table(
            total_pairs,
            styles=styles,
            col_widths=[2.5, 2.4, 2.5, 2.4],
            font_size=8.4,
            left_columns=[0, 1, 2, 3],
            center_columns=[],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Held-out target rows are evaluated only through seasonal-rollout overlap, not as standalone test windows.",
            styles["BodySmall"],
        )
    )
    return story


def _build_figures_section(report_data: dict[str, Any], styles) -> list[Any]:
    history = report_data["history"]
    evaluation_predictions = report_data["test_predictions"]
    seasonal_rollout_predictions = report_data["seasonal_rollout_predictions"]
    seasonal_overlap_predictions = report_data["seasonal_rollout_overlap_predictions"]
    target_lake = report_data["target_lake_label"]

    story: list[Any] = [Paragraph("1. Core Result Figures", styles["SectionHeading"])]

    figure_groups: list[list[tuple[str, Any]]] = [
        [
            ("Training loss curve", create_loss_curves_figure(history)),
            ("Validation metric curves", create_metric_curves_figure(history)),
        ]
    ]

    if seasonal_rollout_predictions is not None and not seasonal_rollout_predictions.empty:
        seasonal_group = [
            ("Seasonal rollout time series", create_lake_timeseries_figure(seasonal_rollout_predictions, target_lake))
        ]
        if seasonal_overlap_predictions is not None and not seasonal_overlap_predictions.empty:
            observed_start = seasonal_overlap_predictions["sample_datetime"].min()
            observed_end = seasonal_overlap_predictions["sample_datetime"].max()
            seasonal_group.append(
                (
                    "Seasonal rollout focus on observed date range",
                    create_lake_timeseries_figure(
                        seasonal_rollout_predictions,
                        target_lake,
                        start_datetime=observed_start,
                        end_datetime=observed_end,
                        title="Seasonal Rollout Focus On Observed Date Range",
                    ),
                )
            )
        figure_groups.append(seasonal_group)

    if evaluation_predictions is not None and not evaluation_predictions.empty:
        figure_groups.append(
            [
                ("Evaluation predicted vs observed", create_pred_vs_obs_figure(evaluation_predictions)),
                ("Prediction vs observation distribution", create_prediction_distribution_figure(evaluation_predictions)),
            ]
        )

    if (
        seasonal_overlap_predictions is not None
        and not seasonal_overlap_predictions.empty
        and not _prediction_frames_match(evaluation_predictions, seasonal_overlap_predictions)
    ):
        figure_groups.append(
            [("Seasonal rollout overlap: predicted vs observed", create_pred_vs_obs_figure(seasonal_overlap_predictions))]
        )

    for group_index, figure_group in enumerate(figure_groups):
        figure_block: list[Any] = []
        for figure_index, (title, fig) in enumerate(figure_group):
            figure_block.extend(_build_figure_block(title, fig, styles))
            if figure_index != len(figure_group) - 1:
                figure_block.append(Spacer(1, 8))
        story.append(KeepTogether(figure_block))
        if group_index != len(figure_groups) - 1:
            story.append(Spacer(1, 10))
    return story


def _build_seasonal_rollout_section(report_data: dict[str, Any], styles) -> list[Any]:
    seasonal_df = report_data["seasonal_rollout_predictions"]
    overlap_df = seasonal_df.loc[seasonal_df["y_true"].notna()].copy()
    overlap_metrics = report_data["seasonal_rollout_overlap_metrics"]
    target_lake = report_data["target_lake_label"]

    story: list[Any] = [
        Paragraph("3. Seasonal Rollout Test", styles["SectionHeading"]),
        Paragraph(
            "The only test in this report is the autoregressive seasonal rollout. Starting from the configured test start date, the model is driven day by day with Xiaoxingkai 12:00 ERA5 forcing and keeps feeding its previous predicted ice thickness into the next step. Because of that, an earlier test start creates a longer accumulated trajectory, and the model behavior at the January 2026 observation period can change with the chosen start date.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]

    metric_rows = [["Field", "Value"]]
    if overlap_metrics:
        for key in ("count", "rmse", "mae", "r2", "bias"):
            if key in overlap_metrics:
                value = overlap_metrics[key]
                metric_rows.append([key, f"{value:.4f}" if isinstance(value, (int, float)) and key != "count" else str(value)])
    metric_rows.extend(
        [
            ["test_method", "seasonal_rollout_overlap"],
            ["test_start_datetime", str(overlap_metrics.get("test_start_datetime"))],
            ["rollout_rows", str(int(len(seasonal_df)))],
            ["overlap_rows", str(int(len(overlap_df)))],
            ["start_datetime", str(seasonal_df["sample_datetime"].min())],
            ["end_datetime", str(seasonal_df["sample_datetime"].max())],
            ["observed_overlap_start", str(overlap_metrics.get("overlap_start_datetime"))],
            ["observed_overlap_end", str(overlap_metrics.get("overlap_end_datetime"))],
        ]
    )
    story.append(_styled_table(metric_rows, col_widths=[2.8, 3.6]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Seasonal rollout time series", styles["Heading3"]))
    story.append(_figure_to_reportlab_image(create_lake_timeseries_figure(seasonal_df, target_lake), max_width=9.6 * inch))
    story.append(Spacer(1, 8))

    if not overlap_df.empty:
        observed_start = overlap_df["sample_datetime"].min()
        observed_end = overlap_df["sample_datetime"].max()
        story.append(Paragraph("Seasonal rollout focus on observed date range", styles["Heading3"]))
        story.append(
            _figure_to_reportlab_image(
                create_lake_timeseries_figure(
                    seasonal_df,
                    target_lake,
                    start_datetime=observed_start,
                    end_datetime=observed_end,
                    title="Seasonal Rollout Focus On Observed Date Range",
                ),
                max_width=9.6 * inch,
            )
        )
        story.append(Spacer(1, 8))

        story.append(Paragraph("Seasonal rollout overlap: predicted vs observed", styles["Heading3"]))
        story.append(_figure_to_reportlab_image(create_pred_vs_obs_figure(overlap_df), max_width=6.8 * inch))
        story.append(Spacer(1, 8))
    return story


def _collect_report_data(run_dir: Path) -> dict[str, Any]:
    config = _read_yaml(run_dir / "config_merged.yaml")
    run_manifest = json.loads((run_dir / "artifacts" / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "metrics.csv")
    history = pd.read_csv(run_dir / "epoch_summary.csv")
    val_predictions = pd.read_csv(run_dir / "val_predictions.csv", parse_dates=["sample_datetime"])
    test_predictions_path = run_dir / "test_predictions.csv"
    if test_predictions_path.exists():
        test_predictions = pd.read_csv(test_predictions_path, parse_dates=["sample_datetime"])
    else:
        test_predictions = pd.DataFrame()
    per_lake_metrics_path = run_dir / "per_lake_metrics.csv"
    per_lake_metrics = pd.read_csv(per_lake_metrics_path) if per_lake_metrics_path.exists() else pd.DataFrame()
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    seasonal_rollout_predictions_path = run_dir / "seasonal_rollout_predictions.csv"
    seasonal_rollout_predictions = (
        pd.read_csv(seasonal_rollout_predictions_path, parse_dates=["sample_datetime"])
        if seasonal_rollout_predictions_path.exists()
        else pd.DataFrame()
    )
    seasonal_rollout_overlap_predictions_path = run_dir / "seasonal_rollout_overlap_predictions.csv"
    if seasonal_rollout_overlap_predictions_path.exists():
        seasonal_rollout_overlap_predictions = pd.read_csv(
            seasonal_rollout_overlap_predictions_path,
            parse_dates=["sample_datetime"],
        )
    elif not seasonal_rollout_predictions.empty and "y_true" in seasonal_rollout_predictions.columns:
        seasonal_rollout_overlap_predictions = seasonal_rollout_predictions.loc[
            seasonal_rollout_predictions["y_true"].notna()
        ].copy()
    else:
        seasonal_rollout_overlap_predictions = pd.DataFrame()
    seasonal_rollout_overlap_metrics_path = run_dir / "seasonal_rollout_overlap_metrics.json"
    seasonal_rollout_overlap_metrics = (
        json.loads(seasonal_rollout_overlap_metrics_path.read_text(encoding="utf-8"))
        if seasonal_rollout_overlap_metrics_path.exists()
        else {}
    )

    prepared_df = _load_report_prepared_dataframe(config, run_manifest)

    include_lakes_cfg = config["data"].get("include_lakes")
    if include_lakes_cfg:
        include_lakes = [str(lake) for lake in include_lakes_cfg]
    else:
        include_lakes = sorted(prepared_df[config["data"]["lake_column"]].dropna().astype(str).unique().tolist())
    target_lake = _resolve_report_target_lake(include_lakes)
    train_lakes = [lake for lake in include_lakes if lake != target_lake]
    fold_df = _build_fold_dataframe(prepared_df, train_lakes, target_lake, config)
    row_stats = _compute_row_stats(prepared_df, fold_df, config["data"]["lake_column"])
    window_stats = _compute_window_stats(fold_df, config, run_manifest)
    totals = _compute_totals(row_stats, window_stats)

    evaluation_predictions = test_predictions if not test_predictions.empty else val_predictions
    diagnostics = {
        "eval_true_mean": float(evaluation_predictions["y_true"].mean()),
        "eval_true_std": float(evaluation_predictions["y_true"].std(ddof=0)),
        "eval_pred_mean": float(evaluation_predictions["y_pred"].mean()),
        "eval_pred_std": float(evaluation_predictions["y_pred"].std(ddof=0)),
        "val_window_count": int(len(val_predictions)),
        "test_window_count": int(len(test_predictions)),
    }
    seasonal_rollout_overview: dict[str, Any] = {}
    if not seasonal_rollout_predictions.empty:
        overlap_start = "none"
        overlap_end = "none"
        if not seasonal_rollout_overlap_predictions.empty:
            overlap_start = str(seasonal_rollout_overlap_predictions["sample_datetime"].min())
            overlap_end = str(seasonal_rollout_overlap_predictions["sample_datetime"].max())
        seasonal_rollout_overview = {
            "rollout_start": str(seasonal_rollout_predictions["sample_datetime"].min()),
            "rollout_end": str(seasonal_rollout_predictions["sample_datetime"].max()),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_rows": int(len(seasonal_rollout_overlap_predictions)),
        }

    raw_excel = _resolve_report_path(config, config["paths"]["raw_excel"])
    target_lake_label = (
        str(evaluation_predictions["lake_name"].iloc[0]) if not evaluation_predictions.empty else XIAOXINGKAI_NAME
    )
    return {
        "run_dir": run_dir,
        "experiment_name": config["experiment"]["name"],
        "raw_excel": raw_excel,
        "config": config,
        "metrics": metrics,
        "history": history,
        "test_predictions": evaluation_predictions,
        "seasonal_rollout_predictions": seasonal_rollout_predictions,
        "seasonal_rollout_overlap_predictions": seasonal_rollout_overlap_predictions,
        "seasonal_rollout_overlap_metrics": seasonal_rollout_overlap_metrics,
        "per_lake_metrics": per_lake_metrics,
        "run_summary": run_summary,
        "row_stats": row_stats,
        "window_stats": window_stats,
        "totals": totals,
        "diagnostics": diagnostics,
        "seasonal_rollout_overview": seasonal_rollout_overview,
        "target_lake_label": target_lake_label,
    }


def _load_report_prepared_dataframe(config: dict[str, Any], run_manifest: dict[str, Any]) -> pd.DataFrame:
    datetime_column = config["data"]["datetime_column"]
    prepared_candidates: list[Path] = [_resolve_report_path(config, config["paths"]["prepared_csv"])]

    data_processing_report_path = run_manifest.get("data_processing_report_path")
    if data_processing_report_path:
        report_path = Path(data_processing_report_path)
        if report_path.exists():
            data_processing_report = json.loads(report_path.read_text(encoding="utf-8"))
            prepared_csv_path = data_processing_report.get("prepared_csv_path")
            if prepared_csv_path:
                prepared_candidates.append(_resolve_report_path(config, prepared_csv_path))

    seen_paths: set[Path] = set()
    for candidate in prepared_candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_paths:
            continue
        seen_paths.add(resolved_candidate)
        if resolved_candidate.exists():
            prepared_df = pd.read_csv(resolved_candidate, parse_dates=[datetime_column])
            return filter_include_lakes(prepared_df, config)

    raw_excel = _resolve_report_path(config, config["paths"]["raw_excel"])
    if not raw_excel.exists():
        attempted = ", ".join(str(path) for path in seen_paths) or "<none>"
        raise FileNotFoundError(
            "Unable to load prepared dataframe for report generation. "
            f"Tried prepared CSV path(s): {attempted}. Missing raw Excel: {raw_excel}"
        )

    raw_df = load_raw_excel(raw_excel, sheet_name=config["data"].get("excel_sheet_name"))
    prepared_df, _ = standardize_dataframe(raw_df, config)
    return prepared_df


def _resolve_report_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()

    runtime_cfg = config.get("runtime", {})
    project_root = Path(runtime_cfg.get("project_root", "."))
    return (project_root / path).resolve()


def _resolve_report_target_lake(include_lakes: list[str]) -> str:
    matches = [lake_name for lake_name in include_lakes if "xiaoxingkai" in _normalize_report_lake_name(lake_name)]
    if len(matches) != 1:
        raise ValueError(
            "Unable to resolve Xiaoxingkai target lake for report generation. "
            f"available={include_lakes}"
        )
    return matches[0]


def _time_split_training_lake(
    lake_df: pd.DataFrame,
    val_fraction: float,
    min_val_rows: int,
    min_train_rows: int,
) -> pd.DataFrame:
    lake_df = lake_df.sort_values("sample_datetime").reset_index(drop=True).copy()
    total_rows = len(lake_df)
    proposed_val_rows = max(min_val_rows, int(np.ceil(total_rows * val_fraction)))
    val_rows = min(proposed_val_rows, max(1, total_rows - min_train_rows))
    split_index = total_rows - val_rows
    lake_df["row_split"] = "train"
    lake_df.loc[split_index:, "row_split"] = "val"
    return lake_df


def _build_fold_dataframe(df: pd.DataFrame, train_lakes: list[str], target_lake: str, config: dict) -> pd.DataFrame:
    custom_cfg = config["custom_split"]
    pieces: list[pd.DataFrame] = []
    for lake_name in train_lakes:
        lake_df = df.loc[df[config["data"]["lake_column"]] == lake_name].copy()
        pieces.append(
            _time_split_training_lake(
                lake_df=lake_df,
                val_fraction=float(custom_cfg["val_fraction"]),
                min_val_rows=int(custom_cfg["min_val_rows_per_lake"]),
                min_train_rows=int(custom_cfg["min_train_rows_per_lake"]),
            )
        )
    target_df = df.loc[df[config["data"]["lake_column"]] == target_lake].copy()
    target_df = target_df.sort_values(config["data"]["datetime_column"]).reset_index(drop=True)
    target_test_start = custom_cfg.get("target_lake_test_start")
    if target_test_start:
        cutoff = pd.Timestamp(target_test_start)
        target_history_df = target_df.loc[pd.to_datetime(target_df[config["data"]["datetime_column"]]) < cutoff].copy()
        target_test_df = target_df.loc[pd.to_datetime(target_df[config["data"]["datetime_column"]]) >= cutoff].copy()
        target_history_df = _time_split_training_lake(
            lake_df=target_history_df,
            val_fraction=float(custom_cfg["val_fraction"]),
            min_val_rows=int(custom_cfg["min_val_rows_per_lake"]),
            min_train_rows=int(custom_cfg["min_train_rows_per_lake"]),
        )
        target_test_df["row_split"] = "test"
        pieces.append(target_history_df)
        pieces.append(target_test_df)
    else:
        target_df["row_split"] = "test"
        pieces.append(target_df)
    return pd.concat(pieces, ignore_index=True)


def _compute_row_stats(prepared_df: pd.DataFrame, fold_df: pd.DataFrame, lake_column: str) -> pd.DataFrame:
    source_counts = prepared_df.groupby(lake_column).size().rename("source_rows")
    split_counts = (
        fold_df.groupby([lake_column, "row_split"]).size().unstack(fill_value=0).rename_axis(None, axis=1)
    )
    for split_name in ("train", "val", "test"):
        if split_name not in split_counts.columns:
            split_counts[split_name] = 0
    merged = source_counts.to_frame().join(split_counts[["train", "val", "test"]], how="left").fillna(0).reset_index()
    merged = merged.rename(
        columns={
            lake_column: "lake_name",
            "train": "train_rows",
            "val": "val_rows",
            "test": "test_rows",
        }
    )
    for column in ("source_rows", "train_rows", "val_rows", "test_rows"):
        merged[column] = merged[column].astype(int)
    return merged.sort_values(["test_rows", "lake_name"], ascending=[False, True]).reset_index(drop=True)


def _compute_window_stats(fold_df: pd.DataFrame, config: dict, run_manifest: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    lake_column = data_cfg["lake_column"]
    time_column = data_cfg["datetime_column"]
    feature_columns = config["features"]["feature_columns"]
    target_column = data_cfg["target_column"]
    window_days = int(config["window"]["window_days"])

    pre_counts: dict[tuple[str, str], list[int]] = {}
    for split_name in ("train", "val", "test"):
        anchor_df = fold_df.loc[fold_df["row_split"] == split_name].copy()
        for lake_name, lake_anchor_df in anchor_df.groupby(lake_column):
            lake_history_df = fold_df.loc[fold_df[lake_column] == lake_name].sort_values(time_column).reset_index(drop=True)
            anchor_times = set(pd.to_datetime(lake_anchor_df[time_column]).tolist())
            lengths: list[int] = []
            for anchor_index in range(len(lake_history_df)):
                anchor_time = lake_history_df.iloc[anchor_index][time_column]
                if anchor_time not in anchor_times:
                    continue
                built = _build_single_window(
                    history_df=lake_history_df.iloc[: anchor_index + 1].copy(),
                    feature_columns=feature_columns,
                    time_column=time_column,
                    target_column=target_column,
                    window_days=window_days,
                    lake_name=str(lake_name),
                    anchor_index=anchor_index,
                )
                if built is not None:
                    lengths.append(int(built["length"]))
            pre_counts[(str(lake_name), split_name)] = lengths

    post_counts: dict[tuple[str, str], list[int]] = {}
    for split_name, key in (("train", "train_window_path"), ("val", "val_window_path"), ("test", "test_window_path")):
        bundle_path = run_manifest.get(key)
        if not bundle_path:
            continue
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        metadata_df = pd.DataFrame(bundle["metadata"])
        if metadata_df.empty:
            continue
        for lake_name, lake_df in metadata_df.groupby("lake_name"):
            post_counts[(str(lake_name), split_name)] = lake_df["length"].astype(int).tolist()

    all_lakes = sorted({lake_name for lake_name, _ in pre_counts} | {lake_name for lake_name, _ in post_counts})
    rows: list[dict[str, Any]] = []
    for lake_name in all_lakes:
        kept_lengths = (
            post_counts.get((lake_name, "train"), [])
            + post_counts.get((lake_name, "val"), [])
            + post_counts.get((lake_name, "test"), [])
        )
        rows.append(
            {
                "lake_name": lake_name,
                "train_pre_balance": len(pre_counts.get((lake_name, "train"), [])),
                "train_post_balance": len(post_counts.get((lake_name, "train"), [])),
                "val_pre_balance": len(pre_counts.get((lake_name, "val"), [])),
                "val_post_balance": len(post_counts.get((lake_name, "val"), [])),
                "test_post_balance": len(post_counts.get((lake_name, "test"), [])),
                "median_window_points": float(pd.Series(kept_lengths).median()) if kept_lengths else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["test_post_balance", "lake_name"], ascending=[False, True]).reset_index(drop=True)


def _compute_totals(row_stats: pd.DataFrame, window_stats: pd.DataFrame) -> dict[str, int]:
    return {
        "source_rows": int(row_stats["source_rows"].sum()),
        "train_rows": int(row_stats["train_rows"].sum()),
        "val_rows": int(row_stats["val_rows"].sum()),
        "test_rows": int(row_stats["test_rows"].sum()),
        "train_pre_balance": int(window_stats["train_pre_balance"].sum()),
        "train_post_balance": int(window_stats["train_post_balance"].sum()),
        "val_pre_balance": int(window_stats["val_pre_balance"].sum()),
        "val_post_balance": int(window_stats["val_post_balance"].sum()),
        "test_post_balance": int(window_stats["test_post_balance"].sum()),
    }


def _figure_to_reportlab_image(fig, max_width: float, max_height: float = 4.8 * inch) -> Image:
    import matplotlib.pyplot as plt

    fig_width_inches, fig_height_inches = fig.get_size_inches()
    if fig_width_inches <= 0 or fig_height_inches <= 0:
        fig_width_inches, fig_height_inches = 1.0, 1.0

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image(buffer)

    width_scale = max_width / float(fig_width_inches * inch)
    height_scale = max_height / float(fig_height_inches * inch)
    scale = min(width_scale, height_scale)
    image.drawWidth = float(fig_width_inches * inch) * scale
    image.drawHeight = float(fig_height_inches * inch) * scale
    return image


def _styled_table(rows: list[list[str]], col_widths: list[float], font_size: float = 9.0) -> Table:
    table = Table(rows, colWidths=[width * inch for width in col_widths], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#7f8fa6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _safe_lake_label(value: str) -> str:
    text = str(value)
    return (
        text.replace("【", "[")
        .replace("】", "] ")
        .replace("Lake ", "Lake ")
        .replace("  ", " ")
        .strip()
    )


def _normalize_report_lake_name(value: str) -> str:
    ascii_text = str(value).encode("ascii", errors="ignore").decode("ascii")
    return "".join(character.lower() for character in ascii_text if character.isalnum())


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _draw_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#566573"))
    canvas.drawRightString(doc.pagesize[0] - 0.45 * inch, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _table_paragraph(text: str, style) -> Paragraph:
    return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), style)


def _build_two_column_text_table(items: list[str], styles, col_widths: list[float]) -> Table:
    half = ceil(len(items) / 2)
    rows: list[list[Any]] = []
    for row_index in range(half):
        left_item = _table_paragraph(items[row_index], styles["BodyCompact"])
        right_index = row_index + half
        right_item: Any = ""
        if right_index < len(items):
            right_item = _table_paragraph(items[right_index], styles["BodyCompact"])
        rows.append([left_item, right_item])

    table = Table(rows, colWidths=[width * inch for width in col_widths])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _build_key_value_pairs_table(
    pairs: list[tuple[str, str]],
    styles,
    col_widths: list[float],
    font_size: float,
    left_columns: list[int],
    center_columns: list[int],
) -> Table:
    rows: list[list[Any]] = [["Field", "Value", "Field", "Value"]]
    for start_index in range(0, len(pairs), 2):
        row: list[Any] = []
        for offset in range(2):
            pair_index = start_index + offset
            if pair_index < len(pairs):
                field, value = pairs[pair_index]
                row.extend(
                    [
                        _table_paragraph(field, styles["BodyCompact"]),
                        _table_paragraph(value, styles["BodyCompact"]),
                    ]
                )
            else:
                row.extend(["", ""])
        rows.append(row)
    return _styled_table(
        rows,
        col_widths=col_widths,
        font_size=font_size,
        left_columns=left_columns,
        center_columns=center_columns,
    )


def _build_figure_block(title: str, fig, styles) -> list[Any]:
    return [
        Paragraph(title, styles["Heading3"]),
        _figure_to_reportlab_image(fig, max_width=10.0 * inch, max_height=2.85 * inch),
    ]


def _prediction_frames_match(left_df: pd.DataFrame, right_df: pd.DataFrame) -> bool:
    if left_df is None or right_df is None or left_df.empty or right_df.empty or len(left_df) != len(right_df):
        return False

    comparison_columns = ["sample_datetime", "lake_name", "y_true", "y_pred"]
    if any(column not in left_df.columns or column not in right_df.columns for column in comparison_columns):
        return False

    left_sorted = left_df[comparison_columns].sort_values(["sample_datetime", "lake_name"]).reset_index(drop=True).copy()
    right_sorted = right_df[comparison_columns].sort_values(["sample_datetime", "lake_name"]).reset_index(drop=True).copy()
    left_sorted["sample_datetime"] = left_sorted["sample_datetime"].astype(str)
    right_sorted["sample_datetime"] = right_sorted["sample_datetime"].astype(str)

    metadata_matches = left_sorted[["sample_datetime", "lake_name"]].equals(right_sorted[["sample_datetime", "lake_name"]])
    numeric_matches = np.allclose(
        left_sorted[["y_true", "y_pred"]].to_numpy(dtype=float),
        right_sorted[["y_true", "y_pred"]].to_numpy(dtype=float),
        equal_nan=True,
    )
    return bool(metadata_matches and numeric_matches)


def _styled_table(
    rows: list[list[Any]],
    col_widths: list[float],
    font_size: float = 9.0,
    left_columns: list[int] | None = None,
    center_columns: list[int] | None = None,
) -> Table:
    if not rows:
        raise ValueError("Table rows must not be empty.")

    column_count = len(rows[0])
    left_columns = [0] if left_columns is None else left_columns
    center_columns = (
        [column_index for column_index in range(column_count) if column_index not in left_columns]
        if center_columns is None
        else center_columns
    )

    table = Table(rows, colWidths=[width * inch for width in col_widths], repeatRows=1)
    table_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#7f8fa6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]
    for column_index in left_columns:
        table_styles.append(("ALIGN", (column_index, 1), (column_index, -1), "LEFT"))
    for column_index in center_columns:
        table_styles.append(("ALIGN", (column_index, 1), (column_index, -1), "CENTER"))
    table.setStyle(TableStyle(table_styles))
    return table


def _safe_lake_label(value: str) -> str:
    text = str(value)
    return text.replace("【", "[").replace("】", "] ").replace("  ", " ").strip()
