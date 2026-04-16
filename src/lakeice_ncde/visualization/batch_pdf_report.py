from __future__ import annotations

import html
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from lakeice_ncde.visualization.plots import (
    create_comparison_loss_curves_figure,
    create_comparison_metric_bars_figure,
    create_comparison_timeseries_figure,
    create_comparison_validation_metric_curves_figure,
)


FOCUSED_METRICS = [
    ("metrics.val.loss", "val", "loss"),
    ("metrics.val.rmse", "val", "rmse"),
    ("metrics.val.mae", "val", "mae"),
    ("metrics.val.r2", "val", "r2"),
    ("metrics.val.bias", "val", "bias"),
    ("metrics.val.negative_count", "val", "negative_count"),
    ("metrics.test.loss", "test", "loss"),
    ("metrics.test.rmse", "test", "rmse"),
    ("metrics.test.mae", "test", "mae"),
    ("metrics.test.r2", "test", "r2"),
    ("metrics.test.bias", "test", "bias"),
    ("metrics.test.negative_count", "test", "negative_count"),
]

CORE_PARAMETER_KEYS = [
    "custom_split.target_lake_test_start",
    "window.window_days",
    "coeffs.interpolation",
    "model.method",
    "model.hidden_channels",
    "model.hidden_hidden_channels",
    "model.num_hidden_layers",
    "train.device",
    "train.batch_parallel",
    "train.batch_size",
    "train.learning_rate",
    "train.weight_decay",
    "train.max_epochs",
    "train.loss",
    "train.huber_delta",
    "features.target_transform",
]

PHYSICS_PARAMETER_KEYS = [
    "train.physics_loss.enabled",
    "train.physics_loss.rule",
    "train.physics_loss.lambda_st",
    "train.physics_loss.lambda_nn",
    "train.physics_loss.init_kappa",
    "train.physics_loss.grow_temp_threshold_celsius",
]


@dataclass(frozen=True)
class BatchExperimentReportData:
    experiment_name: str
    run_dir: Path
    config: dict[str, Any]
    metrics: pd.DataFrame
    history: pd.DataFrame
    run_summary: dict[str, Any]
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    seasonal_rollout_predictions: pd.DataFrame
    seasonal_rollout_overlap_predictions: pd.DataFrame
    comparison_timeseries: pd.DataFrame


def build_batch_pdf_report(
    batch_run_name: str,
    batch_run_dir: Path,
    experiments: list[dict[str, Any]],
    pdf_path: Path,
) -> Path:
    report_data = _collect_batch_report_data(experiments)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=f"{batch_run_name} summary report",
    )
    styles = _build_styles()
    story: list[Any] = []

    story.extend(_build_cover_section(batch_run_name, batch_run_dir, report_data, styles))
    story.append(PageBreak())
    story.extend(_build_metric_figure_section(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_timeseries_section(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_training_section(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_parameter_section(report_data, styles))

    doc.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
    return pdf_path


def _collect_batch_report_data(experiments: list[dict[str, Any]]) -> list[BatchExperimentReportData]:
    report_items: list[BatchExperimentReportData] = []
    for experiment in experiments:
        experiment_name = str(experiment["experiment_name"])
        run_dir = Path(experiment["run_dir"])
        config = _read_yaml(run_dir / "config_merged.yaml")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        metrics["experiment_name"] = experiment_name
        history = pd.read_csv(run_dir / "epoch_summary.csv")
        run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        val_predictions = _read_predictions(run_dir / "val_predictions.csv")
        test_predictions = _read_predictions(run_dir / "test_predictions.csv")
        seasonal_rollout_predictions = _read_predictions(run_dir / "seasonal_rollout_predictions.csv")
        seasonal_rollout_overlap_predictions = _read_predictions(run_dir / "seasonal_rollout_overlap_predictions.csv")
        comparison_timeseries = _build_comparison_timeseries_frame(
            seasonal_rollout_predictions=seasonal_rollout_predictions,
            overlap_predictions=seasonal_rollout_overlap_predictions,
            test_predictions=test_predictions,
        )
        report_items.append(
            BatchExperimentReportData(
                experiment_name=experiment_name,
                run_dir=run_dir,
                config=config,
                metrics=metrics,
                history=history,
                run_summary=run_summary,
                val_predictions=val_predictions,
                test_predictions=test_predictions,
                seasonal_rollout_predictions=seasonal_rollout_predictions,
                seasonal_rollout_overlap_predictions=seasonal_rollout_overlap_predictions,
                comparison_timeseries=comparison_timeseries,
            )
        )
    return report_items


def _build_cover_section(batch_run_name: str, batch_run_dir: Path, report_data: list[BatchExperimentReportData], styles) -> list[Any]:
    experiment_names = [item.experiment_name for item in report_data]
    focused_table = _build_focused_metrics_table(report_data, styles)
    run_rows = [["Experiment", "Run Directory", "Best Epoch", "Best Val RMSE"]]
    for item in report_data:
        run_rows.append(
            [
                item.experiment_name,
                str(item.run_dir),
                str(item.run_summary.get("best_epoch", "n/a")),
                _format_number(item.run_summary.get("best_val_rmse"), digits=4),
            ]
        )

    return [
        Paragraph(f"{batch_run_name} Comparison Report", styles["ReportTitle"]),
        Paragraph(
            f"Batch run directory: {batch_run_dir}<br/>"
            f"Experiments: {', '.join(experiment_names)}<br/>"
            f"Focus: validation and test metrics, Xiaoxingkai time-series comparison, training curves, parameter differences.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Paragraph("Focused Metric Table", styles["SectionHeading"]),
        focused_table,
        Spacer(1, 10),
        Paragraph("Run Overview", styles["SectionHeading"]),
        _styled_table(run_rows, col_widths=[1.8, 5.5, 1.0, 1.2], font_size=8.2, left_columns=[0, 1]),
        Spacer(1, 8),
        Paragraph(
            "The bar charts on the next page compare the exact validation and test metrics listed above. The later sections overlay the same Xiaoxingkai trajectories and training curves across all experiments.",
            styles["BodySmall"],
        ),
    ]


def _build_metric_figure_section(report_data: list[BatchExperimentReportData], styles) -> list[Any]:
    metric_table = pd.concat([item.metrics for item in report_data], ignore_index=True)
    experiment_names = [item.experiment_name for item in report_data]
    val_fig = create_comparison_metric_bars_figure(metric_table, split="val", experiment_names=experiment_names)
    test_fig = create_comparison_metric_bars_figure(metric_table, split="test", experiment_names=experiment_names)
    return [
        Paragraph("1. Validation And Test Metric Comparison", styles["SectionHeading"]),
        Paragraph(
            "Validation and test are separated into two figures so the relative gains remain readable even when test R2 is strongly negative.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
        Paragraph("Validation Metrics", styles["Heading3"]),
        _figure_to_reportlab_image(val_fig, max_width=10.0 * inch, max_height=3.6 * inch),
        Spacer(1, 8),
        Paragraph("Test Metrics", styles["Heading3"]),
        _figure_to_reportlab_image(test_fig, max_width=10.0 * inch, max_height=3.6 * inch),
    ]


def _build_timeseries_section(report_data: list[BatchExperimentReportData], styles) -> list[Any]:
    experiment_frames = {
        item.experiment_name: item.comparison_timeseries
        for item in report_data
        if item.comparison_timeseries is not None and not item.comparison_timeseries.empty
    }
    if not experiment_frames:
        return [
            Paragraph("2. Xiaoxingkai Time-Series Comparison", styles["SectionHeading"]),
            Paragraph("No comparable time-series predictions were found in the batch run outputs.", styles["BodyText"]),
        ]

    focus_start, focus_end = _resolve_focus_range(report_data)
    overall_fig = create_comparison_timeseries_figure(
        experiment_frames,
        title="Xiaoxingkai Ice Thickness Comparison - Overall",
    )
    focus_title = "Xiaoxingkai Ice Thickness Comparison - Focus On Observed Test Period"
    focus_fig = create_comparison_timeseries_figure(
        experiment_frames,
        title=focus_title,
        start_datetime=focus_start,
        end_datetime=focus_end,
    )
    return [
        Paragraph("2. Xiaoxingkai Time-Series Comparison", styles["SectionHeading"]),
        Paragraph(
            "Each experiment is drawn with connected markers in one shared panel. The black curve is the observed Xiaoxingkai ice thickness wherever observations are available.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
        Paragraph("Overall", styles["Heading3"]),
        _figure_to_reportlab_image(overall_fig, max_width=10.0 * inch, max_height=2.9 * inch),
        Spacer(1, 8),
        Paragraph("Local Focus", styles["Heading3"]),
        _figure_to_reportlab_image(focus_fig, max_width=10.0 * inch, max_height=2.9 * inch),
    ]


def _build_training_section(report_data: list[BatchExperimentReportData], styles) -> list[Any]:
    histories = {item.experiment_name: item.history for item in report_data}
    loss_fig = create_comparison_loss_curves_figure(histories)
    metric_fig = create_comparison_validation_metric_curves_figure(histories)
    return [
        Paragraph("3. Training Curve Comparison", styles["SectionHeading"]),
        Paragraph(
            "These curves overlay the full optimization trajectories so you can compare convergence speed, validation stability, and the effect of Stefan regularization.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
        Paragraph("Training And Validation Loss", styles["Heading3"]),
        _figure_to_reportlab_image(loss_fig, max_width=10.0 * inch, max_height=2.8 * inch),
        Spacer(1, 8),
        Paragraph("Validation Metric Curves", styles["Heading3"]),
        _figure_to_reportlab_image(metric_fig, max_width=10.0 * inch, max_height=2.8 * inch),
    ]


def _build_parameter_section(report_data: list[BatchExperimentReportData], styles) -> list[Any]:
    core_table = _build_parameter_table(report_data, CORE_PARAMETER_KEYS, styles)
    physics_table = _build_parameter_table(report_data, PHYSICS_PARAMETER_KEYS, styles)
    return [
        Paragraph("4. Parameter Comparison", styles["SectionHeading"]),
        Paragraph(
            "Core configuration differences are listed first. Physics-only settings are separated so the EXP2 changes are easy to isolate.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
        Paragraph("Core Parameters", styles["Heading3"]),
        core_table,
        Spacer(1, 10),
        Paragraph("Physics Parameters", styles["Heading3"]),
        physics_table,
    ]


def _build_focused_metrics_table(report_data: list[BatchExperimentReportData], styles) -> Table:
    headers = ["Metric", *[item.experiment_name for item in report_data]]
    rows: list[list[Any]] = [headers]
    metric_table = pd.concat([item.metrics for item in report_data], ignore_index=True)

    for row_label, split_name, metric_name in FOCUSED_METRICS:
        row: list[Any] = [row_label]
        for item in report_data:
            value = metric_table.loc[
                (metric_table["experiment_name"] == item.experiment_name) & (metric_table["split"] == split_name),
                metric_name,
            ].iloc[0]
            row.append(_format_metric_value(metric_name, value))
        rows.append(row)

    return _styled_table(
        rows,
        col_widths=[2.8] + [2.35 for _ in report_data],
        font_size=8.4,
        left_columns=[0],
    )


def _build_parameter_table(report_data: list[BatchExperimentReportData], keys: list[str], styles) -> Table:
    rows: list[list[Any]] = [["Parameter", *[item.experiment_name for item in report_data]]]
    for key in keys:
        rows.append([key, *[_stringify_value(_get_nested_value(item.config, key)) for item in report_data]])
    return _styled_table(
        rows,
        col_widths=[2.9] + [2.3 for _ in report_data],
        font_size=8.1,
        left_columns=[0],
    )


def _build_comparison_timeseries_frame(
    seasonal_rollout_predictions: pd.DataFrame,
    overlap_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if seasonal_rollout_predictions is not None and not seasonal_rollout_predictions.empty:
        frame = seasonal_rollout_predictions.copy()
    elif overlap_predictions is not None and not overlap_predictions.empty:
        frame = overlap_predictions.copy()
    else:
        frame = test_predictions.copy()

    if frame.empty:
        return frame

    frame["sample_datetime"] = pd.to_datetime(frame["sample_datetime"])
    if "y_true" not in frame.columns:
        frame["y_true"] = np.nan

    truth_source = overlap_predictions if overlap_predictions is not None and not overlap_predictions.empty else test_predictions
    if truth_source is not None and not truth_source.empty:
        truth_df = truth_source.copy()
        truth_df["sample_datetime"] = pd.to_datetime(truth_df["sample_datetime"])
        truth_df = truth_df[["sample_datetime", "y_true"]].dropna().drop_duplicates(subset=["sample_datetime"])
        frame = frame.merge(truth_df, on="sample_datetime", how="left", suffixes=("", "_observed"))
        if "y_true_observed" in frame.columns:
            frame["y_true"] = frame["y_true"].combine_first(frame["y_true_observed"])
            frame = frame.drop(columns=["y_true_observed"])

    if "lake_name" not in frame.columns:
        frame["lake_name"] = "Xiaoxingkai"
    return frame.sort_values("sample_datetime").reset_index(drop=True)


def _resolve_focus_range(report_data: list[BatchExperimentReportData]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for item in report_data:
        candidate = item.seasonal_rollout_overlap_predictions
        if candidate is None or candidate.empty:
            candidate = item.test_predictions
        if candidate is None or candidate.empty:
            continue
        candidate = candidate.copy()
        candidate["sample_datetime"] = pd.to_datetime(candidate["sample_datetime"])
        starts.append(candidate["sample_datetime"].min())
        ends.append(candidate["sample_datetime"].max())
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["sample_datetime"])


def _get_nested_value(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return "n/a"
        current = current[part]
    return current


def _stringify_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "none"
    return str(value)


def _format_metric_value(metric_name: str, value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if metric_name == "negative_count":
        return str(int(round(numeric)))
    return f"{numeric:.2f}"


def _format_number(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _build_styles():
    styles = getSampleStyleSheet()
    styles["BodyText"].fontName = "Helvetica"
    styles["Heading3"].fontName = "Helvetica-Bold"
    styles["Title"].fontName = "Helvetica-Bold"
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
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
            fontName="Helvetica-Bold",
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
            fontName="Helvetica",
            fontSize=9.3,
            leading=11.8,
            spaceAfter=4,
        )
    )
    return styles


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


def _styled_table(
    rows: list[list[Any]],
    col_widths: list[float],
    font_size: float = 9.0,
    left_columns: list[int] | None = None,
) -> Table:
    left_columns = [0] if left_columns is None else left_columns
    table = Table(
        [[_table_cell(value) for value in row] for row in rows],
        colWidths=[width * inch for width in col_widths],
        repeatRows=1,
    )
    styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#7f8fa6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]
    for column_index in range(len(rows[0])):
        alignment = "LEFT" if column_index in left_columns else "CENTER"
        styles.append(("ALIGN", (column_index, 1), (column_index, -1), alignment))
    table.setStyle(TableStyle(styles))
    return table


def _table_cell(value: Any) -> Paragraph:
    table_style = ParagraphStyle(
        name="BatchTableCell",
        fontName="Helvetica",
        fontSize=8.4,
        leading=10.2,
    )
    return Paragraph(html.escape(str(value)).replace("\n", "<br/>"), table_style)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _draw_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#566573"))
    canvas.drawRightString(doc.pagesize[0] - 0.45 * inch, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()
