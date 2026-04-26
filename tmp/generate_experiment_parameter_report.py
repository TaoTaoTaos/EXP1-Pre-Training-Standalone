from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lakeice_ncde.config import load_config, load_yaml_with_extends  # noqa: E402

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


EXPERIMENT_FILES = [
    ("EXP0", "EXP0_pretrain_autoreg.yaml"),
    ("EXP1", "EXP1_transfer_autoreg.yaml"),
    ("EXP2", "EXP2_transfer_autoreg_stefan.yaml"),
    ("EXP2-B", "EXP2-B-tc2020.yaml"),
    ("PLUS", "EXP2-B-tc2020-PLUS.yaml"),
]

SEARCH_FILES = [
    "EXP2_B_tc2020_structure_search_v1.yaml",
    "EXP2_B_tc2020_curve_physics_search_v1.yaml",
    "Find_EXP2_tc2020_PLUS.yaml",
]

BATCH_FILES = [
    "Run-ALL.yaml",
    "Run-EXP2.yaml",
    "Run-EXP2-B-tc2020.yaml",
    "Run-EXP2-B-tc2020-PLUS.yaml",
]


PARAMETER_NOTES: dict[str, tuple[str, str]] = {
    "experiment.name": ("实验 ID，也是输出目录和注册表的主键。", "保证每个实验的产物隔离，避免不同配置互相覆盖。"),
    "experiment.description": ("人读的实验说明。", "不参与训练，但帮助追踪实验意图。"),
    "experiment.save_figures": ("是否保存图像产物。", "只影响报告和可视化产物，不影响模型。"),
    "experiment.save_pdf_report": ("是否为训练 run 输出 PDF 报告。", "只影响实验记录，不参与优化。"),
    "debug.enabled": ("调试开关。", "开启时通常配合小数据/小 epoch 快速试跑。"),
    "debug.max_lakes": ("调试时最多保留的湖泊数。", "降低数据规模，加快验证流程。"),
    "debug.max_windows_per_split": ("调试时每个 split 最多窗口数。", "降低训练步数，不能代表最终精度。"),
    "debug.max_epochs": ("调试时覆盖最大 epoch。", "快速检查代码路径，不用于正式训练。"),
    "paths.raw_excel": ("原始观测和 ERA5 合并表。", "数据源改变会改变全部训练样本。"),
    "paths.prepared_csv": ("预处理后的缓存 CSV。", "避免重复清洗数据；配置变化会触发或要求重建。"),
    "paths.validation_report_json": ("数据校验报告。", "帮助发现缺列、缺失目标、非法时间等输入问题。"),
    "paths.feature_schema_json": ("特征 schema 输出。", "记录模型实际看到的输入通道。"),
    "paths.split_root": ("split 缓存根目录。", "保存训练/验证/测试划分。"),
    "paths.window_root": ("窗口缓存根目录。", "保存不规则时间窗口张量。"),
    "paths.coeff_root": ("CDE 插值系数缓存根目录。", "减少重复计算 Hermite/linear 系数。"),
    "paths.artifact_root": ("实验级数据产物目录。", "集中放置 schema、报告、缓存等。"),
    "paths.output_root": ("训练输出根目录。", "控制 run、summary、PDF、搜索结果落点。"),
    "data.excel_sheet_name": ("Excel 工作表名。", "读错 sheet 会直接改变训练数据。"),
    "data.lake_column": ("湖泊名称列。", "用于按湖分组、识别目标湖。"),
    "data.lake_id_column": ("湖泊 ID 列。", "当前主要用于数据记录，不直接进入特征。"),
    "data.include_lakes": ("可选湖泊白名单。", "限制训练湖集合；null 表示使用全部可用湖。"),
    "data.datetime_column": ("观测时间列。", "决定时间排序、窗口锚点和季节 rollout 对齐。"),
    "data.era5_datetime_column": ("ERA5 时间列。", "用于外部气象驱动对齐。"),
    "data.target_column": ("预测目标列。", "这里是总冰厚 total_ice_m。"),
    "data.doy_column": ("年内日序列。", "可构造季节周期特征。"),
    "data.latitude_column": ("纬度列。", "可作为空间特征或数据校验字段。"),
    "data.longitude_column": ("经度列。", "可作为空间特征或数据校验字段。"),
    "data.required_columns": ("基础必需列清单。", "缺列时应尽早失败，避免模型吃到错误数据。"),
    "features.time_channel_name": ("CDE 路径的相对时间通道名。", "让模型知道窗口内部点的时间位置。"),
    "features.cyclical_columns": ("周期特征列清单。", "用于显式编码季节性；当前主线未启用。"),
    "features.feature_columns": ("输入特征列清单。", "直接决定模型输入维度和可学习信息。"),
    "features.target_transform": ("目标值变换。", "none 表示直接拟合米制冰厚；log1p 等会改变损失尺度。"),
    "features.input_scaler": ("输入特征缩放方式。", "standard 让不同量纲特征处于可训练尺度。"),
    "features.exclude_target_from_inputs": ("是否禁止当前目标作为输入。", "避免标签泄漏；历史冰厚通过 ice_prev_* 单独进入。"),
    "split.strategy": ("划分策略名。", "主线使用时间留出逻辑。"),
    "split.name": ("划分名称。", "进入缓存路径，确保每个实验的 split 可追踪。"),
    "split.seed": ("划分和采样随机种子。", "保证可复现实验。"),
    "custom_split.val_fraction": ("每个训练湖末尾留作验证的比例。", "控制验证集覆盖季节后段的程度。"),
    "custom_split.min_val_rows_per_lake": ("每湖最少验证观测数。", "避免验证集过小导致指标抖动。"),
    "custom_split.min_train_rows_per_lake": ("每湖最少训练观测数。", "保证每湖有足够历史可学习。"),
    "custom_split.max_train_windows_per_lake": ("每湖最多训练窗口数。", "平衡多观测湖和少观测湖的权重。"),
    "custom_split.max_val_windows_per_lake": ("每湖最多验证窗口数。", "控制验证计算量和湖泊权重。"),
    "custom_split.target_lake_test_start": ("目标湖进入测试保留的日期。", "null 表示目标湖全留作测试；有日期则此前历史参与迁移学习。"),
    "window.window_days": ("每个样本向前看的窗口长度。", "越长越能看季节背景，越短越偏局部动态。"),
    "window.min_points": ("构成窗口的最少观测点。", "避免只有一个点时 CDE 路径退化。"),
    "window.anchor_on_every_observation": ("是否每个观测点都作为预测锚点。", "增加样本数，让训练覆盖更多时间位置。"),
    "window.save_raw_windows": ("是否保存原始窗口。", "便于调试和复现，不改变训练数学。"),
    "coeffs.interpolation": ("CDE 路径插值方法。", "hermite 对不规则观测更平滑，决定 torchcde 系数形式。"),
    "model.hidden_channels": ("Neural CDE 隐状态宽度。", "越大容量越强，也更容易过拟合和变慢。"),
    "model.hidden_hidden_channels": ("向量场 MLP 的内部宽度。", "控制 CDE 动力函数复杂度。"),
    "model.num_hidden_layers": ("向量场 MLP 层数。", "越深可表达非线性越复杂，但训练更不稳定。"),
    "model.dropout": ("MLP dropout 比例。", "抑制过拟合；太高会欠拟合。"),
    "model.method": ("CDE ODE 求解器。", "rk4 固定步更可控；dopri5 自适应但开销不同。"),
    "model.use_adjoint": ("是否使用 adjoint 反传。", "省显存但可能更慢/数值路径不同。"),
    "model.nonnegative_output": ("输出是否经 Softplus 保证非负。", "符合冰厚非负约束，减少负预测。"),
    "train.seed": ("训练随机种子。", "影响初始化、采样和可复现性。"),
    "train.device": ("训练设备。", "auto 会优先 CUDA，否则 CPU。"),
    "train.batch_size": ("每步训练样本数。", "影响梯度噪声、显存和速度。"),
    "train.batch_parallel": ("是否按同形状窗口组批量前向。", "提升变长窗口训练效率。"),
    "train.num_workers": ("DataLoader 进程数。", "Windows 上 0 更稳，较大值可能更快但更易出问题。"),
    "train.optimizer": ("优化器。", "AdamW 同时做自适应学习率和权重衰减。"),
    "train.learning_rate": ("学习率。", "最关键优化尺度，过大震荡，过小收敛慢。"),
    "train.weight_decay": ("权重衰减。", "L2 式正则，降低过拟合。"),
    "train.max_epochs": ("最大训练轮数。", "上限而非一定跑满，早停会提前结束。"),
    "train.gradient_clip_norm": ("梯度裁剪阈值。", "限制梯度爆炸，特别适合 CDE/物理损失混合训练。"),
    "train.loss": ("监督损失类型。", "Huber 在小误差近似 MSE，大误差近似 MAE，抗异常值。"),
    "train.huber_delta": ("Huber 损失转折阈值。", "越小越像 MAE，越大越像 MSE。"),
    "train.early_stopping.patience": ("早停等待 epoch 数。", "验证集长期无改进就停止，防止过拟合和浪费时间。"),
    "train.early_stopping.min_delta": ("视为改进的最小幅度。", "过滤微小指标噪声。"),
    "train.scheduler.name": ("学习率调度器。", "reduce_on_plateau 在验证停滞时降低学习率。"),
    "train.scheduler.factor": ("降学习率倍率。", "0.5 表示每次减半。"),
    "train.scheduler.patience": ("调度器等待 epoch 数。", "比早停更早介入，给模型继续细调机会。"),
    "train.scheduler.min_lr": ("最小学习率。", "避免学习率被降到没有训练意义。"),
    "train.physics_loss.enabled": ("是否启用物理正则。", "开启后总损失等于监督损失加物理项。"),
    "train.physics_loss.rule": ("legacy Stefan 规则名。", "用于标记 Stefan 增长残差。"),
    "train.physics_loss.mode": ("物理损失模式。", "legacy_stefan 使用增量物理；tc2020_curve 使用季节累计曲线。"),
    "train.physics_loss.lambda_st": ("Stefan 增量残差权重。", "约束冰厚增长与冻结度日增量一致。"),
    "train.physics_loss.lambda_nn": ("非负惩罚权重。", "惩罚负冰厚；在 Softplus 输出下通常是双保险。"),
    "train.physics_loss.init_kappa": ("Stefan 系数初值。", "经 softplus 变正，控制冻结度日到冰厚平方增长的比例。"),
    "train.physics_loss.min_prev_ice_m": ("启用 Stefan 项的前一日冰厚下限。", "过滤开水或极薄冰阶段，减少不可靠残差。"),
    "train.physics_loss.grow_temp_threshold_celsius": ("判定增长条件的气温阈值。", "只在足够冷时施加增长物理约束。"),
    "train.physics_loss.prev_ice_column": ("前一观测冰厚列名。", "给自回归特征和 Stefan 残差使用。"),
    "train.physics_loss.gap_days_column": ("相邻观测间隔天数列名。", "把温度驱动转换成累计冻结量。"),
    "train.physics_loss.temperature_column": ("物理损失使用的温度列。", "驱动 AFDD/ATDD 和 Stefan 增长条件。"),
    "train.physics_loss.prev_available_column": ("前一观测是否可用的标记列。", "避免首个样本误用缺失历史。"),
    "train.physics_loss.lambda_curve_grow": ("TC2020 增长期曲线损失权重。", "把预测拉向 alpha * sqrt(AFDD) 的季节增长曲线。"),
    "train.physics_loss.lambda_curve_decay": ("TC2020 衰退期曲线损失权重。", "把融化期预测拉向增长曲线减 ATDD 衰减。"),
    "train.physics_loss.enable_decay": ("是否启用 TC2020 衰退项。", "True 时曲线包含 alpha_decay * ATDD。"),
    "train.physics_loss.init_alpha": ("TC2020 增长系数初值。", "控制 AFDD 到冰厚的转换尺度。"),
    "train.physics_loss.init_alpha_decay": ("TC2020 衰退系数初值。", "控制 ATDD 融化消减强度。"),
    "train.physics_loss.afdd_column": ("累计冻结度日列名。", "TC2020 增长曲线的核心驱动。"),
    "train.physics_loss.atdd_column": ("累计融化度日列名。", "TC2020 衰退曲线的核心驱动。"),
    "train.physics_loss.growth_phase_column": ("增长阶段 mask 列名。", "决定哪些样本施加 grow curve loss。"),
    "train.physics_loss.decay_phase_column": ("衰退阶段 mask 列名。", "决定哪些样本施加 decay curve loss。"),
    "train.physics_loss.stable_ice_mask_column": ("稳定冰状态 mask 列名。", "过滤无冰/薄冰噪声样本。"),
    "train.physics_loss.season_start_month": ("冰季累计起始月份。", "决定 AFDD/ATDD 从哪一月重新累计。"),
    "train.physics_loss.stable_ice_min_m": ("稳定冰 mask 的冰厚下限。", "越大越保守，物理项覆盖样本更少。"),
    "train.physics_loss.phase_tolerance_m": ("增长/衰退阶段判定容差。", "避免微小观测噪声把阶段翻转。"),
    "train.physics_loss.enable_stefan_grow": ("PLUS 中是否叠加 Stefan 增长期残差。", "给 TC2020 曲线再加局部增量物理约束。"),
    "train.physics_loss.enable_rollout_stability": ("是否启用闭环 rollout 稳定损失。", "训练时模拟把当前预测作为下一步历史冰厚，减少部署漂移。"),
    "train.physics_loss.lambda_rollout_stability": ("rollout 稳定损失权重。", "越大越重视闭环稳定，过大可能牺牲单步拟合。"),
    "train.physics_loss.rollout_stability_huber_beta": ("rollout 稳定项的 Smooth L1 beta。", "控制闭环误差对异常点的敏感度。"),
    "train.physics_loss.rollout_stability_detach_prev_prediction": ("闭环稳定项是否截断前一预测梯度。", "True 让稳定项更像数据增强，避免二阶依赖过强。"),
    "eval.plot_sample_windows": ("报告里绘制多少个样本窗口。", "只影响诊断图数量。"),
    "eval.interpolation_debug_points": ("插值调试曲线点数。", "只影响可视化平滑程度。"),
    "eval.prediction_clip_min": ("评估时预测下限裁剪。", "配合非负输出，避免负值影响指标。"),
    "eval.metrics": ("评估指标清单。", "RMSE/MAE/R2/Bias/negative_count 共同看精度和偏差。"),
    "seasonal_rollout.enabled": ("是否运行连续季节 rollout。", "开启后 test 指标来自 rollout 与观测重叠日期。"),
    "seasonal_rollout.autoregressive_history": ("rollout 是否把预测喂回历史冰厚。", "更接近真实部署，但误差会累积。"),
    "seasonal_rollout.target_lake_name": ("rollout 目标湖。", "当前为 Xiaoxingkai。"),
    "seasonal_rollout.era5_csv": ("目标湖连续 ERA5 驱动 CSV。", "决定测试季节的逐日气象输入。"),
    "seasonal_rollout.test_start_datetime": ("rollout 起始时间。", "决定连续预测从哪天开始。"),
    "seasonal_rollout.end_datetime": ("rollout 结束时间。", "决定测试季节覆盖范围。"),
    "seasonal_rollout.daily_hour": ("每日使用的小时。", "保证 ERA5 驱动与观测时刻对齐。"),
    "seasonal_rollout.reset_initial_state_from_month": ("从某月开始重置初始冰状态。", "帮助新冰季从开水或近零状态开始。"),
    "seasonal_rollout.open_water_projection_enabled": ("是否启用开水投影规则。", "用温度和前冰厚规则限制无冰期虚假增长。"),
    "seasonal_rollout.open_water_temperature_column": ("开水规则使用的温度列。", "通常与训练温度列一致。"),
    "seasonal_rollout.open_water_temperature_threshold_celsius": ("开水规则温度阈值。", "高于阈值更倾向保持开水状态。"),
    "seasonal_rollout.open_water_prev_ice_max_m": ("判定开水/薄冰的前冰厚上限。", "过滤极薄冰导致的自回归误差放大。"),
}


GROUP_TITLES = {
    "experiment": "一、实验身份与记录",
    "debug": "二、调试开关",
    "paths": "三、路径与产物",
    "data": "四、数据列与输入源",
    "features": "五、特征工程",
    "split": "六、基础划分",
    "custom_split": "七、目标湖迁移划分",
    "window": "八、窗口构造",
    "coeffs": "九、CDE 插值系数",
    "model": "十、Neural CDE 模型结构",
    "train": "十一、训练与物理损失",
    "eval": "十二、评估与报告",
    "seasonal_rollout": "十三、季节 rollout 测试",
}


SEARCH_KEY_NOTES = {
    "search.name": ("搜索任务 ID。", "用于 Optuna study 和输出路径命名。"),
    "search.base_batch_config": ("搜索基于哪个 batch 配置运行。", "决定每个 trial 实际启动哪些实验。"),
    "search.output_root": ("搜索输出根目录。", "保存 trial 配置、结果表和 journal。"),
    "search.n_trials": ("计划 trial 数。", "控制搜索预算。"),
    "search.sampler.name": ("采样器类型。", "grid 穷举离散空间；tpe 用贝叶斯式采样探索连续空间。"),
    "search.sampler.seed": ("采样随机种子。", "保证搜索可复现。"),
    "search.sampler.constant_liar": ("并行 TPE 的 optimistic/pessimistic 占位策略。", "减少并行 trial 采样冲突。"),
    "search.storage.type": ("Optuna 存储类型。", "journal 适合本地文件恢复。"),
    "search.storage.path": ("study journal 路径。", "断点续跑和并行 worker 共享状态。"),
    "search.execution.max_parallel_trials": ("最大并行 trial 数。", "控制 GPU/CPU 资源竞争。"),
    "search.objective.experiment_name": ("目标优化的实验名。", "搜索只按这个实验的指标打分。"),
    "search.objective.split": ("目标 split。", "当前用 test，也就是 rollout overlap 指标。"),
    "search.objective.metric": ("阈值判断指标。", "当前主要看 r2。"),
    "search.objective.success_threshold": ("成功阈值。", "用于标记 trial 是否达到目标。"),
    "search.objective.score_formula": ("实际排序打分公式。", "r2_dominant_composite = R2 - 0.20*RMSE - 0.05*log1p(negative_count)。"),
}


SEARCH_PARAM_PURPOSE = {
    "model.hidden_channels": "搜索模型隐状态容量。",
    "model.hidden_hidden_channels": "搜索 CDE 向量场 MLP 宽度。",
    "model.num_hidden_layers": "搜索 CDE 向量场深度。",
    "model.dropout": "搜索正则强度。",
    "train.learning_rate": "搜索优化步长。",
    "train.weight_decay": "搜索权重衰减正则。",
    "train.huber_delta": "搜索 Huber 损失鲁棒阈值。",
    "train.physics_loss.lambda_curve_grow": "搜索 TC2020 增长曲线约束强度。",
    "train.physics_loss.lambda_curve_decay": "搜索 TC2020 衰退曲线约束强度。",
    "train.physics_loss.lambda_st": "搜索 Stefan 增量约束强度。",
    "train.physics_loss.init_kappa": "搜索 Stefan 系数初值。",
    "train.physics_loss.init_alpha": "搜索 TC2020 增长系数初值。",
    "train.physics_loss.init_alpha_decay": "搜索 TC2020 衰退系数初值。",
    "train.physics_loss.min_prev_ice_m": "搜索启用 Stefan 的前冰厚门槛。",
    "train.physics_loss.lambda_rollout_stability": "搜索闭环 rollout 稳定约束强度。",
    "train.physics_loss.rollout_stability_huber_beta": "搜索闭环稳定损失的鲁棒阈值。",
    "train.physics_loss.season_start_month": "搜索冰季累计起始月。",
    "train.physics_loss.stable_ice_min_m": "搜索稳定冰 mask 的冰厚阈值。",
    "train.physics_loss.phase_tolerance_m": "搜索增长/衰退阶段判定容差。",
}


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/Deng.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/Dengb.ttf"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    bold_path = next((path for path in bold_candidates if path.exists()), font_path)
    if font_path is None:
        return "Helvetica", "Helvetica-Bold"
    pdfmetrics.registerFont(TTFont("CN", str(font_path)))
    pdfmetrics.registerFont(TTFont("CN-Bold", str(bold_path)))
    return "CN", "CN-Bold"


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten(value, full_key))
        else:
            output[full_key] = value
    return output


def short_value(value: Any, max_items: int = 8) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
            return f"{value:.4g}"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        items = [short_value(item) for item in value]
        if len(items) > max_items:
            head = ", ".join(items[:max_items])
            return f"{len(items)} 项: {head}, ..."
        return "[" + ", ".join(items) + "]"
    return str(value)


def value_diff(values_by_exp: dict[str, Any]) -> str:
    values = list(values_by_exp.values())
    if not values:
        return ""
    first = short_value(values[0])
    if all(short_value(value) == first for value in values):
        return f"全部: {first}"
    return "<br/>".join(f"{name}: {short_value(value)}" for name, value in values_by_exp.items())


def note_for_key(key: str) -> tuple[str, str]:
    if key in PARAMETER_NOTES:
        return PARAMETER_NOTES[key]
    if key.startswith("train.physics_loss."):
        return ("物理损失扩展参数。", "影响监督损失之外的物理约束强度、mask 或可学习物理系数。")
    if key.startswith("paths."):
        return ("文件路径参数。", "影响读取/缓存/输出位置，不直接改变模型数学。")
    if key.startswith("data."):
        return ("数据字段映射。", "决定如何从表格中取出训练数据和元信息。")
    if key.startswith("features."):
        return ("特征工程参数。", "决定模型输入通道和数值尺度。")
    if key.startswith("model."):
        return ("模型结构参数。", "改变模型容量、数值求解或输出约束。")
    if key.startswith("train."):
        return ("训练超参数。", "改变优化过程、正则强度或损失形状。")
    if key.startswith("seasonal_rollout."):
        return ("季节 rollout 参数。", "影响连续预测测试流程。")
    return ("配置参数。", "请结合所在分组理解其运行时作用。")


def para(text: str, style: ParagraphStyle) -> Paragraph:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("&lt;br/&gt;", "<br/>")
    return Paragraph(text, style)


def make_table(
    rows: list[list[Any]],
    col_widths: list[float],
    styles: dict[str, ParagraphStyle],
    header: bool = True,
) -> Table:
    converted: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        converted_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                converted_row.append(cell)
            else:
                converted_row.append(para(str(cell), styles["table_header" if row_index == 0 and header else "table"]))
        converted.append(converted_row)
    table = Table(converted, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D7DEE8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F7F9FC")))
    table.setStyle(TableStyle(commands))
    return table


def category_for_key(key: str) -> str:
    return key.split(".", 1)[0]


def load_experiment_configs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    configs: dict[str, dict[str, Any]] = {}
    flat_configs: dict[str, dict[str, Any]] = {}
    for label, filename in EXPERIMENT_FILES:
        config_path = PROJECT_ROOT / "configs" / "experiments" / filename
        config = load_config(PROJECT_ROOT, config_path)
        config.pop("runtime", None)
        configs[label] = config
        flat_configs[label] = flatten(config)
    return configs, flat_configs


def build_styles(font_name: str, bold_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["title"] = ParagraphStyle(
        "TitleCN",
        parent=base["Title"],
        fontName=bold_name,
        fontSize=24,
        leading=31,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#14324A"),
        spaceAfter=14,
    )
    styles["subtitle"] = ParagraphStyle(
        "SubtitleCN",
        parent=base["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4E5C68"),
        spaceAfter=18,
    )
    styles["h1"] = ParagraphStyle(
        "H1CN",
        parent=base["Heading1"],
        fontName=bold_name,
        fontSize=17,
        leading=23,
        textColor=colors.HexColor("#14324A"),
        spaceBefore=10,
        spaceAfter=8,
    )
    styles["h2"] = ParagraphStyle(
        "H2CN",
        parent=base["Heading2"],
        fontName=bold_name,
        fontSize=12.5,
        leading=17,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=8,
        spaceAfter=6,
    )
    styles["body"] = ParagraphStyle(
        "BodyCN",
        parent=base["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#1E2933"),
        spaceAfter=7,
        wordWrap="CJK",
        splitLongWords=True,
    )
    styles["small"] = ParagraphStyle(
        "SmallCN",
        parent=styles["body"],
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#4E5C68"),
    )
    styles["table"] = ParagraphStyle(
        "TableCN",
        parent=styles["small"],
        fontSize=7.2,
        leading=9.6,
        wordWrap="CJK",
        splitLongWords=True,
    )
    styles["table_header"] = ParagraphStyle(
        "TableHeaderCN",
        parent=styles["table"],
        fontName=bold_name,
        textColor=colors.white,
        alignment=TA_LEFT,
    )
    styles["callout"] = ParagraphStyle(
        "CalloutCN",
        parent=styles["body"],
        fontName=font_name,
        fontSize=9,
        leading=14,
        backColor=colors.HexColor("#EEF6FF"),
        borderColor=colors.HexColor("#9BC2E6"),
        borderWidth=0.6,
        borderPadding=7,
        spaceBefore=6,
        spaceAfter=10,
    )
    return styles


def add_header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setFont("CN", 7.5)
    canvas.setFillColor(colors.HexColor("#5B6770"))
    canvas.drawString(1.05 * cm, 0.55 * cm, "Neural-CDE IceTransfer 参数梳理")
    canvas.drawRightString(width - 1.05 * cm, 0.55 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def experiment_overview_rows(configs: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = [["实验", "核心问题", "目标湖训练策略", "物理约束", "模型容量", "测试方式"]]
    for label, config in configs.items():
        physics = config["train"].get("physics_loss", {})
        if not physics.get("enabled", False):
            phys = "无物理项，仅监督 Huber + 非负输出"
        elif physics.get("mode") == "tc2020_curve" and physics.get("enable_stefan_grow"):
            phys = "TC2020 曲线 + Stefan 增长期 + rollout 稳定"
        elif physics.get("mode") == "tc2020_curve":
            phys = "TC2020 AFDD/ATDD 曲线约束"
        else:
            phys = "legacy Stefan 增长残差"
        cutoff = config["custom_split"].get("target_lake_test_start")
        target_strategy = "Xiaoxingkai 全部留作测试" if cutoff is None else f"Xiaoxingkai 在 {cutoff} 前参与 train/val"
        rows.append(
            [
                f"{label}<br/>{config['experiment']['name']}",
                config["experiment"]["description"],
                target_strategy,
                phys,
                (
                    f"hidden={config['model']['hidden_channels']}, "
                    f"field={config['model']['hidden_hidden_channels']}x{config['model']['num_hidden_layers']}, "
                    f"dropout={config['model']['dropout']}"
                ),
                (
                    f"seasonal rollout overlap<br/>"
                    f"{config['seasonal_rollout']['test_start_datetime']} 到 {config['seasonal_rollout']['end_datetime']}"
                ),
            ]
        )
    return rows


def build_parameter_table_for_group(
    group: str,
    keys: list[str],
    flat_configs: dict[str, dict[str, Any]],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    rows = [["参数", "有效值/实验差异", "参数是干嘛的", "对训练的作用"]]
    for key in keys:
        values_by_exp = {exp: flat.get(key, "未定义") for exp, flat in flat_configs.items()}
        purpose, impact = note_for_key(key)
        rows.append([key, value_diff(values_by_exp), purpose, impact])
    return [
        Paragraph(GROUP_TITLES.get(group, group), styles["h2"]),
        make_table(rows, [5.1 * cm, 7.0 * cm, 6.7 * cm, 6.7 * cm], styles),
        Spacer(1, 0.22 * cm),
    ]


def build_feature_lists(configs: dict[str, dict[str, Any]], styles: dict[str, ParagraphStyle]) -> list[Any]:
    exp0 = configs["EXP0"]
    feature_columns = exp0["features"]["feature_columns"]
    required_columns = exp0["data"]["required_columns"]
    rows = [["清单", "项目"]]
    rows.append(["模型输入 feature_columns", "<br/>".join(f"{idx + 1}. {name}" for idx, name in enumerate(feature_columns))])
    rows.append(["required_columns", "<br/>".join(f"{idx + 1}. {name}" for idx, name in enumerate(required_columns))])
    rows.append(
        [
            "自动派生列",
            "ice_prev_m: 前一观测冰厚<br/>ice_prev_gap_days: 与前一观测间隔天数<br/>ice_prev_available: 前一观测是否存在<br/>"
            "tc2020 模式额外派生 afdd、atdd、is_growth_phase、is_decay_phase、stable_ice_mask",
        ]
    )
    return [
        Paragraph("输入列清单", styles["h1"]),
        Paragraph(
            "这里单独列出来，是因为它们最容易让人忘：这些列直接决定模型的输入空间和物理损失上下文。",
            styles["body"],
        ),
        make_table(rows, [5.4 * cm, 19.9 * cm], styles),
    ]


def search_range_text(parameter: dict[str, Any]) -> str:
    ptype = parameter.get("type")
    if ptype in {"int", "float"}:
        parts = [f"{parameter.get('low')} - {parameter.get('high')}"]
        if parameter.get("step") is not None:
            parts.append(f"step={parameter.get('step')}")
        if parameter.get("log"):
            parts.append("log")
        return ", ".join(parts)
    if ptype in {"categorical", "bool"}:
        return "[" + ", ".join(short_value(item) for item in parameter.get("choices", [])) + "]"
    return short_value(parameter)


def build_search_sections(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = [PageBreak(), Paragraph("搜索配置与被调参数", styles["h1"])]
    story.append(
        Paragraph(
            "搜索配置不直接定义模型结构，但会在每个 trial 里覆盖实验 YAML 的某些键。因此报告也把这些“会被自动改写的参数”列全。",
            styles["body"],
        )
    )
    for filename in SEARCH_FILES:
        path = PROJECT_ROOT / "configs" / "search" / filename
        raw = load_yaml_with_extends(path)
        search = raw["search"]
        story.append(Paragraph(filename, styles["h2"]))
        top_rows = [["搜索键", "当前值", "用途", "训练/搜索影响"]]
        flattened = flatten({"search": {k: v for k, v in search.items() if k != "parameters"}})
        for key in [
            "search.name",
            "search.base_batch_config",
            "search.output_root",
            "search.n_trials",
            "search.sampler.name",
            "search.sampler.seed",
            "search.sampler.constant_liar",
            "search.storage.type",
            "search.storage.path",
            "search.execution.max_parallel_trials",
            "search.objective.experiment_name",
            "search.objective.split",
            "search.objective.metric",
            "search.objective.success_threshold",
            "search.objective.score_formula",
        ]:
            if key in flattened:
                purpose, impact = SEARCH_KEY_NOTES.get(key, ("搜索配置。", "影响 trial 如何运行和打分。"))
                top_rows.append([key, short_value(flattened[key]), purpose, impact])
        story.append(make_table(top_rows, [5.2 * cm, 7.0 * cm, 6.8 * cm, 6.3 * cm], styles))
        story.append(Spacer(1, 0.12 * cm))
        param_rows = [["状态", "搜索参数名", "覆盖键", "范围/候选", "scope", "为什么调它"]]
        for param in search.get("parameters", []):
            key = str(param.get("key", ""))
            param_rows.append(
                [
                    "启用" if param.get("enabled") else "关闭",
                    str(param.get("name")),
                    key,
                    search_range_text(param),
                    ", ".join(str(item) for item in param.get("scope", [])),
                    SEARCH_PARAM_PURPOSE.get(key, "候选调参项。"),
                ]
            )
        story.append(make_table(param_rows, [1.7 * cm, 4.4 * cm, 5.8 * cm, 5.5 * cm, 3.0 * cm, 4.9 * cm], styles))
        story.append(Spacer(1, 0.35 * cm))
    return story


def build_batch_section(styles: dict[str, ParagraphStyle]) -> list[Any]:
    rows = [["batch 配置", "max_workers", "包含实验", "用途"]]
    for filename in BATCH_FILES:
        path = PROJECT_ROOT / "configs" / "experiments" / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        batch = raw.get("batch", {})
        experiments = []
        for item in batch.get("experiments", []):
            experiments.append(str(item.get("config", "")))
        rows.append(
            [
                filename,
                short_value(batch.get("max_workers")),
                "<br/>".join(experiments),
                raw.get("experiment", {}).get("description", ""),
            ]
        )
    return [
        PageBreak(),
        Paragraph("批量运行配置", styles["h1"]),
        Paragraph(
            "batch 配置负责把单个实验 YAML 编排成一次运行或一次搜索 trial 的执行计划。",
            styles["body"],
        ),
        make_table(rows, [5.1 * cm, 2.6 * cm, 7.6 * cm, 10.0 * cm], styles),
    ]


def build_report() -> Path:
    font_name, bold_name = register_fonts()
    styles = build_styles(font_name, bold_name)
    output_dir = PROJECT_ROOT / "output" / "pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / os.environ.get(
        "PARAM_REPORT_OUTPUT",
        "exp0_to_plus_parameter_report.pdf",
    )

    configs, flat_configs = load_experiment_configs()
    all_keys = sorted({key for flat in flat_configs.values() for key in flat if not key.startswith("runtime.")})

    story: list[Any] = []
    today = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Spacer(1, 1.1 * cm))
    story.append(Paragraph("EXP0 到 EXP2-B PLUS 参数全景报告", styles["title"]))
    story.append(
        Paragraph(
            f"Neural-CDE-based IceTransfer | 生成时间: {today} | 来源: configs/base + configs/experiments + configs/search",
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "这份报告只覆盖当前主线配置：EXP0、EXP1、EXP2、EXP2-B-tc2020、EXP2-B-tc2020-PLUS，以及对应 batch/search 配置；"
            "不纳入 configs/experiments/Olds 历史备份和 debug_quick。",
            styles["callout"],
        )
    )
    story.append(Paragraph("先看主线逻辑", styles["h1"]))
    bullets = [
        "EXP0: 只用源湖训练，Xiaoxingkai 完全不参与训练，用来做目标湖测试。",
        "EXP1: 在 EXP0 基础上，把 Xiaoxingkai 在 2026-01-01 前的历史加入 train/val，做迁移学习。",
        "EXP2: 在 EXP1 基础上加入 legacy Stefan 增长残差，让冷天冰厚增量更符合冻结度日关系。",
        "EXP2-B: 改用 TC2020 AFDD/ATDD 曲线约束，用季节累计冻融驱动冰厚增长和衰退。",
        "PLUS: 在 TC2020 曲线上再叠加 Stefan 增长期局部约束和闭环 rollout 稳定损失，并缩小模型深度/宽度到当前搜索结果。",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, styles["body"]), bulletColor=colors.HexColor("#1F4E79")) for item in bullets],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(make_table(experiment_overview_rows(configs), [2.9 * cm, 5.7 * cm, 4.8 * cm, 4.9 * cm, 3.5 * cm, 3.5 * cm], styles))
    story.append(PageBreak())

    story.extend(build_feature_lists(configs, styles))
    story.append(PageBreak())
    story.append(Paragraph("调参优先级", styles["h1"]))
    priority_rows = [
        ["层级", "参数", "建议"],
        [
            "通常固定",
            "paths.*, data.*, features.feature_columns, split.name, seasonal_rollout 日期",
            "除非数据源或实验定义变了，否则不要在搜索里乱动。它们决定问题本身。",
        ],
        [
            "可小范围调",
            "model.hidden_channels, model.hidden_hidden_channels, model.num_hidden_layers, dropout",
            "控制模型容量。看 val/test 是否过拟合、是否欠拟合，再决定扩/缩。",
        ],
        [
            "核心优化",
            "learning_rate, weight_decay, huber_delta, gradient_clip_norm",
            "控制训练是否稳、收敛速度和异常值鲁棒性。",
        ],
        [
            "物理约束",
            "lambda_st, lambda_curve_grow, lambda_curve_decay, lambda_rollout_stability, init_*",
            "决定模型听数据还是听物理。权重过小没效果，过大会把数据拟合压坏。",
        ],
        [
            "mask/阶段阈值",
            "stable_ice_min_m, phase_tolerance_m, min_prev_ice_m, season_start_month",
            "决定哪些样本被物理项约束。它们改变物理损失覆盖面，影响很大但解释性也强。",
        ],
    ]
    story.append(make_table(priority_rows, [3.0 * cm, 8.9 * cm, 13.4 * cm], styles))
    story.append(PageBreak())

    story.append(Paragraph("全部有效实验参数", styles["h1"]))
    story.append(
        Paragraph(
            "下面的表是合并 base 配置和实验继承之后的有效值。若某行显示“全部”，表示五个实验值相同；否则逐实验列出差异。",
            styles["body"],
        )
    )

    grouped: dict[str, list[str]] = {}
    for key in all_keys:
        grouped.setdefault(category_for_key(key), []).append(key)
    group_order = [
        "experiment",
        "debug",
        "paths",
        "data",
        "features",
        "split",
        "custom_split",
        "window",
        "coeffs",
        "model",
        "train",
        "eval",
        "seasonal_rollout",
    ]
    for group in group_order:
        if group in grouped:
            story.extend(build_parameter_table_for_group(group, grouped[group], flat_configs, styles))
    for group, keys in grouped.items():
        if group not in group_order:
            story.extend(build_parameter_table_for_group(group, keys, flat_configs, styles))

    story.extend(build_search_sections(styles))
    story.extend(build_batch_section(styles))

    story.append(PageBreak())
    story.append(Paragraph("读这套实验时的心智模型", styles["h1"]))
    final_rows = [
        ["模块", "一句话"],
        ["数据和特征", "告诉模型“看什么”：气象驱动 + 上一次冰厚状态 + 时间位置。"],
        ["窗口和 CDE", "把不规则观测变成连续路径，Neural CDE 学的是这条路径驱动下的隐藏状态演化。"],
        ["监督损失", "让预测贴近观测冰厚，是所有实验的主力信号。"],
        ["物理损失", "在数据少、迁移到目标湖时，给模型加上“冰不会乱长/乱化”的约束。"],
        ["rollout 测试", "模拟真实使用：模型今天的预测会成为明天的历史输入，因此闭环稳定比单步拟合更关键。"],
        ["搜索", "不是新增实验理论，而是在固定理论框架里寻找合适的容量和物理权重。"],
    ]
    story.append(make_table(final_rows, [5.2 * cm, 20.1 * cm], styles))
    story.append(
        Paragraph(
            "最容易混的点：custom_split.target_lake_test_start 控制目标湖哪些历史能进训练；"
            "seasonal_rollout.test_start_datetime 控制连续测试从哪一天开始。二者不是同一个开关。",
            styles["callout"],
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=0.9 * cm,
        bottomMargin=0.9 * cm,
        title="EXP0 to PLUS Parameter Report",
        author="Codex",
    )
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    return output_path


if __name__ == "__main__":
    path = build_report()
    print(path)
