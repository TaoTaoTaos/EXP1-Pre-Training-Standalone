from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PhysicsLossBreakdown:
    """物理损失各分量的拆分结果。

    这个结构体既用于训练时把总物理损失加回主损失，
    也用于日志记录，便于观察不同物理约束分别贡献了多少惩罚。
    """

    # Stefan 风格增长约束项，刻画预测冰厚是否符合平方厚度增长关系。
    stefan: torch.Tensor
    # TC2020-PLUS 增长期 Stefan 增量项，专门约束“今天该增长多少”的热力学残差。
    stefan_grow: torch.Tensor
    # TC2020 曲线在生长期上的约束项。
    curve_grow: torch.Tensor
    # TC2020 曲线在消融期上的约束项。
    curve_decay: torch.Tensor
    # 非负约束项，防止模型输出负冰厚。
    nonneg: torch.Tensor
    # 物理损失总和，通常会按配置中的权重线性组合。
    total: torch.Tensor
    # 当前使用的有效 kappa（经过 softplus 后保证为正）。
    kappa: torch.Tensor
    # 当前使用的有效 alpha（经过 softplus 后保证为正）。
    alpha: torch.Tensor
    # 当前使用的有效 alpha_decay（经过 softplus 后保证为正）。
    alpha_decay: torch.Tensor


def build_loss(config: dict) -> nn.Module:
    """根据配置构建主回归损失函数。

    这里的“主损失”指的是模型预测值与监督目标之间的统计损失，
    例如 MSE / MAE / Huber。物理损失不在这里处理，而是在训练循环中
    单独计算后再叠加到主损失上。
    """
    loss_name = config["train"]["loss"]
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=float(config["train"]["huber_delta"]))
    raise ValueError(f"Unsupported loss: {loss_name}")


def check_loss_is_finite(loss: torch.Tensor) -> None:
    """对 loss 做数值稳定性检查。

    一旦训练中出现 NaN 或 Inf，继续反向传播通常只会把问题扩散，
    因此这里选择立刻报错，尽早暴露数据、学习率或物理约束中的异常。
    """
    if not torch.isfinite(loss):
        raise ValueError(f"Loss is not finite: {loss.item()}")


def compute_physics_loss(
    predictions_transformed: torch.Tensor,
    physics_context: dict[str, torch.Tensor] | None,
    config: dict,
    target_transform: str,
    theta_kappa: torch.Tensor | None,
    theta_alpha: torch.Tensor | None = None,
    theta_alpha_decay: torch.Tensor | None = None,
) -> PhysicsLossBreakdown:
    """物理损失。

    参数说明：
    - predictions_transformed: 模型输出的目标值，仍处在训练时使用的 target transform 空间中。
    - physics_context: 与当前 batch 对齐的物理辅助字段，例如上一时刻冰厚、间隔天数、气温等。
    - config: 全局配置，用于读取物理损失开关、阈值和权重。
    - target_transform: 目标变量使用的变换方式，例如 none 或 log1p。
    - theta_kappa: legacy_stefan 模式下的可训练无约束参数；通过 softplus 映射为正的 kappa。
    - theta_alpha: tc2020_curve 模式下的可训练无约束参数；通过 softplus 映射为正的 alpha。
    - theta_alpha_decay: tc2020_curve 模式下的可训练无约束参数；通过 softplus 映射为正的 alpha_decay。

    设计思路：
    1. 先把模型输出从变换空间还原回真实冰厚空间。
    2. 根据不同 mode，选择 legacy Stefan 约束或 tc2020 曲线约束。
    3. 仅在具有明确物理解释的样本子集上施加对应约束。
    4. 再额外加入非负惩罚，避免输出出现物理上无意义的负冰厚。
    """
    physics_cfg = config["train"].get("physics_loss", {})
    zero = predictions_transformed.new_zeros(())
    if not physics_cfg.get("enabled", False):
        # 没开启物理损失时，返回全 0，占位但不影响主损失流程。
        return PhysicsLossBreakdown(
            stefan=zero,
            stefan_grow=zero,
            curve_grow=zero,
            curve_decay=zero,
            nonneg=zero,
            total=zero,
            kappa=zero,
            alpha=zero,
            alpha_decay=zero,
        )
    if physics_context is None:
        raise ValueError(
            "Physics loss is enabled but physics_context is missing from the batch."
        )
    mode = str(physics_cfg.get("mode", "legacy_stefan"))
    if mode == "legacy_stefan":
        return _compute_legacy_stefan_loss(
            predictions_transformed=predictions_transformed,
            physics_context=physics_context,
            physics_cfg=physics_cfg,
            target_transform=target_transform,
            theta_kappa=theta_kappa,
        )
    if mode == "tc2020_curve":
        return _compute_tc2020_curve_loss(
            predictions_transformed=predictions_transformed,
            physics_context=physics_context,
            physics_cfg=physics_cfg,
            target_transform=target_transform,
            theta_kappa=theta_kappa,
            theta_alpha=theta_alpha,
            theta_alpha_decay=theta_alpha_decay,
        )
    raise ValueError(f"Unsupported physics loss mode: {mode}")


def compute_tc2020_curve_thickness(
    afdd: torch.Tensor,
    atdd: torch.Tensor,
    theta_alpha: torch.Tensor,
    theta_alpha_decay: torch.Tensor | None,
    enable_decay: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 TC2020 曲线先验构造冰厚参考曲线。"""
    alpha = F.softplus(theta_alpha)
    if theta_alpha_decay is None:
        alpha_decay = alpha.new_tensor(1.0)
    else:
        alpha_decay = F.softplus(theta_alpha_decay)

    h_grow = alpha * torch.sqrt(torch.clamp(afdd, min=0.0))
    if enable_decay:
        h_curve = h_grow - alpha_decay * torch.clamp(atdd, min=0.0)
    else:
        h_curve = h_grow
    return h_curve, alpha, alpha_decay


def _compute_legacy_stefan_loss(
    *,
    predictions_transformed: torch.Tensor,
    physics_context: dict[str, torch.Tensor],
    physics_cfg: dict,
    target_transform: str,
    theta_kappa: torch.Tensor | None,
) -> PhysicsLossBreakdown:
    if theta_kappa is None:
        raise ValueError("Physics loss mode 'legacy_stefan' requires theta_kappa to be initialized.")

    # 物理约束必须在真实冰厚空间中计算，不能直接在 log1p 等变换空间中做。
    pred_ice = inverse_transform_target_tensor(
        predictions_transformed, target_transform
    )
    prev_ice = _get_physics_field(physics_context, "ice_prev_m", pred_ice.device)
    gap_days = _get_physics_field(physics_context, "ice_prev_gap_days", pred_ice.device)
    temp_c = _get_physics_field(
        physics_context, "Air_Temperature_celsius", pred_ice.device
    )
    prev_ok = (
        _get_physics_field(physics_context, "ice_prev_available", pred_ice.device) > 0.5
    )

    # 只在“物理条件足够可信且满足结冰增长前提”的样本上施加 Stefan 约束：
    # - 必须有上一时刻冰厚
    # - 上一时刻冰厚不能太接近 0，否则平方厚度关系不稳定
    # - 当前温度必须低于增长阈值，才认为存在冻结驱动力
    grow_mask = (
        prev_ok
        & (prev_ice > float(physics_cfg.get("min_prev_ice_m", 1.0e-3)))
        & (temp_c < float(physics_cfg.get("grow_temp_threshold_celsius", -0.5)))
    )

    # 冻结度日（freezing degree days）的近似写法：
    # 温度越低、间隔时间越长，冻结驱动越强。这里用 ReLU(-temp_c) 去掉非冻结情况。
    delta_F = torch.relu(-temp_c) * gap_days
    # 通过 softplus 保证 kappa 始终为正，避免训练过程中跑到物理上不可解释的负值。
    kappa = F.softplus(theta_kappa)
    # Stefan 风格残差：
    # h_t^2 - h_{t-1}^2 ≈ kappa * delta_F
    # 残差越接近 0，说明预测越符合所设定的结冰增长规律。
    stefan_residual = (pred_ice.square() - prev_ice.square()) - kappa * delta_F
    # 只对满足 grow_mask 的样本求均值；分母加上极小值，避免全空掩码时除零。
    loss_stefan = _masked_mean(stefan_residual.square(), grow_mask)
    # 非负约束：当预测冰厚 < 0 时触发惩罚，否则为 0。
    loss_nonneg = torch.relu(-pred_ice).square().mean()
    # 总物理损失由多个分量按权重线性组合。
    total = (
        float(physics_cfg.get("lambda_st", 1.0)) * loss_stefan
        + float(physics_cfg.get("lambda_nn", 1.0)) * loss_nonneg
    )
    return PhysicsLossBreakdown(
        stefan=loss_stefan,
        stefan_grow=loss_stefan,
        curve_grow=predictions_transformed.new_zeros(()),
        curve_decay=predictions_transformed.new_zeros(()),
        nonneg=loss_nonneg,
        total=total,
        kappa=kappa,
        alpha=predictions_transformed.new_zeros(()),
        alpha_decay=predictions_transformed.new_zeros(()),
    )


def _compute_tc2020_curve_loss(
    *,
    predictions_transformed: torch.Tensor,
    physics_context: dict[str, torch.Tensor],
    physics_cfg: dict,
    target_transform: str,
    theta_kappa: torch.Tensor | None,
    theta_alpha: torch.Tensor | None,
    theta_alpha_decay: torch.Tensor | None,
) -> PhysicsLossBreakdown:
    if theta_alpha is None:
        raise ValueError("Physics loss mode 'tc2020_curve' requires theta_alpha to be initialized.")

    lambda_curve_grow = float(
        _require_physics_config_value(physics_cfg, "lambda_curve_grow")
    )
    lambda_curve_decay = float(
        _require_physics_config_value(physics_cfg, "lambda_curve_decay")
    )
    lambda_nn = float(_require_physics_config_value(physics_cfg, "lambda_nn"))
    enable_decay = bool(_require_physics_config_value(physics_cfg, "enable_decay"))
    enable_stefan_grow = bool(physics_cfg.get("enable_stefan_grow", False))

    pred_ice = inverse_transform_target_tensor(
        predictions_transformed, target_transform
    )

    # AFDD/ATDD 分别表示累计冻结度日和累计融化度日：
    # AFDD 控制结冰阶段的曲线抬升，ATDD 则在启用消融项时控制厚度回落。
    afdd = _get_physics_field(physics_context, "afdd", pred_ice.device)
    atdd = _get_physics_field(physics_context, "atdd", pred_ice.device)
    growth_phase = (
        _get_physics_field(physics_context, "is_growth_phase", pred_ice.device) > 0.5
    )
    decay_phase = (
        _get_physics_field(physics_context, "is_decay_phase", pred_ice.device) > 0.5
    )
    stable_mask = (
        _get_physics_field(physics_context, "stable_ice_mask", pred_ice.device) > 0.5
    )

    h_curve, alpha, alpha_decay = compute_tc2020_curve_thickness(
        afdd=afdd,
        atdd=atdd,
        theta_alpha=theta_alpha,
        theta_alpha_decay=theta_alpha_decay,
        enable_decay=enable_decay,
    )

    # 曲线约束只在 stable mask 内启用：
    # 这些样本通常已经进入稳定结冰/消融阶段，AFDD/ATDD 与厚度关系更可解释，
    # 能减少初冻、融尽或观测噪声较大阶段对曲线先验的误导。
    grow_mask = growth_phase & stable_mask
    decay_mask = decay_phase & stable_mask & enable_decay

    loss_curve_grow = _masked_mean((pred_ice - h_curve).pow(2), grow_mask)
    loss_curve_decay = _masked_mean((pred_ice - h_curve).pow(2), decay_mask)
    zero = predictions_transformed.new_zeros(())

    if enable_stefan_grow:
        if theta_kappa is None:
            raise ValueError(
                "Physics loss mode 'tc2020_curve' with enable_stefan_grow=true requires theta_kappa to be initialized."
            )

        # TC2020-PLUS 只在增长期额外启用 Stefan 增量项：
        # 这里沿用已有的 grow_mask，再叠加上一时刻冰厚、时间间隔和低温条件，避免在融化期、
        # 初生薄冰或缺少历史观测的样本上强行施加“今天该长多少”的热力学关系。
        prev_ice = _get_physics_field(physics_context, "ice_prev_m", pred_ice.device)
        gap_days = _get_physics_field(
            physics_context, "ice_prev_gap_days", pred_ice.device
        )
        temp_c = _get_physics_field(
            physics_context, "Air_Temperature_celsius", pred_ice.device
        )
        prev_ok = (
            _get_physics_field(
                physics_context, "ice_prev_available", pred_ice.device
            )
            > 0.5
        )
        min_prev_ice_m = float(physics_cfg.get("min_prev_ice_m", 0.05))
        grow_temp_threshold_celsius = float(
            physics_cfg.get("grow_temp_threshold_celsius", -0.5)
        )
        stefan_mask = (
            grow_mask
            & prev_ok
            & (prev_ice > min_prev_ice_m)
            & (temp_c < grow_temp_threshold_celsius)
        )

        # 冻结驱动量 delta_F = relu(-temp_c) * gap_days。
        # 气温越低、距离上一观测越久，理论上可支持的平方冰厚增量越大。
        delta_F = torch.relu(-temp_c) * gap_days
        # kappa 仍用无约束参数 theta_kappa 训练，再通过 softplus 映射到正数，
        # 保证 Stefan 系数在优化过程中始终有物理意义。
        kappa = F.softplus(theta_kappa)
        stefan_residual = (pred_ice.square() - prev_ice.square()) - kappa * delta_F
        loss_stefan_grow = _masked_mean(stefan_residual.square(), stefan_mask)
    else:
        kappa = zero
        loss_stefan_grow = zero

    # 非负项保持与现有实现一致，避免新模式改动主监督外的基础约束行为。
    loss_nonneg = torch.relu(-pred_ice).square().mean()
    total = (
        lambda_curve_grow * loss_curve_grow
        + lambda_curve_decay * loss_curve_decay
        + float(physics_cfg.get("lambda_st", 0.0)) * loss_stefan_grow
        + lambda_nn * loss_nonneg
    )
    return PhysicsLossBreakdown(
        stefan=loss_stefan_grow,
        stefan_grow=loss_stefan_grow,
        curve_grow=loss_curve_grow,
        curve_decay=loss_curve_decay,
        nonneg=loss_nonneg,
        total=total,
        kappa=kappa,
        alpha=alpha,
        alpha_decay=alpha_decay,
    )


def inverse_transform_target_tensor(
    values: torch.Tensor, transform: str
) -> torch.Tensor:
    """在 Torch 张量上执行目标变量的逆变换。

    训练时目标变量可能经过了 `none`、`log1p` 等预处理。
    物理损失计算需要回到真实物理量空间，因此这里提供一个与
    数据预处理逻辑一致的张量版本逆变换。
    """
    if transform == "none":
        return values
    if transform == "log1p":
        return torch.expm1(values)
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_softplus(value: float) -> float:
    """把正数空间中的初始化值映射回 softplus 之前的无约束参数空间。

    训练时真正参与优化的是 `theta_kappa`，而不是直接优化 `kappa`。
    这是因为我们希望通过 `softplus(theta_kappa)` 保证最终的 `kappa > 0`。
    因此如果配置里给的是期望的正初始化值，就需要先做一次 inverse softplus。
    """
    if value <= 0:
        raise ValueError(f"Expected a positive kappa initialization, got {value}.")
    return float(
        torch.log(torch.expm1(torch.tensor(value, dtype=torch.float32))).item()
    )


def _get_physics_field(
    physics_context: dict[str, torch.Tensor], field_name: str, device: torch.device
) -> torch.Tensor:
    """从 physics_context 中读取指定字段，并移动到当前计算设备。

    这里统一封装字段检查，避免在主逻辑里反复写存在性判断。
    如果缺字段，直接抛错，比静默返回错误结果更容易定位问题。
    """
    if field_name not in physics_context:
        raise ValueError(f"Physics loss requires '{field_name}' in physics_context.")
    return physics_context[field_name].to(device)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | bool) -> torch.Tensor:
    mask_float = mask.float()
    return (values * mask_float).sum() / (mask_float.sum() + 1.0e-8)


def _require_physics_config_value(physics_cfg: dict, field_name: str) -> object:
    if field_name not in physics_cfg:
        raise ValueError(
            f"Physics loss mode '{physics_cfg.get('mode', 'legacy_stefan')}' requires config field '{field_name}'."
        )
    return physics_cfg[field_name]
