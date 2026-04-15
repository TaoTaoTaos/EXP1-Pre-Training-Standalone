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
    # 非负约束项，防止模型输出负冰厚。
    nonneg: torch.Tensor
    # 物理损失总和，通常会按配置中的权重线性组合。
    total: torch.Tensor
    # 当前使用的有效 kappa（经过 softplus 后保证为正）。
    kappa: torch.Tensor


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
) -> PhysicsLossBreakdown:
    """Stefan 物理损失。

    参数说明：
    - predictions_transformed: 模型输出的目标值，仍处在训练时使用的 target transform 空间中。
    - physics_context: 与当前 batch 对齐的物理辅助字段，例如上一时刻冰厚、间隔天数、气温等。
    - config: 全局配置，用于读取物理损失开关、阈值和权重。
    - target_transform: 目标变量使用的变换方式，例如 none 或 log1p。
    - theta_kappa: 可训练的无约束参数；通过 softplus 映射为正的 kappa。

    设计思路：
    1. 先把模型输出从变换空间还原回真实冰厚空间。
    2. 根据上一时刻冰厚、温度和时间间隔，筛出“应该发生结冰增长”的样本。
    3. 在这些样本上施加 Stefan 型平方厚度增长约束。
    4. 再额外加入非负惩罚，避免输出出现物理上无意义的负冰厚。
    """
    physics_cfg = config["train"].get("physics_loss", {})
    zero = predictions_transformed.new_zeros(())
    if not physics_cfg.get("enabled", False):
        # 没开启物理损失时，返回全 0，占位但不影响主损失流程。
        return PhysicsLossBreakdown(stefan=zero, nonneg=zero, total=zero, kappa=zero)
    if physics_context is None:
        raise ValueError(
            "Physics loss is enabled but physics_context is missing from the batch."
        )
    if theta_kappa is None:
        raise ValueError("Physics loss is enabled but theta_kappa is not initialized.")

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

    grow_mask_float = grow_mask.float()
    # 只对满足 grow_mask 的样本求均值；分母加上极小值，避免全空掩码时除零。
    loss_stefan = (grow_mask_float * stefan_residual.square()).sum() / (
        grow_mask_float.sum() + 1.0e-8
    )
    # 非负约束：当预测冰厚 < 0 时触发惩罚，否则为 0。
    loss_nonneg = torch.relu(-pred_ice).square().mean()
    # 总物理损失由多个分量按权重线性组合。
    total = (
        float(physics_cfg.get("lambda_st", 1.0)) * loss_stefan
        + float(physics_cfg.get("lambda_nn", 1.0)) * loss_nonneg
    )
    return PhysicsLossBreakdown(
        stefan=loss_stefan, nonneg=loss_nonneg, total=total, kappa=kappa
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
