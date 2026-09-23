"""GA 与 PSO 共享的配置校验与候选评估逻辑。

这里严格区分两类“坏候选”，避免评估器自身的故障被搜索算法吞掉：

- **候选不可行**（领域内可预期）：候选违反间距/边界约束，或目标函数主动
  抛出 :class:`InfeasibleCandidate`。这类候选按明确罚分处理，搜索继续。
- **评估器故障**（搜索无法消化的错误）：目标函数抛出其它异常，或返回
  NaN/Inf 等非有限值。此时立即抛出 :class:`EvaluatorError` 终止搜索，
  并保留故障候选的索引、布局与原始原因。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import check_min_spacing


class OptimizerConfigError(ValueError):
    """优化器配置（或构造参数）非法，在创建优化器时立即抛出。"""


class InfeasibleCandidate(Exception):
    """目标函数主动声明当前候选布局不可行。

    这是*领域内*的拒绝信号（例如某布局在物理模型中不被允许），
    与间距/边界违反一样按明确罚分处理，搜索继续。它不是评估器故障。
    """


class EvaluatorError(RuntimeError):
    """目标函数（评估器）自身故障，搜索应立即终止。

    Attributes
    ----------
    candidate_index : int
        故障候选在本批评估中的索引（种群/粒子群内下标）。
    iteration : int
        迭代轮次，0 表示初始种群/粒子群评估。
    stage : str
        故障发生阶段的人类可读描述。
    positions : np.ndarray
        触发故障的候选布局，形状 (n_turbines, 2)。
    reason : str
        原始原因（非有限返回值，或底层异常类型与消息）。
    """

    def __init__(
        self,
        *,
        candidate_index: int,
        iteration: int,
        stage: str,
        positions: np.ndarray,
        reason: str,
    ) -> None:
        self.candidate_index = int(candidate_index)
        self.iteration = int(iteration)
        self.stage = stage
        self.positions = np.asarray(positions, dtype=np.float64).copy()
        self.reason = reason
        super().__init__(
            f"评估器故障（{stage}，候选 #{self.candidate_index}）：{reason}"
        )


@dataclass
class BatchEvaluation:
    """一批候选的评估结果。

    Attributes
    ----------
    fitness : np.ndarray
        适应度数组；不可行候选为负罚分。
    feasible : np.ndarray
        布尔数组，True 表示该候选得到了目标函数给出的有限评估值。
    """

    fitness: np.ndarray
    feasible: np.ndarray


# ---------------------------------------------------------------------------
# 配置校验原语
# ---------------------------------------------------------------------------


def require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    """校验整数参数，返回清洗后的值。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise OptimizerConfigError(
            f"{name} 必须为整数，实际类型为 {type(value).__name__}: {value!r}"
        )
    if minimum is not None and value < minimum:
        raise OptimizerConfigError(f"{name} 不能小于 {minimum}，实际值为 {value}")
    return value


def require_float(
    name: str,
    value: object,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """校验有限浮点参数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OptimizerConfigError(
            f"{name} 必须为数值，实际类型为 {type(value).__name__}: {value!r}"
        )
    value = float(value)
    if not np.isfinite(value):
        raise OptimizerConfigError(f"{name} 必须为有限值，实际值为 {value}")
    if positive and value <= 0.0:
        raise OptimizerConfigError(f"{name} 必须为正数，实际值为 {value}")
    if non_negative and value < 0.0:
        raise OptimizerConfigError(f"{name} 不能为负数，实际值为 {value}")
    return value


def require_rate(name: str, value: object) -> float:
    """校验取值必须落在 [0, 1] 的概率/比例参数。"""
    value = require_float(name, value)
    if not 0.0 <= value <= 1.0:
        raise OptimizerConfigError(f"{name} 必须落在 [0, 1] 区间，实际值为 {value}")
    return value


def validate_constructor_inputs(
    n_turbines: object,
    rotor_diameters: object,
    boundary: object,
    fitness_fn: object,
    min_spacing_multiple: object,
    penalty_factor: object,
) -> np.ndarray:
    """校验两个优化器共同的构造参数，返回清洗后的直径数组。"""
    require_int("n_turbines", n_turbines, minimum=1)

    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    if diameters.shape != (n_turbines,):
        raise OptimizerConfigError(
            f"rotor_diameters 形状必须为 ({n_turbines},)，实际为 {diameters.shape}"
        )
    if not np.all(np.isfinite(diameters)) or np.any(diameters <= 0.0):
        raise OptimizerConfigError("rotor_diameters 必须全部为正的有限值")

    if not isinstance(boundary, SiteBoundary):
        raise OptimizerConfigError(
            f"boundary 必须为 SiteBoundary，实际类型为 {type(boundary).__name__}"
        )
    if boundary.x_max <= boundary.x_min or boundary.y_max <= boundary.y_min:
        raise OptimizerConfigError(
            "场地边界范围退化：x/y 方向跨度必须为正"
        )

    if not callable(fitness_fn):
        raise OptimizerConfigError("fitness_fn 必须可调用")

    require_float("min_spacing_multiple", min_spacing_multiple, positive=True)
    require_float("penalty_factor", penalty_factor, positive=True)

    return diameters


# ---------------------------------------------------------------------------
# 候选评估
# ---------------------------------------------------------------------------


def compute_layout_penalty(
    positions_flat: np.ndarray,
    *,
    n_turbines: int,
    boundary: SiteBoundary,
    min_spacing: float,
    penalty_factor: float,
) -> float:
    """计算候选布局的约束违反惩罚（边界 + 最小间距）。"""
    positions = positions_flat.reshape(n_turbines, 2)

    penalty = 0.0

    inside = boundary.contains_all(positions)
    if not inside.all():
        n_violations = int(np.sum(~inside))
        penalty += n_violations * penalty_factor

    valid, violations = check_min_spacing(positions, min_spacing)
    if not valid:
        for i, j in violations:
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            penalty += (min_spacing - dist) * penalty_factor

    return float(penalty)


def evaluate_batch(
    population: np.ndarray,
    *,
    n_turbines: int,
    fitness_fn: Callable[[np.ndarray], float],
    boundary: SiteBoundary,
    min_spacing: float,
    penalty_factor: float,
    iteration: int,
    stage: str,
) -> BatchEvaluation:
    """评估一整批候选，GA 与 PSO 共用同一套语义。

    对每个候选：

    1. 违反几何约束 → 记负罚分，不调用目标函数；
    2. 目标函数抛出 :class:`InfeasibleCandidate` → 记固定罚分，继续；
    3. 目标函数抛出其它异常 → 包装为 :class:`EvaluatorError` 立即终止，
       保留候选索引、布局与原始异常；
    4. 目标函数返回 NaN/Inf → 同样包装为 :class:`EvaluatorError` 终止；
    5. 否则记录有限评估值。
    """
    pop_size = population.shape[0]
    fitness = np.zeros(pop_size, dtype=np.float64)
    feasible = np.zeros(pop_size, dtype=bool)

    for i in range(pop_size):
        positions = population[i].reshape(n_turbines, 2)

        penalty = compute_layout_penalty(
            population[i],
            n_turbines=n_turbines,
            boundary=boundary,
            min_spacing=min_spacing,
            penalty_factor=penalty_factor,
        )
        if penalty > 0.0:
            fitness[i] = -penalty
            continue

        try:
            value = float(fitness_fn(positions))
        except InfeasibleCandidate:
            # 领域内显式声明的不可行候选：明确罚分，搜索继续。
            fitness[i] = -penalty_factor
            continue
        except Exception as exc:
            raise EvaluatorError(
                candidate_index=i,
                iteration=iteration,
                stage=stage,
                positions=positions,
                reason=f"目标函数抛出 {type(exc).__name__}: {exc}",
            ) from exc

        if not np.isfinite(value):
            raise EvaluatorError(
                candidate_index=i,
                iteration=iteration,
                stage=stage,
                positions=positions,
                reason=f"目标函数返回非有限值: {value!r}",
            )

        fitness[i] = value
        feasible[i] = True

    return BatchEvaluation(fitness=fitness, feasible=feasible)
