"""GA/PSO 共用的候选评估、故障处理与参数校验。

本模块明确区分两类性质完全不同的情况：

- **候选不可行**：布局违反几何约束（边界、最小间距），或目标函数通过抛出
  :class:`InfeasibleCandidate` 主动声明候选不可行。此时候选按明确的惩罚值
  计入适应度，搜索继续进行。
- **评估器故障**：目标函数返回 NaN/Inf 等非有限值，或抛出
  :class:`InfeasibleCandidate` 以外的异常（典型如维度不匹配导致的
  ``ValueError``）。这属于评估器自身的缺陷，立即终止搜索，并通过
  :class:`ObjectiveFailureError` 保留故障候选的索引、布局和原始原因，
  绝不允许把故障伪装成一个"大负分"候选继续运行。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


class InfeasibleCandidate(Exception):
    """目标函数主动声明某个候选不可行的领域异常。

    目标函数遇到领域层面无法评估的布局（例如机位与既有设施冲突）时抛出本
    异常；优化器会把该候选按 ``-penalty_factor`` 计入适应度并继续搜索，
    区别于评估器自身故障。
    """


class ObjectiveFailureError(RuntimeError):
    """目标函数自身故障：返回非有限值，或抛出非领域异常。

    Attributes
    ----------
    kind : str
        ``"non_finite"`` 表示返回了 NaN/Inf；``"exception"`` 表示抛出了
        :class:`InfeasibleCandidate` 以外的异常。
    candidate_index : int
        故障候选在本批评估中的索引（从 0 开始）。
    generation : int
        发生故障的代数/迭代次数，``0`` 表示初始种群/粒子群评估。
    positions : np.ndarray
        故障候选的布局副本，形状为 ``(n_turbines, 2)``。
    algorithm : Optional[str]
        触发故障的算法名称。
    value : Optional[float]
        当 ``kind == "non_finite"`` 时目标函数返回的原始值。
    exception : Optional[BaseException]
        当 ``kind == "exception"`` 时目标函数抛出的原始异常（同时作为
        ``__cause__`` 保留完整回溯链）。
    """

    NON_FINITE = "non_finite"
    EXCEPTION = "exception"

    def __init__(
        self,
        *,
        kind: str,
        candidate_index: int,
        generation: int,
        positions: np.ndarray,
        algorithm: Optional[str] = None,
        value: Optional[float] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        self.kind = kind
        self.candidate_index = int(candidate_index)
        self.generation = int(generation)
        self.positions = np.asarray(positions, dtype=np.float64).copy()
        self.algorithm = algorithm
        self.value = value
        self.exception = exception
        super().__init__(self._format_message())

    @property
    def description(self) -> str:
        """故障原因的人类可读描述（保留原始信息）。"""
        if self.kind == self.NON_FINITE:
            return f"目标函数返回非有限值: {self.value!r}"
        exc = self.exception
        return f"目标函数抛出非领域异常 {type(exc).__name__}: {exc}"

    def _format_message(self) -> str:
        algo = self.algorithm or "优化器"
        stage = "初始种群/粒子群" if self.generation == 0 else f"第 {self.generation} 代/次迭代"
        return (
            f"{algo}目标函数评估失败并已终止（{stage}，"
            f"批内候选索引 {self.candidate_index}）: {self.description}"
        )


def report_failure(error: ObjectiveFailureError) -> None:
    """将评估器故障以醒目方式输出到 stderr（GA/PSO 保持一致）。"""
    lines = [
        "",
        "=" * 60,
        "  优化已提前终止：目标函数发生评估器故障（非候选不可行）",
        "=" * 60,
        f"  算法:           {error.algorithm}",
        f"  候选索引:       {error.candidate_index}",
        f"  代/迭代(0=初始): {error.generation}",
        f"  原始原因:       {error.description}",
        "  说明:           该故障未被当作惩罚，搜索立即终止；",
        "                   请检查目标函数的维度约定与数值有效性。",
        "=" * 60,
    ]
    print("\n".join(lines), file=sys.stderr)


@dataclass
class OptimizeResult:
    """优化结果。

    Parameters
    ----------
    best_positions : np.ndarray
        最优风机位置 (N_turb, 2)
    best_fitness : float
        最优适应度（净AEP，MWh/year）
    best_generation : int
        找到最优解的代数/迭代
    convergence_history : list[float]
        每代/每次迭代最优适应度历史
    mean_history : list[float]
        每代/每次迭代平均适应度历史
    final_population : np.ndarray
        最终种群/粒子群 (pop_size, N_turb*2)
    final_fitness : np.ndarray
        最终种群/粒子群适应度 (pop_size,)
    best_feasible : bool
        ``best_*`` 是否来自通过目标函数评估的可行候选。``False`` 表示整个
        运行过程中没有任何可行候选（全部为惩罚分），此时 ``best_*`` 不构成
        有效最优解，调用方不得将其作为优化成功的结果使用。
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list[float]
    mean_history: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    best_feasible: bool = True


def evaluate_candidates(
    fitness_fn: Callable[[np.ndarray], Any],
    population: np.ndarray,
    penalties: np.ndarray,
    n_turbines: int,
    penalty_factor: float,
    generation: int,
    algorithm: str = "优化器",
) -> tuple[np.ndarray, np.ndarray, int]:
    """批量评估候选（GA/PSO 共用）。

    对每个候选：

    1. ``penalties[i] > 0``：几何约束违规，适应度取 ``-penalties[i]``，
       不调用目标函数，搜索继续；
    2. 目标函数抛出 :class:`InfeasibleCandidate`：按明确惩罚
       ``-penalty_factor`` 处理，搜索继续；
    3. 目标函数抛出其他异常：立即抛出 :class:`ObjectiveFailureError`，
       保留候选索引、布局与原始异常；
    4. 目标函数返回非有限值（NaN/+-Inf）或无法转为浮点的对象：同样立即
       抛出 :class:`ObjectiveFailureError`；
    5. 正常返回有限值：作为适应度，候选标记为可行。

    Parameters
    ----------
    fitness_fn : callable
        目标函数，输入 ``(n_turbines, 2)`` 布局，返回标量适应度。
    population : np.ndarray
        展平的候选数组，形状 ``(batch_size, n_turbines*2)``。
    penalties : np.ndarray
        每个候选预先计算好的几何约束惩罚（非负）。
    n_turbines : int
        风机台数。
    penalty_factor : float
        目标函数声明不可行时使用的明确惩罚系数。
    generation : int
        当前代/迭代编号（初始批次为 0），用于故障定位。
    algorithm : str
        算法名称，用于故障信息。

    Returns
    -------
    fitness : np.ndarray
        适应度数组 ``(batch_size,)``。
    feasible_mask : np.ndarray
        布尔数组，仅当候选真实通过目标函数评估时为 ``True``。
    n_declared_infeasible : int
        本批中被目标函数以 :class:`InfeasibleCandidate` 声明为不可行的
        候选数量。
    """
    batch_size = population.shape[0]
    fitness = np.zeros(batch_size, dtype=np.float64)
    feasible_mask = np.zeros(batch_size, dtype=bool)
    n_declared_infeasible = 0

    for i in range(batch_size):
        if penalties[i] > 0.0:
            # 几何约束违规：明确惩罚，继续搜索。
            fitness[i] = -float(penalties[i])
            continue

        positions = population[i].reshape(n_turbines, 2)

        try:
            raw_value = fitness_fn(positions)
            value = float(raw_value)
        except InfeasibleCandidate:
            # 领域层面的不可行声明：明确惩罚，继续搜索。
            fitness[i] = -float(penalty_factor)
            n_declared_infeasible += 1
            continue
        except Exception as exc:
            # 非领域异常：评估器故障，立即终止并保留现场。
            raise ObjectiveFailureError(
                kind=ObjectiveFailureError.EXCEPTION,
                candidate_index=i,
                generation=generation,
                positions=positions,
                algorithm=algorithm,
                exception=exc,
            ) from exc

        if not math.isfinite(value):
            raise ObjectiveFailureError(
                kind=ObjectiveFailureError.NON_FINITE,
                candidate_index=i,
                generation=generation,
                positions=positions,
                algorithm=algorithm,
                value=value,
            )

        fitness[i] = value
        feasible_mask[i] = True

    return fitness, feasible_mask, n_declared_infeasible


# ---------------------------------------------------------------------------
# 配置与构造参数校验
# ---------------------------------------------------------------------------


def require_int(
    name: str,
    value: Any,
    *,
    min_value: Optional[int] = None,
) -> int:
    """校验参数必须是整数（拒绝 bool），并可限定下界。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} 必须是整数，当前类型为 {type(value).__name__}: {value!r}"
        )
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} 必须 >= {min_value}，当前为 {value}")
    return value


def require_finite(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """校验参数必须是有限数值，并可限定正数/非负数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} 必须是数值，当前类型为 {type(value).__name__}: {value!r}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值，当前为 {value}")
    if positive and value <= 0.0:
        raise ValueError(f"{name} 必须 > 0，当前为 {value}")
    if non_negative and value < 0.0:
        raise ValueError(f"{name} 必须 >= 0，当前为 {value}")
    return value


def require_unit_interval(name: str, value: Any) -> float:
    """校验参数位于闭区间 [0, 1]。"""
    value = require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]，当前为 {value}")
    return value


def require_optional_int(name: str, value: Any) -> Optional[int]:
    """校验参数为整数或 None。"""
    if value is None:
        return None
    return require_int(name, value)


def validate_optimizer_inputs(
    n_turbines: Any,
    rotor_diameters: Any,
    boundary: Any,
    fitness_fn: Any,
) -> np.ndarray:
    """校验优化器构造参数（范围与组合关系），返回标准化的直径数组。

    在创建优化器时立即失败，避免非法参数拖到搜索深处才以难懂的数组错误
    暴露。
    """
    require_int("n_turbines", n_turbines, min_value=1)

    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    if diameters.ndim != 1:
        raise ValueError(
            f"rotor_diameters 必须是一维数组，当前形状为 {diameters.shape}"
        )
    if diameters.shape[0] != n_turbines:
        raise ValueError(
            f"rotor_diameters 长度 ({diameters.shape[0]}) 必须等于 "
            f"n_turbines ({n_turbines})"
        )
    if not np.all(np.isfinite(diameters)):
        raise ValueError("rotor_diameters 包含非有限值")
    if np.any(diameters <= 0.0):
        raise ValueError("rotor_diameters 必须全部为正数")

    if not callable(fitness_fn):
        raise ValueError(
            f"fitness_fn 必须可调用，当前类型为 {type(fitness_fn).__name__}"
        )

    x_range = boundary.x_max - boundary.x_min
    y_range = boundary.y_max - boundary.y_min
    if not (math.isfinite(x_range) and math.isfinite(y_range)):
        raise ValueError("场地边界范围包含非有限值")
    if x_range <= 0.0 or y_range <= 0.0:
        raise ValueError(
            f"场地边界必须具有正的宽和高，当前 x 范围 {x_range}、y 范围 {y_range}"
        )

    return diameters
