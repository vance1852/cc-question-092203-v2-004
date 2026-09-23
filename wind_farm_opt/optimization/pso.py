"""粒子群优化器。"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)
from .evaluation import (
    InfeasibleCandidate,
    ObjectiveFailureError,
    OptimizeResult,
    evaluate_candidates,
    require_finite,
    require_int,
    require_optional_int,
    report_failure,
    validate_optimizer_inputs,
)

__all__ = [
    "PSOConfig",
    "ParticleSwarmOptimizer",
    "OptimizeResult",
    "InfeasibleCandidate",
    "ObjectiveFailureError",
]


@dataclass
class PSOConfig:
    """粒子群算法配置参数。

    Parameters
    ----------
    swarm_size : int
        粒子群大小（>= 2）
    max_iterations : int
        最大迭代次数（>= 1）
    inertia_weight : float
        惯性权重 w（有限，>= 0）
    cognitive_coeff : float
        认知系数 c1（有限，>= 0）
    social_coeff : float
        社会系数 c2（有限，>= 0）
    max_velocity : float
        最大速度（占场地范围的比例，> 0）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径，> 0）
    penalty_factor : float
        约束违反/声明不可行的惩罚因子（> 0）
    seed : Optional[int]
        随机种子
    """

    swarm_size: int = 40
    max_iterations: int = 150
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.49
    social_coeff: float = 1.49
    max_velocity: float = 0.2
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """在配置创建时完成范围与组合校验。"""
        require_int("swarm_size", self.swarm_size, min_value=2)
        require_int("max_iterations", self.max_iterations, min_value=1)
        require_finite("inertia_weight", self.inertia_weight, non_negative=True)
        require_finite("cognitive_coeff", self.cognitive_coeff, non_negative=True)
        require_finite("social_coeff", self.social_coeff, non_negative=True)
        require_finite("max_velocity", self.max_velocity, positive=True)
        require_finite(
            "min_spacing_multiple", self.min_spacing_multiple, positive=True
        )
        require_finite("penalty_factor", self.penalty_factor, positive=True)
        require_optional_int("seed", self.seed)


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。

    与 :class:`~wind_farm_opt.optimization.ga.GeneticAlgorithm` 遵循完全一
    致的候选评估与故障处理约定：

    - 几何违规或目标函数抛出 :class:`InfeasibleCandidate`：按明确惩罚处理，
      搜索继续；
    - 目标函数返回 NaN/Inf 或抛出其他异常：抛出
      :class:`ObjectiveFailureError` 立即终止，保留候选索引、布局与原始
      原因；
    - 整个运行没有可行候选时，结果的 ``best_feasible`` 为 ``False``，
      ``best_*`` 仅为诊断布局，不构成有效最优解。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[PSOConfig] = None,
    ) -> None:
        self.config = config if config is not None else PSOConfig()
        if not isinstance(self.config, PSOConfig):
            raise ValueError(
                f"config 必须是 PSOConfig，当前类型为 {type(self.config).__name__}"
            )

        self.n_turbines = n_turbines
        self.rotor_diameters = validate_optimizer_inputs(
            n_turbines, rotor_diameters, boundary, fitness_fn
        )
        self.boundary = boundary
        self.fitness_fn = fitness_fn

        self.rng = np.random.default_rng(self.config.seed)

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        self.vel_range = np.zeros(self.n_dim, dtype=np.float64)
        for i in range(self.n_dim):
            self.vel_range[i] = (
                self.x_range if i % 2 == 0 else self.y_range
            ) * self.config.max_velocity

        self.pos_bounds = np.zeros((self.n_dim, 2), dtype=np.float64)
        for i in range(self.n_dim):
            if i % 2 == 0:
                self.pos_bounds[i] = [boundary.x_min, boundary.x_max]
            else:
                self.pos_bounds[i] = [boundary.y_min, boundary.y_max]

        self._best_global_pos = None
        self._best_global_fitness = -np.inf
        self._best_iteration = -1
        self._has_feasible_best = False
        self._total_declared_infeasible = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    def _initialize_swarm(self, swarm_size: int) -> tuple[np.ndarray, np.ndarray]:
        """初始化粒子群。"""
        positions = np.zeros((swarm_size, self.n_dim), dtype=np.float64)
        velocities = np.zeros((swarm_size, self.n_dim), dtype=np.float64)

        for i in range(swarm_size):
            pos = self._generate_valid_layout()
            positions[i] = pos.flatten()
            velocities[i] = self.rng.uniform(
                -self.vel_range, self.vel_range, self.n_dim
            )

        return positions, velocities

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足约束的初始布局。"""
        max_attempts = 100

        for _ in range(max_attempts):
            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                valid, _ = check_min_spacing(positions, self.min_spacing)
                if valid:
                    return positions
            except RuntimeError:
                continue

            try:
                positions = self.boundary.sample_random_points(
                    self.n_turbines, self.rng, max_attempts=50
                )
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
                return positions
            except RuntimeError:
                continue

        raise RuntimeError("无法生成满足约束的初始布局")

    def _compute_penalty(self, positions_flat: np.ndarray) -> float:
        """计算约束违反惩罚。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

        penalty = 0.0

        inside = self.boundary.contains_all(positions)
        if not inside.all():
            n_violations = np.sum(~inside)
            penalty += n_violations * self.config.penalty_factor

        valid, violations = check_min_spacing(positions, self.min_spacing)
        if not valid:
            for i, j in violations:
                dist = np.linalg.norm(positions[i] - positions[j])
                penalty += (self.min_spacing - dist) * self.config.penalty_factor

        return penalty

    def _evaluate_particles(
        self, positions: np.ndarray, iteration: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """评估所有粒子的适应度。

        几何违规或目标函数声明不可行的粒子按明确惩罚处理；目标函数返回非
        有限值或抛出非领域异常时抛出 :class:`ObjectiveFailureError` 立即
        终止，索引、布局与原始原因保留在异常中。
        """
        penalties = np.array(
            [self._compute_penalty(pos) for pos in positions], dtype=np.float64
        )

        fitness, feasible_mask, n_declared = evaluate_candidates(
            self.fitness_fn,
            positions,
            penalties,
            self.n_turbines,
            self.config.penalty_factor,
            generation=iteration,
            algorithm="PSO",
        )
        self._total_declared_infeasible += n_declared
        return fitness, feasible_mask

    def _repair(self, positions_flat: np.ndarray) -> np.ndarray:
        """修复违反约束的粒子。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

        for i in range(self.n_turbines):
            if not self.boundary.contains_point(positions[i]):
                positions[i] = self.boundary.project_to_boundary(positions[i])

        valid, _ = check_min_spacing(positions, self.min_spacing)
        inside = self.boundary.contains_all(positions).all()

        if not (valid and inside):
            try:
                positions = enforce_min_spacing(
                    positions, self.min_spacing, self.boundary, self.rng
                )
            except RuntimeError:
                pass

        return positions.flatten()

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        Returns
        -------
        OptimizeResult
            优化结果。当整个运行中没有任何可行候选时，``best_feasible``
            为 ``False``，此时 ``best_*`` 仅为惩罚最小的诊断布局，不构成
            有效最优解。

        Raises
        ------
        ObjectiveFailureError
            目标函数返回非有限值或抛出非领域异常时立即抛出。
        """
        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"最大迭代: {max_iter}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            print("=" * 35)

        try:
            positions, velocities = self._initialize_swarm(swarm_size)
            fitness, feasible_mask = self._evaluate_particles(positions, 0)

            best_personal_pos = positions.copy()
            best_personal_fitness = fitness.copy()
            # 仅有真实可行的粒子才持有有效的个人历史最优。
            best_personal_feasible = feasible_mask.copy()

            self._update_global_best(positions, fitness, feasible_mask, 0)

            for iteration in range(max_iter):
                self.convergence_history.append(
                    float(self._best_global_fitness)
                    if self._has_feasible_best
                    else np.nan
                )
                self.mean_history.append(float(np.mean(fitness)))

                r1 = self.rng.random((swarm_size, self.n_dim))
                r2 = self.rng.random((swarm_size, self.n_dim))

                if self._has_feasible_best:
                    best_global_flat = self._best_global_pos.flatten()
                else:
                    # 尚无可行全局最优：以当前粒子自身位置为引导，避免
                    # None/惩罚布局把群体拖向无效区域。
                    best_global_flat = positions

                velocities = (
                    w * velocities
                    + c1 * r1 * (best_personal_pos - positions)
                    + c2 * r2 * (best_global_flat - positions)
                )

                velocities = np.clip(velocities, -self.vel_range, self.vel_range)

                positions = positions + velocities

                positions = np.clip(
                    positions,
                    self.pos_bounds[:, 0],
                    self.pos_bounds[:, 1],
                )

                for i in range(swarm_size):
                    positions[i] = self._repair(positions[i])

                fitness, feasible_mask = self._evaluate_particles(
                    positions, iteration + 1
                )

                # 个人最优：可行粒子间直接比较；此前不可行的粒子一旦可行即
                # 记录；不可行粒子之间按惩罚分更新，保证其个人历史不劣化。
                improved = np.zeros(swarm_size, dtype=bool)
                for i in range(swarm_size):
                    if feasible_mask[i] and best_personal_feasible[i]:
                        improved[i] = fitness[i] > best_personal_fitness[i]
                    elif feasible_mask[i] and not best_personal_feasible[i]:
                        improved[i] = True
                    elif not feasible_mask[i] and not best_personal_feasible[i]:
                        improved[i] = fitness[i] > best_personal_fitness[i]

                best_personal_pos[improved] = positions[improved].copy()
                best_personal_fitness[improved] = fitness[improved].copy()
                best_personal_feasible[improved] = feasible_mask[improved]

                self._update_global_best(
                    positions, fitness, feasible_mask, iteration + 1
                )

                if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                    n_feasible = int(np.count_nonzero(feasible_mask))
                    if self._has_feasible_best:
                        best_str = f"{self._best_global_fitness/1e3:8.2f} GWh"
                    else:
                        best_str = "     N/A"
                    print(
                        f"Iter {iteration+1:3d} | "
                        f"Best: {best_str} | "
                        f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                        f"可行粒子: {n_feasible}/{swarm_size} | "
                        f"Found@Iter {self._best_iteration}"
                    )
        except ObjectiveFailureError as error:
            if verbose:
                report_failure(error)
            raise

        if self._has_feasible_best:
            best_positions = self._best_global_pos.copy()
            best_fitness = float(self._best_global_fitness)
            best_iteration = self._best_iteration
        else:
            # 全部候选无效：保留惩罚最小的布局仅作诊断，明确标记为无效。
            diag_idx = int(np.argmax(fitness))
            best_positions = positions[diag_idx].reshape(self.n_turbines, 2).copy()
            best_fitness = float(fitness[diag_idx])
            best_iteration = -1
            if verbose:
                print("=" * 35)
                print(
                    "警告: 整个优化过程中没有任何候选通过目标函数评估，"
                    "不存在有效最优解！"
                )
                print(
                    f"（返回的 best_* 仅为惩罚最小的诊断布局，"
                    f"惩罚分: {best_fitness:.1f}）"
                )

        if verbose and self._has_feasible_best:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_global_fitness/1e3:.2f} GWh")
            print(f"找到最优解的迭代: {self._best_iteration}")

        return OptimizeResult(
            best_positions=best_positions,
            best_fitness=best_fitness,
            best_generation=best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
            best_feasible=self._has_feasible_best,
        )

    def _update_global_best(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        feasible_mask: np.ndarray,
        iteration: int,
    ) -> None:
        """仅在真实可行粒子中更新全局最优。"""
        if not np.any(feasible_mask):
            return

        feasible_idx = np.flatnonzero(feasible_mask)
        idx = feasible_idx[np.argmax(fitness[feasible_idx])]
        if not self._has_feasible_best or fitness[idx] > self._best_global_fitness:
            self._best_global_fitness = float(fitness[idx])
            self._best_global_pos = positions[idx].reshape(
                self.n_turbines, 2
            ).copy()
            self._best_iteration = iteration
            self._has_feasible_best = True
