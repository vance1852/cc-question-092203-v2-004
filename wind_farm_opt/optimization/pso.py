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
from .common import (
    BatchEvaluation,
    OptimizerConfigError,
    evaluate_batch,
    require_float,
    require_int,
    validate_constructor_inputs,
)


@dataclass
class PSOConfig:
    """粒子群算法配置参数。

    Parameters
    ----------
    swarm_size : int
        粒子群大小（>=1）
    max_iterations : int
        最大迭代次数（>=1）
    inertia_weight : float
        惯性权重 w（有限值）
    cognitive_coeff : float
        认知系数 c1（>=0）
    social_coeff : float
        社会系数 c2（>=0）
    max_velocity : float
        最大速度（占场地范围的比例，>=0）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径，>0）
    penalty_factor : float
        约束违反惩罚因子（>0）
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
        """在构造时完成所有范围与组合校验。"""
        require_int("swarm_size", self.swarm_size, minimum=1)
        require_int("max_iterations", self.max_iterations, minimum=1)

        require_float("inertia_weight", self.inertia_weight)
        require_float("cognitive_coeff", self.cognitive_coeff, non_negative=True)
        require_float("social_coeff", self.social_coeff, non_negative=True)
        require_float("max_velocity", self.max_velocity, non_negative=True)

        # c1 与 c2 同时为 0 时粒子只会惯性滑行、完全不学习，属于退化配置。
        if self.cognitive_coeff == 0.0 and self.social_coeff == 0.0:
            raise OptimizerConfigError(
                "cognitive_coeff 与 social_coeff 不能同时为 0，粒子将无法更新"
            )

        require_float(
            "min_spacing_multiple", self.min_spacing_multiple, positive=True
        )
        require_float("penalty_factor", self.penalty_factor, positive=True)

        if self.seed is not None:
            require_int("seed", self.seed)


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。

    候选不可行（几何违反或目标函数显式声明）按罚分处理；目标函数抛出
    非领域异常或返回非有限值时，抛出
    :class:`~wind_farm_opt.optimization.common.EvaluatorError` 立即终止，
    与遗传算法保持完全一致的语义。
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
            raise OptimizerConfigError(
                f"config 必须为 PSOConfig，实际类型为 {type(self.config).__name__}"
            )

        self.n_turbines = n_turbines
        self.rotor_diameters = validate_constructor_inputs(
            n_turbines,
            rotor_diameters,
            boundary,
            fitness_fn,
            self.config.min_spacing_multiple,
            self.config.penalty_factor,
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
        self._best_iteration = 0
        self._found_feasible = False

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

    def _evaluate_particles(
        self, positions: np.ndarray, iteration: int
    ) -> BatchEvaluation:
        """评估所有粒子（几何/声明不可行记罚分，评估器故障立即终止）。"""
        batch = evaluate_batch(
            positions,
            n_turbines=self.n_turbines,
            fitness_fn=self.fitness_fn,
            boundary=self.boundary,
            min_spacing=self.min_spacing,
            penalty_factor=self.config.penalty_factor,
            iteration=iteration,
            stage="粒子群初始群" if iteration == 0 else f"粒子群第 {iteration} 次迭代",
        )
        self._found_feasible = self._found_feasible or bool(batch.feasible.any())
        return batch

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

    def optimize(self, verbose: bool = True) -> "OptimizeResult":
        """执行优化。

        Returns
        -------
        OptimizeResult
            优化结果；若所有粒子始终不可行，结果 ``success=False``，
            不得当作有效最优解使用。

        Raises
        ------
        EvaluatorError
            目标函数抛出非领域异常或返回非有限值时立即终止。
        """
        from .ga import OptimizeResult

        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        self._found_feasible = False
        self._best_global_fitness = -np.inf
        self._best_iteration = 0
        self.convergence_history = []
        self.mean_history = []

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"最大迭代: {max_iter}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            print("=" * 35)

        positions, velocities = self._initialize_swarm(swarm_size)
        batch = self._evaluate_particles(positions, iteration=0)
        fitness = batch.fitness

        best_personal_pos = positions.copy()
        best_personal_fitness = fitness.copy()

        best_global_idx = int(np.argmax(fitness))
        self._best_global_pos = positions[best_global_idx].reshape(self.n_turbines, 2).copy()
        self._best_global_fitness = float(fitness[best_global_idx])
        self._best_iteration = 0

        for iteration in range(max_iter):
            self.convergence_history.append(float(self._best_global_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            r1 = self.rng.random((swarm_size, self.n_dim))
            r2 = self.rng.random((swarm_size, self.n_dim))

            best_global_flat = self._best_global_pos.flatten()

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

            batch = self._evaluate_particles(positions, iteration=iteration + 1)
            fitness = batch.fitness

            improved_mask = fitness > best_personal_fitness
            best_personal_pos[improved_mask] = positions[improved_mask].copy()
            best_personal_fitness[improved_mask] = fitness[improved_mask].copy()

            current_best_idx = int(np.argmax(fitness))
            if fitness[current_best_idx] > self._best_global_fitness:
                self._best_global_fitness = float(fitness[current_best_idx])
                self._best_global_pos = positions[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_iteration = iteration + 1

            if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                n_feasible = int(batch.feasible.sum())
                feasibility_tag = f"可行 {n_feasible}/{swarm_size}"
                print(
                    f"Iter {iteration+1:3d} | "
                    f"Best: {self._best_global_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"{feasibility_tag} | "
                    f"Found@Iter {self._best_iteration}"
                )

        success = self._found_feasible
        if success:
            message = ""
            if verbose:
                print("=" * 35)
                print(f"优化完成!")
                print(f"最优净AEP: {self._best_global_fitness/1e3:.2f} GWh")
                print(f"找到最优解的迭代: {self._best_iteration}")
        else:
            message = (
                "整个搜索期间没有任何粒子通过目标函数的有效评估，"
                "best_fitness 仅为约束惩罚值，不是有效最优解；"
                "请检查约束设置或目标函数的 InfeasibleCandidate 声明。"
            )
            if verbose:
                print("=" * 35)
                print(f"优化异常结束（无有效粒子）!")
                print(f"警告: {message}")

        return OptimizeResult(
            best_positions=self._best_global_pos.copy(),
            best_fitness=float(self._best_global_fitness),
            best_generation=self._best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
            success=success,
            message=message,
        )
