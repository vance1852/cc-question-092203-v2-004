"""遗传算法优化器。"""

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
    require_unit_interval,
    report_failure,
    validate_optimizer_inputs,
)

# 向后兼容：OptimizeResult / 异常类型统一在 evaluation 中定义。
__all__ = [
    "GAConfig",
    "GeneticAlgorithm",
    "OptimizeResult",
    "InfeasibleCandidate",
    "ObjectiveFailureError",
]


@dataclass
class GAConfig:
    """遗传算法配置参数。

    Parameters
    ----------
    population_size : int
        种群大小（>= 2）
    max_generations : int
        最大迭代代数（>= 1）
    crossover_rate : float
        交叉概率，取值 [0, 1]
    mutation_rate : float
        变异概率，取值 [0, 1]
    mutation_strength : float
        变异强度（坐标标准差占场地范围的比例，>= 0）
    elite_ratio : float
        精英保留比例，取值 [0, 1)；且保留精英数必须严格小于种群大小
    tournament_size : int
        锦标赛选择的规模（1 <= tournament_size <= population_size）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径，> 0）
    penalty_factor : float
        约束违反/声明不可行的惩罚因子（> 0）
    seed : Optional[int]
        随机种子
    """

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    elite_ratio: float = 0.1
    tournament_size: int = 3
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """在配置创建时完成范围与组合校验。"""
        require_int("population_size", self.population_size, min_value=2)
        require_int("max_generations", self.max_generations, min_value=1)
        require_unit_interval("crossover_rate", self.crossover_rate)
        require_unit_interval("mutation_rate", self.mutation_rate)
        require_finite("mutation_strength", self.mutation_strength, non_negative=True)
        require_unit_interval("elite_ratio", self.elite_ratio)
        require_int("tournament_size", self.tournament_size, min_value=1)
        require_finite(
            "min_spacing_multiple", self.min_spacing_multiple, positive=True
        )
        require_finite("penalty_factor", self.penalty_factor, positive=True)
        require_optional_int("seed", self.seed)

        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) 不能大于 "
                f"population_size ({self.population_size})"
            )

        n_elite = max(1, int(self.population_size * self.elite_ratio))
        if n_elite >= self.population_size:
            raise ValueError(
                f"精英保留数量 ({n_elite}) 必须严格小于种群大小 "
                f"({self.population_size})，请调小 elite_ratio"
            )


class GeneticAlgorithm:
    """遗传算法机位优化器。

    优化目标：最大化年净发电量（等价于最小化尾流损失）。
    约束：最小间距、场地边界内。
    """

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[GAConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_turbines : int
            风机台数
        rotor_diameters : np.ndarray
            每台风机的转子直径
        boundary : SiteBoundary
            场地边界
        fitness_fn : Callable[[np.ndarray], float]
            适应度函数，输入位置数组 (N_turb, 2)，返回净AEP。
            抛出 :class:`InfeasibleCandidate` 表示声明候选不可行（按惩罚
            处理）；返回 NaN/Inf 或抛出其他异常视为评估器故障并立即终止。
        config : Optional[GAConfig]
            算法配置参数
        """
        self.config = config if config is not None else GAConfig()
        if not isinstance(self.config, GAConfig):
            raise ValueError(
                f"config 必须是 GAConfig，当前类型为 {type(self.config).__name__}"
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

        self._best_positions = None
        self._best_fitness = -np.inf
        self._best_generation = -1
        self._has_feasible_best = False
        self._total_declared_infeasible = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        """初始化种群。

        每个个体是展平的位置向量：[x1, y1, x2, y2, ..., xn, yn]
        """
        population = np.zeros((pop_size, self.n_dim), dtype=np.float64)

        for i in range(pop_size):
            positions = self._generate_valid_layout()
            population[i] = positions.flatten()

        return population

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

    def _evaluate_population(
        self, population: np.ndarray, generation: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """评估整个种群的适应度（带惩罚）。

        几何违规或目标函数声明不可行的候选按明确惩罚处理；目标函数返回非
        有限值或抛出非领域异常时抛出 :class:`ObjectiveFailureError` 立即
        终止，索引、布局与原始原因保留在异常中。
        """
        penalties = np.array(
            [self._compute_penalty(ind) for ind in population], dtype=np.float64
        )

        fitness, feasible_mask, n_declared = evaluate_candidates(
            self.fitness_fn,
            population,
            penalties,
            self.n_turbines,
            self.config.penalty_factor,
            generation=generation,
            algorithm="GA",
        )
        self._total_declared_infeasible += n_declared
        return fitness, feasible_mask

    def _tournament_selection(
        self, population: np.ndarray, fitness: np.ndarray, n_select: int
    ) -> np.ndarray:
        """锦标赛选择。"""
        pop_size = population.shape[0]
        selected = np.zeros((n_select, self.n_dim), dtype=np.float64)

        for i in range(n_select):
            candidates = self.rng.integers(0, pop_size, size=self.config.tournament_size)
            best_idx = candidates[np.argmax(fitness[candidates])]
            selected[i] = population[best_idx]

        return selected

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """均匀交叉。"""
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy()

        mask = self.rng.integers(0, 2, size=self.n_dim, dtype=bool)
        child = np.where(mask, parent1, parent2)

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """高斯变异。"""
        mutated = individual.copy()

        for i in range(self.n_dim):
            if self.rng.random() < self.config.mutation_rate:
                range_sigma = (
                    self.x_range if i % 2 == 0 else self.y_range
                ) * self.config.mutation_strength
                mutated[i] += self.rng.normal(0.0, range_sigma)

        return mutated

    def _repair(self, individual: np.ndarray) -> np.ndarray:
        """修复违反约束的个体。"""
        positions = individual.reshape(self.n_turbines, 2)

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

    def _update_best(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        feasible_mask: np.ndarray,
        generation: int,
    ) -> None:
        """仅在真实可行候选中更新已记录的最优解。"""
        if not np.any(feasible_mask):
            return

        feasible_idx = np.flatnonzero(feasible_mask)
        idx = feasible_idx[np.argmax(fitness[feasible_idx])]
        if not self._has_feasible_best or fitness[idx] > self._best_fitness:
            self._best_fitness = float(fitness[idx])
            self._best_positions = population[idx].reshape(
                self.n_turbines, 2
            ).copy()
            self._best_generation = generation
            self._has_feasible_best = True

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息

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
        pop_size = self.config.population_size
        max_gen = self.config.max_generations

        n_elite = max(1, int(pop_size * self.config.elite_ratio))

        if verbose:
            print(f"\n=== 遗传算法优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"最大代数: {max_gen}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"场地面积: {self.boundary.area / 1e6:.2f} km²")
            print("=" * 35)

        try:
            population = self._initialize_population(pop_size)
            fitness, feasible_mask = self._evaluate_population(population, 0)
            self._update_best(population, fitness, feasible_mask, 0)

            for gen in range(max_gen):
                self.convergence_history.append(
                    float(self._best_fitness) if self._has_feasible_best else np.nan
                )
                self.mean_history.append(float(np.mean(fitness)))

                elite_idx = np.argsort(fitness)[-n_elite:]
                elites = population[elite_idx].copy()

                parents = self._tournament_selection(
                    population, fitness, pop_size - n_elite
                )

                offspring = np.zeros(
                    (pop_size - n_elite, self.n_dim), dtype=np.float64
                )
                for i in range(0, pop_size - n_elite, 2):
                    p1 = parents[i]
                    p2 = parents[(i + 1) % (pop_size - n_elite)]
                    c1 = self._crossover(p1, p2)
                    c2 = self._crossover(p2, p1)
                    offspring[i] = self._mutate(c1)
                    if i + 1 < pop_size - n_elite:
                        offspring[i + 1] = self._mutate(c2)

                for i in range(len(offspring)):
                    offspring[i] = self._repair(offspring[i])

                population[:n_elite] = elites
                population[n_elite:] = offspring

                fitness, feasible_mask = self._evaluate_population(
                    population, gen + 1
                )
                self._update_best(population, fitness, feasible_mask, gen + 1)

                if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                    n_feasible = int(np.count_nonzero(feasible_mask))
                    if self._has_feasible_best:
                        best_str = f"{self._best_fitness/1e3:8.2f} GWh"
                    else:
                        best_str = "     N/A"
                    print(
                        f"Gen {gen+1:3d} | "
                        f"Best: {best_str} | "
                        f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                        f"可行候选: {n_feasible}/{pop_size} | "
                        f"Found@Gen {self._best_generation}"
                    )
        except ObjectiveFailureError as error:
            if verbose:
                report_failure(error)
            raise

        if self._has_feasible_best:
            best_positions = self._best_positions.copy()
            best_fitness = float(self._best_fitness)
            best_generation = self._best_generation
        else:
            # 全部候选无效：保留惩罚最小的布局仅作诊断，明确标记为无效。
            diag_idx = int(np.argmax(fitness))
            best_positions = population[diag_idx].reshape(self.n_turbines, 2).copy()
            best_fitness = float(fitness[diag_idx])
            best_generation = -1
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
            print(f"最优净AEP: {self._best_fitness/1e3:.2f} GWh")
            print(f"找到最优解的代数: {self._best_generation}")

        return OptimizeResult(
            best_positions=best_positions,
            best_fitness=best_fitness,
            best_generation=best_generation,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            best_feasible=self._has_feasible_best,
        )
