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
from .common import (
    BatchEvaluation,
    OptimizerConfigError,
    evaluate_batch,
    require_float,
    require_int,
    require_rate,
    validate_constructor_inputs,
)


@dataclass
class GAConfig:
    """遗传算法配置参数。

    Parameters
    ----------
    population_size : int
        种群大小（>=2）
    max_generations : int
        最大迭代代数（>=1）
    crossover_rate : float
        交叉概率，取值 [0, 1]
    mutation_rate : float
        变异概率，取值 [0, 1]
    mutation_strength : float
        变异强度（坐标标准差占场地范围的比例，>=0）
    elite_ratio : float
        精英保留比例，取值 [0, 1)；实际精英数必须小于种群大小
    tournament_size : int
        锦标赛选择的规模（1 <= tournament_size <= population_size）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径，>0）
    penalty_factor : float
        约束违反惩罚因子（>0）
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
        """在构造时完成所有范围与组合校验。"""
        require_int("population_size", self.population_size, minimum=2)
        require_int("max_generations", self.max_generations, minimum=1)

        require_rate("crossover_rate", self.crossover_rate)
        require_rate("mutation_rate", self.mutation_rate)
        require_float("mutation_strength", self.mutation_strength, non_negative=True)

        # elite_ratio=1 会令全部个体都是精英、无后代可产生；因此上界为开区间。
        require_rate("elite_ratio", self.elite_ratio)
        if self.elite_ratio >= 1.0:
            raise OptimizerConfigError("elite_ratio 必须小于 1.0")
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        if n_elite >= self.population_size:
            raise OptimizerConfigError(
                f"精英数量({n_elite})必须小于种群大小({self.population_size})，"
                f"请调小 elite_ratio"
            )

        require_int("tournament_size", self.tournament_size, minimum=1)
        if self.tournament_size > self.population_size:
            raise OptimizerConfigError(
                f"tournament_size({self.tournament_size})不能大于"
                f"population_size({self.population_size})"
            )

        require_float(
            "min_spacing_multiple", self.min_spacing_multiple, positive=True
        )
        require_float("penalty_factor", self.penalty_factor, positive=True)

        if self.seed is not None:
            require_int("seed", self.seed)


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
        找到最优解的代数
    convergence_history : list[float]
        每代最优适应度历史
    mean_history : list[float]
        每代平均适应度历史
    final_population : np.ndarray
        最终种群 (pop_size, N_turb*2)
    final_fitness : np.ndarray
        最终种群适应度 (pop_size,)
    success : bool
        是否至少有一个候选得到过有效的目标函数评估值。False 表示整个
        搜索期间所有候选都不可行，``best_fitness`` 只是惩罚值，不是有效最优解。
    message : str
        状态说明，``success=False`` 时给出原因。
    """

    best_positions: np.ndarray
    best_fitness: float
    best_generation: int
    convergence_history: list[float]
    mean_history: list[float]
    final_population: np.ndarray
    final_fitness: np.ndarray
    success: bool = True
    message: str = ""


class GeneticAlgorithm:
    """遗传算法机位优化器。

    优化目标：最大化年净发电量（等价于最小化尾流损失）。
    约束：最小间距、场地边界内。

    候选不可行（几何违反或目标函数显式声明）按罚分处理；目标函数抛出
    非领域异常或返回非有限值时，抛出
    :class:`~wind_farm_opt.optimization.common.EvaluatorError` 立即终止，
    绝不伪装成低适应度候选。
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
            适应度函数，输入位置数组 (N_turb, 2)，返回净AEP；
            可抛出 InfeasibleCandidate 显式声明候选不可行
        config : Optional[GAConfig]
            算法配置参数
        """
        self.config = config if config is not None else GAConfig()
        if not isinstance(self.config, GAConfig):
            raise OptimizerConfigError(
                f"config 必须为 GAConfig，实际类型为 {type(self.config).__name__}"
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

        self._best_positions = None
        self._best_fitness = -np.inf
        self._best_generation = 0
        self._found_feasible = False

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

    def _evaluate_population(
        self, population: np.ndarray, generation: int
    ) -> BatchEvaluation:
        """评估整个种群（几何/声明不可行记罚分，评估器故障立即终止）。"""
        batch = evaluate_batch(
            population,
            n_turbines=self.n_turbines,
            fitness_fn=self.fitness_fn,
            boundary=self.boundary,
            min_spacing=self.min_spacing,
            penalty_factor=self.config.penalty_factor,
            iteration=generation,
            stage="遗传算法初始种群" if generation == 0 else f"遗传算法第 {generation} 代",
        )
        self._found_feasible = self._found_feasible or bool(batch.feasible.any())
        return batch

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

    def optimize(self, verbose: bool = True) -> OptimizeResult:
        """执行优化。

        Parameters
        ----------
        verbose : bool
            是否打印进度信息

        Returns
        -------
        OptimizeResult
            优化结果；若所有候选始终不可行，结果 ``success=False``，
            不得当作有效最优解使用。

        Raises
        ------
        EvaluatorError
            目标函数抛出非领域异常或返回非有限值时立即终止。
        """
        pop_size = self.config.population_size
        max_gen = self.config.max_generations

        n_elite = max(1, int(pop_size * self.config.elite_ratio))

        self._found_feasible = False
        self._best_fitness = -np.inf
        self._best_generation = 0
        self.convergence_history = []
        self.mean_history = []

        if verbose:
            print(f"\n=== 遗传算法优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"种群大小: {pop_size}")
            print(f"最大代数: {max_gen}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"场地面积: {self.boundary.area / 1e6:.2f} km²")
            print("=" * 35)

        population = self._initialize_population(pop_size)
        batch = self._evaluate_population(population, generation=0)
        fitness = batch.fitness

        best_idx = int(np.argmax(fitness))
        self._best_fitness = float(fitness[best_idx])
        self._best_positions = population[best_idx].reshape(self.n_turbines, 2)
        self._best_generation = 0

        for gen in range(max_gen):
            self.convergence_history.append(float(self._best_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = population[elite_idx].copy()

            parents = self._tournament_selection(population, fitness, pop_size - n_elite)

            offspring = np.zeros((pop_size - n_elite, self.n_dim), dtype=np.float64)
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

            batch = self._evaluate_population(population, generation=gen + 1)
            fitness = batch.fitness

            current_best_idx = int(np.argmax(fitness))
            if fitness[current_best_idx] > self._best_fitness:
                self._best_fitness = float(fitness[current_best_idx])
                self._best_positions = population[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_generation = gen + 1

            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                n_feasible = int(batch.feasible.sum())
                feasibility_tag = f"可行 {n_feasible}/{pop_size}"
                print(
                    f"Gen {gen+1:3d} | "
                    f"Best: {self._best_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"{feasibility_tag} | "
                    f"Found@Gen {self._best_generation}"
                )

        success = self._found_feasible
        if success:
            message = ""
            if verbose:
                print("=" * 35)
                print(f"优化完成!")
                print(f"最优净AEP: {self._best_fitness/1e3:.2f} GWh")
                print(f"找到最优解的代数: {self._best_generation}")
        else:
            message = (
                "整个搜索期间没有任何候选通过目标函数的有效评估，"
                "best_fitness 仅为约束惩罚值，不是有效最优解；"
                "请检查约束设置或目标函数的 InfeasibleCandidate 声明。"
            )
            if verbose:
                print("=" * 35)
                print(f"优化异常结束（无有效候选）!")
                print(f"警告: {message}")

        return OptimizeResult(
            best_positions=self._best_positions.copy(),
            best_fitness=float(self._best_fitness),
            best_generation=self._best_generation,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=population.copy(),
            final_fitness=fitness.copy(),
            success=success,
            message=message,
        )
