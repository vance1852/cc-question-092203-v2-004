"""GA 与 PSO 共同行为测试（参数化两种算法）。

覆盖四类关键差异：

1. 首个候选即失败（非领域异常 / 非有限返回值）：立即终止，保留索引、
   布局与原始原因；
2. 搜索中途失败：在对应代/迭代终止，而不是被伪装成大负分继续；
3. 合法惩罚：几何违规与 ``InfeasibleCandidate`` 声明均按明确惩罚继续；
4. 全部候选无效：运行正常结束但 ``best_feasible=False``，不构成最优解。
"""

import math

import numpy as np
import pytest

from conftest import (
    N_TURBINES,
    OPTIMIZER_CASES,
    build_optimizer,
)
from wind_farm_opt.optimization.evaluation import (
    InfeasibleCandidate,
    ObjectiveFailureError,
    evaluate_candidates,
)
from wind_farm_opt.optimization.ga import GAConfig
from wind_farm_opt.optimization.pso import PSOConfig


# ---------------------------------------------------------------------------
# 公共测试体
# ---------------------------------------------------------------------------


def healthy_objective(positions):
    """一个正常的目标函数：坐标和越大分越高，始终有限。"""
    return float(1e5 + positions.sum())


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_successful_run_returns_feasible_best(case, boundary, rotor_diameters):
    """正常目标函数下两种算法都应返回有效最优解。"""
    optimizer_cls, config_factory, batch_size = case
    optimizer = build_optimizer(
        optimizer_cls, config_factory(), boundary, rotor_diameters, healthy_objective
    )

    result = optimizer.optimize(verbose=False)

    assert result.best_feasible is True
    assert math.isfinite(result.best_fitness)
    assert result.best_positions.shape == (N_TURBINES, 2)
    assert result.best_generation >= 0
    assert len(result.convergence_history) == len(result.mean_history)
    # 历史中不应出现 NaN（始终存在可行最优）。
    assert all(math.isfinite(v) for v in result.convergence_history)


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_first_candidate_exception_aborts_immediately(
    case, boundary, rotor_diameters
):
    """首个候选即抛非领域异常：立即终止，且现场完整保留。"""
    optimizer_cls, config_factory, batch_size = case
    received = []

    def broken_objective(positions):
        received.append(positions.copy())
        # 模拟新接入评估函数的典型故障：维度不匹配。
        raise ValueError(f"维度错误: 期望 (?, 3)，实际 {positions.shape}")

    optimizer = build_optimizer(
        optimizer_cls, config_factory(), boundary, rotor_diameters, broken_objective
    )

    with pytest.raises(ObjectiveFailureError) as exc_info:
        optimizer.optimize(verbose=False)

    error = exc_info.value
    assert error.kind == ObjectiveFailureError.EXCEPTION
    # 首批评估即失败，代/迭代编号必须为 0。
    assert error.generation == 0
    assert error.candidate_index == 0
    # 布局按候选索引原样保留。
    assert error.positions.shape == (N_TURBINES, 2)
    np.testing.assert_allclose(error.positions, received[0])
    # 原始异常类型与信息保留（__cause__ 链不被吞掉）。
    assert isinstance(error.exception, ValueError)
    assert "维度错误" in str(error.exception)
    assert exc_info.value.__cause__ is error.exception
    assert "候选索引 0" in str(error)
    # 只评估了第一个候选就终止，而不是整群被打成大负分。
    assert len(received) == 1


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_first_candidate_non_finite_aborts_immediately(
    case, boundary, rotor_diameters
):
    """首个候选返回 NaN：按评估器故障立即终止，而非当作 -inf 候选。"""
    optimizer_cls, config_factory, batch_size = case

    def nan_objective(positions):
        return np.nan

    optimizer = build_optimizer(
        optimizer_cls, config_factory(), boundary, rotor_diameters, nan_objective
    )

    with pytest.raises(ObjectiveFailureError) as exc_info:
        optimizer.optimize(verbose=False)

    error = exc_info.value
    assert error.kind == ObjectiveFailureError.NON_FINITE
    assert error.generation == 0
    assert error.candidate_index == 0
    assert math.isnan(error.value)
    assert error.positions.shape == (N_TURBINES, 2)


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_mid_run_failure_aborts_at_correct_generation(
    case, boundary, rotor_diameters
):
    """首批全部正常、第二批开始失败：必须在第 1 代/迭代终止。

    使用零变异(GA)/零速度系数(PSO)的确定性配置，使新一代候选必然几何可
    行，从而故障候选索引确定：GA 有 1 个精英先行评估，PSO 从索引 0 开始。
    """
    optimizer_cls, config_factory, batch_size = case
    calls = {"n": 0}

    def flaky_objective(positions):
        idx = calls["n"]
        calls["n"] += 1
        # 初始批次（batch_size 个候选）全部正常，之后立刻故障。
        if idx >= batch_size:
            raise RuntimeError(f"评估器在第 {idx} 次调用时内部崩溃")
        return healthy_objective(positions)

    if optimizer_cls.__name__ == "GeneticAlgorithm":
        config = config_factory(crossover_rate=0.0, mutation_rate=0.0)
    else:
        config = config_factory(
            inertia_weight=0.0, cognitive_coeff=0.0, social_coeff=0.0
        )

    optimizer = build_optimizer(
        optimizer_cls, config, boundary, rotor_diameters, flaky_objective
    )

    with pytest.raises(ObjectiveFailureError) as exc_info:
        optimizer.optimize(verbose=False)

    error = exc_info.value
    assert error.kind == ObjectiveFailureError.EXCEPTION
    assert error.generation == 1
    assert error.candidate_index == 0
    assert isinstance(error.exception, RuntimeError)
    assert "内部崩溃" in str(error.exception)
    # 初始批次 + 新一代第一个候选后即终止，没有跑满全部迭代。
    assert calls["n"] == batch_size + 1


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_declared_infeasible_gets_explicit_penalty_and_continues(
    case, boundary, rotor_diameters
):
    """目标函数以领域异常声明部分候选不可行：按惩罚继续，不终止。"""
    optimizer_cls, config_factory, batch_size = case

    def partially_infeasible_objective(positions):
        # 以第一个机位的 x 坐标人为划分：一部分布局可行、一部分声明不可行。
        if positions[0, 0] < 0.0:
            raise InfeasibleCandidate("该机位与既有设施冲突")
        return healthy_objective(positions)

    config = config_factory()
    optimizer = build_optimizer(
        optimizer_cls, config, boundary, rotor_diameters,
        partially_infeasible_objective,
    )

    result = optimizer.optimize(verbose=False)

    # 存在可行候选：结果必须来自可行候选，而不是惩罚分。
    assert result.best_feasible is True
    assert result.best_fitness > 0.0
    assert result.best_positions[0, 0] >= 0.0
    # 运行过程中至少出现过一次声明不可行（按明确惩罚处理）。
    assert optimizer._total_declared_infeasible > 0


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_all_candidates_declared_infeasible(case, boundary, rotor_diameters):
    """全部候选被声明不可行：正常结束但明确标记无有效最优解。"""
    optimizer_cls, config_factory, batch_size = case

    def always_infeasible(positions):
        raise InfeasibleCandidate("领域规则拒绝该布局")

    config = config_factory()
    optimizer = build_optimizer(
        optimizer_cls, config, boundary, rotor_diameters, always_infeasible
    )

    result = optimizer.optimize(verbose=False)

    assert result.best_feasible is False
    assert result.best_generation == -1
    assert result.best_fitness == -config.penalty_factor
    assert np.all(result.final_fitness == -config.penalty_factor)
    assert result.best_positions.shape == (N_TURBINES, 2)
    # 收敛历史在无可行最优时为 NaN，绘图/下游可据此识别。
    assert len(result.convergence_history) > 0
    assert all(math.isnan(v) for v in result.convergence_history)


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_geometry_violation_gets_penalty_not_failure(case, boundary, rotor_diameters):
    """几何违规候选按明确惩罚处理，绝不触发 ObjectiveFailureError。"""
    optimizer_cls, config_factory, batch_size = case

    def objective_should_not_be_called(positions):
        raise AssertionError("几何违规候选不应进入目标函数")

    config = config_factory()
    optimizer = build_optimizer(
        optimizer_cls, config, boundary, rotor_diameters,
        objective_should_not_be_called,
    )

    # 全部风机堆在原点（矩形场地内部，但严重违反最小间距）。
    bad = np.zeros((batch_size, N_TURBINES * 2), dtype=np.float64)

    if optimizer_cls.__name__ == "GeneticAlgorithm":
        fitness, feasible = optimizer._evaluate_population(bad, 0)
    else:
        fitness, feasible = optimizer._evaluate_particles(bad, 0)

    assert not np.any(feasible)
    assert np.all(fitness < 0.0)
    assert optimizer._total_declared_infeasible == 0


# ---------------------------------------------------------------------------
# 共用批量评估器的单元测试
# ---------------------------------------------------------------------------


def test_evaluate_candidates_geometry_penalty_skips_objective():
    """几何违规候选不调用目标函数，直接取负惩罚。"""
    n_turb = 2
    population = np.array(
        [[0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 1000.0, 0.0]],
        dtype=np.float64,
    )
    penalties = np.array([5e6, 0.0])
    calls = []

    def fn(positions):
        calls.append(1)
        return 42.0

    fitness, feasible, n_declared = evaluate_candidates(
        fn, population, penalties, n_turb, penalty_factor=1e6,
        generation=0, algorithm="TEST",
    )

    assert fitness[0] == -5e6
    assert feasible[0] is np.False_
    assert fitness[1] == 42.0
    assert feasible[1] is np.True_
    assert n_declared == 0
    assert len(calls) == 1  # 违规候选被跳过


def test_evaluate_candidates_declared_infeasible_uses_penalty():
    population = np.zeros((1, 4), dtype=np.float64)

    def fn(positions):
        raise InfeasibleCandidate("不可行")

    fitness, feasible, n_declared = evaluate_candidates(
        fn, population, np.zeros(1), n_turbines=2, penalty_factor=1e6,
        generation=2, algorithm="TEST",
    )

    assert fitness[0] == -1e6
    assert feasible[0] is np.False_
    assert n_declared == 1


def test_evaluate_candidates_preserves_mid_batch_index_and_layout():
    """批次中途的故障必须保留正确的候选索引与布局。"""
    population = np.array(
        [[1.0, 1.0, 2.0, 2.0],
         [3.0, 3.0, 4.0, 4.0],
         [5.0, 5.0, 6.0, 6.0]],
        dtype=np.float64,
    )

    def fn(positions):
        if positions[0, 0] == 3.0:
            raise IndexError("内部数组越界")
        return 1.0

    with pytest.raises(ObjectiveFailureError) as exc_info:
        evaluate_candidates(
            fn, population, np.zeros(3), n_turbines=2, penalty_factor=1e6,
            generation=3, algorithm="TEST",
        )

    error = exc_info.value
    assert error.candidate_index == 1
    assert error.generation == 3
    np.testing.assert_array_equal(error.positions[0], [3.0, 3.0])
    assert isinstance(error.exception, IndexError)


def test_evaluate_candidates_rejects_inf():
    population = np.zeros((1, 4), dtype=np.float64)

    def fn(positions):
        return float("inf")

    with pytest.raises(ObjectiveFailureError) as exc_info:
        evaluate_candidates(
            fn, population, np.zeros(1), n_turbines=2, penalty_factor=1e6,
            generation=0,
        )

    assert exc_info.value.kind == ObjectiveFailureError.NON_FINITE
    assert math.isinf(exc_info.value.value)


# ---------------------------------------------------------------------------
# 创建期配置与构造参数校验（两种算法一致）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_illegal_population_size_fails_at_creation(
    case, boundary, rotor_diameters
):
    """非法种群规模必须在创建配置时就报错，而非深入搜索后数组报错。"""
    optimizer_cls, config_factory, _ = case
    size_field = (
        "population_size"
        if optimizer_cls.__name__ == "GeneticAlgorithm"
        else "swarm_size"
    )

    with pytest.raises(ValueError, match=size_field):
        config_factory(**{size_field: 1})  # 小于 2

    with pytest.raises(ValueError, match=size_field):
        config_factory(**{size_field: 0})

    with pytest.raises(ValueError, match=size_field):
        config_factory(**{size_field: 2.5})  # 非整数

    with pytest.raises(ValueError, match=size_field):
        config_factory(**{size_field: True})  # bool 不得冒充整数


@pytest.mark.parametrize(
    "case",
    list(OPTIMIZER_CASES.values()),
    ids=list(OPTIMIZER_CASES.keys()),
)
def test_constructor_combo_validation(case, boundary, rotor_diameters):
    """优化器创建时完成组合关系校验。"""
    optimizer_cls, config_factory, _ = case

    # n_turbines 非正/非整数。
    with pytest.raises(ValueError, match="n_turbines"):
        optimizer_cls(
            n_turbines=0,
            rotor_diameters=rotor_diameters,
            boundary=boundary,
            fitness_fn=healthy_objective,
            config=config_factory(),
        )

    # 直径数量与风机台数不匹配。
    with pytest.raises(ValueError, match="rotor_diameters"):
        optimizer_cls(
            n_turbines=N_TURBINES + 1,
            rotor_diameters=rotor_diameters,
            boundary=boundary,
            fitness_fn=healthy_objective,
            config=config_factory(),
        )

    # 直径含非有限值。
    bad_diameters = rotor_diameters.copy()
    bad_diameters[0] = np.nan
    with pytest.raises(ValueError, match="rotor_diameters"):
        optimizer_cls(
            n_turbines=N_TURBINES,
            rotor_diameters=bad_diameters,
            boundary=boundary,
            fitness_fn=healthy_objective,
            config=config_factory(),
        )

    # 目标函数不可调用。
    with pytest.raises(ValueError, match="fitness_fn"):
        optimizer_cls(
            n_turbines=N_TURBINES,
            rotor_diameters=rotor_diameters,
            boundary=boundary,
            fitness_fn=12345,
            config=config_factory(),
        )

    # 配置类型错误。
    wrong_config = (
        PSOConfig() if optimizer_cls.__name__ == "GeneticAlgorithm" else GAConfig()
    )
    with pytest.raises(ValueError, match="config"):
        optimizer_cls(
            n_turbines=N_TURBINES,
            rotor_diameters=rotor_diameters,
            boundary=boundary,
            fitness_fn=healthy_objective,
            config=wrong_config,
        )


def test_ga_specific_config_validation():
    """GA 专属组合校验：锦标赛规模与精英比例等。"""
    with pytest.raises(ValueError, match="tournament_size"):
        GAConfig(population_size=10, tournament_size=11)

    with pytest.raises(ValueError, match="tournament_size"):
        GAConfig(tournament_size=0)

    with pytest.raises(ValueError, match="elite_ratio"):
        GAConfig(population_size=4, elite_ratio=1.0)  # 精英数等于种群

    with pytest.raises(ValueError, match="crossover_rate"):
        GAConfig(crossover_rate=1.2)

    with pytest.raises(ValueError, match="mutation_rate"):
        GAConfig(mutation_rate=-0.1)

    with pytest.raises(ValueError, match="penalty_factor"):
        GAConfig(penalty_factor=0.0)

    with pytest.raises(ValueError, match="min_spacing_multiple"):
        GAConfig(min_spacing_multiple=-1.0)

    with pytest.raises(ValueError, match="max_generations"):
        GAConfig(max_generations=0)


def test_pso_specific_config_validation():
    """PSO 专属范围校验。"""
    with pytest.raises(ValueError, match="max_iterations"):
        PSOConfig(max_iterations=0)

    with pytest.raises(ValueError, match="max_velocity"):
        PSOConfig(max_velocity=0.0)

    with pytest.raises(ValueError, match="cognitive_coeff"):
        PSOConfig(cognitive_coeff=-0.1)

    with pytest.raises(ValueError, match="social_coeff"):
        PSOConfig(social_coeff=float("nan"))

    with pytest.raises(ValueError, match="inertia_weight"):
        PSOConfig(inertia_weight=-0.01)
