"""GA 与 PSO 的共同行为测试。

两种算法必须在以下场景上保持一致的语义：

- 首个候选即触发评估器故障（异常 / NaN）→ 立即终止，保留索引与布局；
- 中途候选触发评估器故障 → 同样立即终止，错误指向正确的候选与轮次；
- 合法的不可行候选（几何违反 / 显式 InfeasibleCandidate）→ 明确罚分，
  搜索继续并产出成功结果；
- 全部候选不可行 → 结果 success=False，不伪装成有效最优解；
- 非法配置在创建优化器时即报错，而不是运行深处的数组错误。
"""

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.optimization import (
    EvaluatorError,
    GAConfig,
    GeneticAlgorithm,
    InfeasibleCandidate,
    PSOConfig,
    ParticleSwarmOptimizer,
)
from wind_farm_opt.optimization.common import OptimizerConfigError

N_TURB = 4
DIAMETER = 100.0
BOUNDARY = create_rectangular_boundary(width=2000.0, height=2000.0)
ROTOR_DIAMETERS = np.full(N_TURB, DIAMETER)
POP_SIZE = 6
MAX_STEPS = 3

# ---------------------------------------------------------------------------
# 测试夹具：为两种算法构造相同语义的优化器
# ---------------------------------------------------------------------------


def make_optimizer(fitness_fn, *, config=None, algorithm="ga"):
    """用统一的小规模配置创建 GA 或 PSO 优化器。"""
    if algorithm == "ga":
        cfg = config or GAConfig(
            population_size=POP_SIZE,
            max_generations=MAX_STEPS,
            min_spacing_multiple=5.0,
            seed=123,
        )
        return GeneticAlgorithm(
            n_turbines=N_TURB,
            rotor_diameters=ROTOR_DIAMETERS,
            boundary=BOUNDARY,
            fitness_fn=fitness_fn,
            config=cfg,
        )

    cfg = config or PSOConfig(
        swarm_size=POP_SIZE,
        max_iterations=MAX_STEPS,
        min_spacing_multiple=5.0,
        seed=123,
    )
    return ParticleSwarmOptimizer(
        n_turbines=N_TURB,
        rotor_diameters=ROTOR_DIAMETERS,
        boundary=BOUNDARY,
        fitness_fn=fitness_fn,
        config=cfg,
    )


def good_fitness(positions):
    """合法的确定性评估函数：坐标和越大分越高，且始终有限。"""
    return float(1000.0 + positions.sum())


@pytest.fixture(params=["ga", "pso"])
def algorithm(request):
    return request.param


# ---------------------------------------------------------------------------
# 基线：正常评估时两种算法都能跑通
# ---------------------------------------------------------------------------


def test_normal_run_succeeds(algorithm):
    optimizer = make_optimizer(good_fitness, algorithm=algorithm)
    result = optimizer.optimize(verbose=False)

    assert result.success
    assert result.message == ""
    assert np.isfinite(result.best_fitness)
    assert result.best_fitness >= 1000.0
    assert result.best_positions.shape == (N_TURB, 2)
    assert result.final_fitness.shape == (POP_SIZE,)
    assert result.convergence_history  # 非空


# ---------------------------------------------------------------------------
# 场景一：首个候选评估器故障（异常 / NaN）→ 立即终止
# ---------------------------------------------------------------------------


def test_first_candidate_exception_aborts_immediately(algorithm):
    calls = []

    def failing_fitness(positions):
        calls.append(positions.copy())
        # 模拟评估器内部的维度错误：非领域异常，必须传播而非吞掉。
        raise ValueError("内部维度不匹配: 期望扇区数 12，实际 11")

    optimizer = make_optimizer(failing_fitness, algorithm=algorithm)

    with pytest.raises(EvaluatorError) as exc_info:
        optimizer.optimize(verbose=False)

    err = exc_info.value
    assert err.candidate_index == 0
    assert err.iteration == 0
    assert err.positions.shape == (N_TURB, 2)
    assert np.array_equal(err.positions, calls[0].reshape(N_TURB, 2))
    assert "ValueError" in err.reason
    assert "内部维度不匹配" in err.reason
    # 原始异常通过 __cause__ 保留
    assert isinstance(err.__cause__, ValueError)
    # “立即终止”：初始批次第一个候选失败后不得再评估其它候选
    assert len(calls) == 1


def test_first_candidate_nan_aborts_immediately(algorithm):
    def nan_fitness(positions):
        return float("nan")

    optimizer = make_optimizer(nan_fitness, algorithm=algorithm)

    with pytest.raises(EvaluatorError) as exc_info:
        optimizer.optimize(verbose=False)

    err = exc_info.value
    assert err.candidate_index == 0
    assert err.iteration == 0
    assert "非有限值" in err.reason
    assert err.positions.shape == (N_TURB, 2)


def test_first_candidate_inf_aborts(algorithm):
    def inf_fitness(positions):
        return np.inf

    with pytest.raises(EvaluatorError) as exc_info:
        make_optimizer(inf_fitness, algorithm=algorithm).optimize(verbose=False)

    assert exc_info.value.candidate_index == 0


# ---------------------------------------------------------------------------
# 场景二：中途候选故障 → 立即终止且定位到正确的候选/轮次/布局
# ---------------------------------------------------------------------------


def test_midrun_exception_aborts_with_candidate_context(algorithm):
    """前若干候选正常，第 3 个候选（索引 2）抛出非领域异常。"""
    call_log = []

    def flaky_fitness(positions):
        idx = len(call_log)
        call_log.append(positions.copy())
        if idx == 2:
            raise IndexError("扇区风速数组越界")
        return good_fitness(positions)

    optimizer = make_optimizer(flaky_fitness, algorithm=algorithm)

    with pytest.raises(EvaluatorError) as exc_info:
        optimizer.optimize(verbose=False)

    err = exc_info.value
    assert err.candidate_index == 2
    assert err.iteration == 0
    assert isinstance(err.__cause__, IndexError)
    assert "扇区风速数组越界" in err.reason
    # 保留的是第 3 个候选的原始布局
    assert np.array_equal(err.positions, call_log[2].reshape(N_TURB, 2))
    # 故障候选之后的候选不得被评估
    assert len(call_log) == 3


def test_midrun_nan_aborts_and_keeps_layout(algorithm):
    """初始批次全部成功后，第二轮首个调用目标函数的候选返回 -inf。"""
    call_log = []

    def fail_second_batch(positions):
        call_log.append(positions.copy())
        # 初始批次由 N 个有效生成的布局组成，恰好产生 POP_SIZE 次调用；
        # 之后第一次调用目标函数即返回非有限值。
        if len(call_log) > POP_SIZE:
            return float("-inf")
        return good_fitness(positions)

    optimizer = make_optimizer(fail_second_batch, algorithm=algorithm)

    with pytest.raises(EvaluatorError) as exc_info:
        optimizer.optimize(verbose=False)

    err = exc_info.value
    # 初始批次完整评估完，故障发生在第二轮
    assert err.iteration == 1
    assert 0 <= err.candidate_index < POP_SIZE
    assert "非有限值" in err.reason
    # 保留的布局正是触发 -inf 的那个候选
    assert np.array_equal(err.positions, call_log[-1].reshape(N_TURB, 2))
    assert len(call_log) == POP_SIZE + 1


# ---------------------------------------------------------------------------
# 场景三：合法的不可行候选 → 明确罚分，搜索继续
# ---------------------------------------------------------------------------


def test_declared_infeasible_candidate_gets_penalty_and_run_succeeds(algorithm):
    """目标函数对部分候选显式声明不可行，其余候选正常。"""
    state = {"rejected": 0}

    def reject_half(positions):
        if positions[:, 0].mean() < 0.0:
            state["rejected"] += 1
            raise InfeasibleCandidate("该布局在主导风向下不可接受")
        return good_fitness(positions)

    optimizer = make_optimizer(reject_half, algorithm=algorithm)
    result = optimizer.optimize(verbose=False)

    assert result.success
    # 搜索过程中确实出现过被显式拒绝的候选（均值历史曾被罚分拖低）
    assert state["rejected"] > 0
    assert min(result.mean_history) < 0.0
    # 存在有效评估值时，最优解必须来自有效候选而非罚分
    assert result.best_fitness > 0.0


def test_geometrically_infeasible_candidate_gets_penalty(algorithm):
    """几何违反候选不调用目标函数，按 -penalty_factor 量级罚分。"""
    evaluated = []

    def counting_fitness(positions):
        evaluated.append(positions.copy())
        return good_fitness(positions)

    cfg_kwargs = dict(
        min_spacing_multiple=5.0,
        penalty_factor=1e6,
        seed=7,
    )
    if algorithm == "ga":
        cfg = GAConfig(population_size=POP_SIZE, max_generations=1, **cfg_kwargs)
    else:
        cfg = PSOConfig(swarm_size=POP_SIZE, max_iterations=1, **cfg_kwargs)

    optimizer = make_optimizer(counting_fitness, config=cfg, algorithm=algorithm)

    # 手工构造批次：候选 0 合法（4 台机间距 1200m），候选 1 完全重叠。
    population = np.zeros((POP_SIZE, N_TURB * 2), dtype=np.float64)
    population[0] = np.array(
        [[-600.0, -600.0], [600.0, -600.0], [-600.0, 600.0], [600.0, 600.0]]
    ).flatten()
    # 其余候选所有风机堆在 (0, 0)：在边界内但严重违反间距
    overlapping = np.tile([0.0, 0.0], N_TURB)
    for k in range(1, POP_SIZE):
        population[k] = overlapping

    from wind_farm_opt.optimization.common import evaluate_batch

    batch = evaluate_batch(
        population,
        n_turbines=N_TURB,
        fitness_fn=counting_fitness,
        boundary=BOUNDARY,
        min_spacing=optimizer.min_spacing,
        penalty_factor=1e6,
        iteration=0,
        stage="测试批次",
    )

    assert batch.feasible[0]
    assert batch.fitness[0] > 0.0
    assert not batch.feasible[1]
    assert batch.fitness[1] <= -1e6  # 多对重叠 → 累计罚分
    # 几何违反候选不得触发目标函数：只评估了候选 0
    assert len(evaluated) == 1
    assert np.allclose(evaluated[0], population[0].reshape(N_TURB, 2))


# ---------------------------------------------------------------------------
# 场景四：全部候选无效 → success=False，不伪装成有效最优解
# ---------------------------------------------------------------------------


def test_all_candidates_infeasible_reports_failure(algorithm):
    def reject_all(positions):
        raise InfeasibleCandidate("物理模型拒绝任何布局")

    optimizer = make_optimizer(reject_all, algorithm=algorithm)
    result = optimizer.optimize(verbose=False)

    assert not result.success
    assert "没有任何" in result.message
    # 适应度全部是罚分（负值），但不得被报告为有效最优
    assert np.all(result.final_fitness < 0.0)
    assert result.best_fitness < 0.0
    assert result.best_positions.shape == (N_TURB, 2)


def test_all_infeasible_progress_does_not_claim_valid_aep(algorithm, capsys):
    def reject_all(positions):
        raise InfeasibleCandidate("拒绝")

    optimizer = make_optimizer(reject_all, algorithm=algorithm)
    result = optimizer.optimize(verbose=True)

    captured = capsys.readouterr()
    assert not result.success
    # 进度/收尾输出必须明确宣告无效，而不是打印“优化完成/最优净AEP”
    assert "无有效" in captured.out
    assert "最优净AEP" not in captured.out


# ---------------------------------------------------------------------------
# 配置校验：创建优化器时立即失败
# ---------------------------------------------------------------------------


def test_invalid_population_size_raises_at_construction(algorithm):
    bad_kwargs = dict(
        n_turbines=N_TURB,
        rotor_diameters=ROTOR_DIAMETERS,
        boundary=BOUNDARY,
        fitness_fn=good_fitness,
    )

    if algorithm == "ga":
        with pytest.raises(OptimizerConfigError, match="population_size"):
            GeneticAlgorithm(config=GAConfig(population_size=1), **bad_kwargs)
        with pytest.raises(OptimizerConfigError, match="population_size"):
            GeneticAlgorithm(config=GAConfig(population_size=0), **bad_kwargs)
    else:
        with pytest.raises(OptimizerConfigError, match="swarm_size"):
            ParticleSwarmOptimizer(config=PSOConfig(swarm_size=0), **bad_kwargs)


def test_invalid_tournament_size_raises_at_construction():
    """锦标赛规模越界（旧代码会在深处以 numpy 索引错误结束）。"""
    base = dict(
        n_turbines=N_TURB,
        rotor_diameters=ROTOR_DIAMETERS,
        boundary=BOUNDARY,
        fitness_fn=good_fitness,
    )

    with pytest.raises(OptimizerConfigError, match="tournament_size"):
        GeneticAlgorithm(config=GAConfig(population_size=5, tournament_size=6), **base)

    with pytest.raises(OptimizerConfigError, match="tournament_size"):
        GeneticAlgorithm(config=GAConfig(tournament_size=0), **base)


def test_invalid_rates_and_factors_raise_at_construction(algorithm):
    base = dict(
        n_turbines=N_TURB,
        rotor_diameters=ROTOR_DIAMETERS,
        boundary=BOUNDARY,
        fitness_fn=good_fitness,
    )

    if algorithm == "ga":
        with pytest.raises(OptimizerConfigError, match="crossover_rate"):
            GeneticAlgorithm(config=GAConfig(crossover_rate=1.5), **base)
        with pytest.raises(OptimizerConfigError, match="elite_ratio"):
            GeneticAlgorithm(config=GAConfig(elite_ratio=1.0), **base)
    else:
        with pytest.raises(OptimizerConfigError, match="cognitive_coeff"):
            ParticleSwarmOptimizer(
                config=PSOConfig(cognitive_coeff=-1.0), **base
            )
        with pytest.raises(OptimizerConfigError):
            ParticleSwarmOptimizer(
                config=PSOConfig(cognitive_coeff=0.0, social_coeff=0.0), **base
            )

    common_cfg_kwargs = dict(
        min_spacing_multiple=-5.0,
    )
    with pytest.raises(OptimizerConfigError, match="min_spacing_multiple"):
        if algorithm == "ga":
            GeneticAlgorithm(config=GAConfig(**common_cfg_kwargs), **base)
        else:
            ParticleSwarmOptimizer(config=PSOConfig(**common_cfg_kwargs), **base)


def test_invalid_constructor_inputs_raise_at_construction(algorithm):
    cls = ParticleSwarmOptimizer if algorithm == "pso" else GeneticAlgorithm

    with pytest.raises(OptimizerConfigError, match="n_turbines"):
        cls(
            n_turbines=0,
            rotor_diameters=ROTOR_DIAMETERS,
            boundary=BOUNDARY,
            fitness_fn=good_fitness,
        )

    with pytest.raises(OptimizerConfigError, match="rotor_diameters"):
        cls(
            n_turbines=N_TURB,
            rotor_diameters=np.full(N_TURB + 1, DIAMETER),
            boundary=BOUNDARY,
            fitness_fn=good_fitness,
        )

    with pytest.raises(OptimizerConfigError, match="fitness_fn"):
        cls(
            n_turbines=N_TURB,
            rotor_diameters=ROTOR_DIAMETERS,
            boundary=BOUNDARY,
            fitness_fn="not-callable",
        )
