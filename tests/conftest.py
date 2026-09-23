"""GA/PSO 共同测试的共享夹具与构造工具。"""

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.optimization.pso import PSOConfig, ParticleSwarmOptimizer


N_TURBINES = 4
ROTOR_DIAMETER = 100.0


@pytest.fixture
def boundary():
    return create_rectangular_boundary(width=4000, height=4000)


@pytest.fixture
def rotor_diameters():
    return np.full(N_TURBINES, ROTOR_DIAMETER, dtype=np.float64)


def make_small_ga(seed=123, **overrides):
    """小规模 GA 配置，保证测试快速且确定。"""
    params = dict(
        population_size=6,
        max_generations=3,
        elite_ratio=0.2,
        tournament_size=2,
        seed=seed,
    )
    params.update(overrides)
    return GAConfig(**params)


def make_small_pso(seed=123, **overrides):
    """小规模 PSO 配置，保证测试快速且确定。"""
    params = dict(
        swarm_size=6,
        max_iterations=3,
        seed=seed,
    )
    params.update(overrides)
    return PSOConfig(**params)


OPTIMIZER_CASES = {
    "ga": (GeneticAlgorithm, make_small_ga, 6),
    "pso": (ParticleSwarmOptimizer, make_small_pso, 6),
}


def build_optimizer(optimizer_cls, config, boundary, rotor_diameters, fitness_fn):
    """按算法类型构造优化器。"""
    return optimizer_cls(
        n_turbines=N_TURBINES,
        rotor_diameters=rotor_diameters,
        boundary=boundary,
        fitness_fn=fitness_fn,
        config=config,
    )
