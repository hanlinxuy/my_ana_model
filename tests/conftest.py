"""Pytest fixtures for MoE simulator tests."""

import pytest
from moe_simulator.core.cache import ExpertCache
from moe_simulator.core.config import RouterConfig
from moe_simulator.core.latency import BandwidthModel, LatencyCalculator
from moe_simulator.strategies.factory import StrategyFactory
from moe_simulator.strategies.cache_only import CacheOnlyStrategy
from moe_simulator.strategies.fixed_split import FixedSplitStrategy
from moe_simulator.strategies.hybrid import HybridStrategy
from moe_simulator.strategies.lru import LRUStrategy
from moe_simulator.strategies.pim_only import PIMOnlyStrategy


@pytest.fixture
def cache():
    """Create a fresh ExpertCache for each test."""
    return ExpertCache(capacity=4)


@pytest.fixture
def cache_small():
    """Create a small cache with capacity 2."""
    return ExpertCache(capacity=2)


@pytest.fixture
def cache_large():
    """Create a large cache with capacity 10."""
    return ExpertCache(capacity=10)


@pytest.fixture
def bandwidth_model():
    """Create default bandwidth model."""
    return BandwidthModel()


@pytest.fixture
def bandwidth_model_custom():
    """Create custom bandwidth model."""
    return BandwidthModel(
        flash_to_ddr=2.0,
        ddr_to_npu=16.0,
        flash_to_pim=8.0,
    )


@pytest.fixture
def latency_calculator():
    """Create latency calculator with default values."""
    return LatencyCalculator()


@pytest.fixture
def latency_calculator_custom():
    """Create latency calculator with custom values."""
    return LatencyCalculator(
        bandwidth_model=BandwidthModel(
            flash_to_ddr=2.0,
            ddr_to_npu=16.0,
            flash_to_pim=8.0,
        ),
        k1=2.0,
        k2=3.0,
        k3=1.5,
    )


@pytest.fixture
def router_config():
    """Create default router config."""
    return RouterConfig(
        num_experts=8,
        K=2,
        cache_size=4,
        k1=1,
        k2=2,
    )


@pytest.fixture
def router_config_small():
    """Create router config for small setup."""
    return RouterConfig(
        num_experts=4,
        K=1,
        cache_size=2,
        k1=0,
        k2=1,
    )


@pytest.fixture
def router_config_large():
    """Create router config for large setup."""
    return RouterConfig(
        num_experts=16,
        K=4,
        cache_size=8,
        k1=2,
        k2=4,
    )


@pytest.fixture
def strategies():
    """Register and return all strategy instances."""
    StrategyFactory.clear()
    StrategyFactory.register("cache_only", CacheOnlyStrategy)
    StrategyFactory.register("fixed_split", FixedSplitStrategy)
    StrategyFactory.register("hybrid", HybridStrategy)
    StrategyFactory.register("lru", LRUStrategy)
    StrategyFactory.register("pim_only", PIMOnlyStrategy)
    return {
        "cache_only": StrategyFactory.create("cache_only"),
        "fixed_split": StrategyFactory.create("fixed_split"),
        "hybrid": StrategyFactory.create("hybrid"),
        "lru": StrategyFactory.create("lru"),
        "pim_only": StrategyFactory.create("pim_only"),
    }


@pytest.fixture
def scores():
    """Create sample routing scores."""
    return [0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4]


@pytest.fixture
def scores_uniform():
    """Create uniform routing scores."""
    return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


@pytest.fixture
def scores_descending():
    """Create descending routing scores."""
    return [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
