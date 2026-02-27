"""Simulation runner for executing MoE routing simulations.

Provides the core simulation logic that executes routing strategies
across multiple smoothness levels and aggregates results.
"""

import random
from typing import Callable, Dict, List, Optional

import numpy as np

from moe_simulator.core.cache import ExpertCache
from moe_simulator.core.config import RouterConfig
from moe_simulator.core.config_loader import SimulationConfig
from moe_simulator.core.latency import BandwidthModel, LatencyCalculator
from moe_simulator.core.results import ResultsAggregator, SimulationResult
from moe_simulator.strategies.base import RoutingStrategy
from moe_simulator.strategies.factory import StrategyFactory


def generate_scores(
    num_experts: int,
    num_tokens: int,
    smoothness: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate routing scores based on smoothness level.

    Args:
        num_experts: Number of experts.
        num_tokens: Number of tokens to generate.
        smoothness: Smoothness level (0.0 = random, 1.0 = static).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (num_tokens, num_experts) with scores.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Base static scores (highest for first expert, decreasing)
    static_scores = np.array([1.0 - (i / num_experts) for i in range(num_experts)])

    # Generate random noise based on smoothness
    # smoothness=1.0 -> all static, smoothness=0.0 -> all random
    noise = np.random.rand(num_tokens, num_experts)

    # Interpolate between static and random based on smoothness
    # At smoothness=1.0, we want pure static (no noise)
    # At smoothness=0.0, we want pure random (all noise)
    scores = (1 - smoothness) * noise + smoothness * static_scores

    return scores


class SimulationRunner:
    """Runner for executing simulation experiments.

    Executes routing strategies across different smoothness levels
    and computes latency metrics.
    """

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize simulation runner.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.router_config = config.to_router_config()
        self.bandwidth_model = config.to_bandwidth_model()
        self.latency_calculator = LatencyCalculator(
            bandwidth_model=self.bandwidth_model,
            k1=1.0,  # Will be set per-token
            k2=1.0,
            k3=1.0,
        )
        self._register_strategies()

    def _register_strategies(self) -> None:
        """Register all available strategies."""
        # Import here to avoid circular imports
        from moe_simulator.strategies.cache_only import CacheOnlyStrategy
        from moe_simulator.strategies.fixed_split import FixedSplitStrategy
        from moe_simulator.strategies.hybrid import HybridStrategy
        from moe_simulator.strategies.lru import LRUStrategy
        from moe_simulator.strategies.pim_only import PIMOnlyStrategy

        # Clear and re-register
        StrategyFactory.clear()
        StrategyFactory.register("lru", LRUStrategy)
        StrategyFactory.register("pim_only", PIMOnlyStrategy)
        StrategyFactory.register("cache_only", CacheOnlyStrategy)
        StrategyFactory.register("fixed_split", FixedSplitStrategy)
        StrategyFactory.register("hybrid", HybridStrategy)

    def run_single_strategy(
        self,
        strategy_name: str,
        smoothness: float,
    ) -> SimulationResult:
        """Run simulation for a single strategy at a given smoothness level.

        Args:
            strategy_name: Name of the strategy to run.
            smoothness: Smoothness level for score generation.

        Returns:
            SimulationResult with metrics.
        """
        # Generate scores
        scores = generate_scores(
            num_experts=self.config.num_experts,
            num_tokens=self.config.num_tokens,
            smoothness=smoothness,
            seed=self.config.random_seed,
        )

        # Create strategy and cache
        strategy = StrategyFactory.create(strategy_name)
        cache = ExpertCache(capacity=self.config.cache_size)

        # Track metrics
        k1_counts: List[int] = []
        k2_counts: List[int] = []
        k3_counts: List[int] = []
        latencies: List[float] = []

        # Run simulation for each token
        for token_scores in scores:
            # Select experts
            k1_list, k2_list, k3_list = strategy.select_experts(
                token_scores,
                cache,
                K=self.config.K,
                k1_limit=int(self.config.k1),
                k2_limit=int(self.config.k2),
            )

            # Update cache with selected experts
            for expert_id in list(k1_list) + list(k2_list):
                cache.put(expert_id, {"state": f"expert_{expert_id}"})

            # Count routing decisions
            k1_counts.append(len(k1_list))
            k2_counts.append(len(k2_list))
            k3_counts.append(len(k3_list))

            # Calculate latency
            latency = self.latency_calculator.calculate(
                k1=len(k1_list),
                k2=len(k2_list),
                k3=len(k3_list),
            )
            latencies.append(latency)

        # Compute aggregate metrics
        total_k1 = sum(k1_counts)
        total_k2 = sum(k2_counts)
        total_k3 = sum(k3_counts)
        total_experts = total_k1 + total_k2 + total_k3

        avg_latency = np.mean(latencies) if latencies else 0.0
        total_latency = sum(latencies)

        # Cache hit rate = k3 / (k1 + k2 + k3)
        cache_hit_rate = total_k3 / total_experts if total_experts > 0 else 0.0

        # Compute ratios
        k1_ratio = total_k1 / total_experts if total_experts > 0 else 0.0
        k2_ratio = total_k2 / total_experts if total_experts > 0 else 0.0
        k3_ratio = total_k3 / total_experts if total_experts > 0 else 0.0

        return SimulationResult(
            strategy=strategy_name,
            smoothness=smoothness,
            num_tokens=self.config.num_tokens,
            k1_count=total_k1,
            k2_count=total_k2,
            k3_count=total_k3,
            avg_latency=float(avg_latency),
            total_latency=float(total_latency),
            cache_hit_rate=float(cache_hit_rate),
            cache_miss_count=total_k1 + total_k2,
            k1_ratio=float(k1_ratio),
            k2_ratio=float(k2_ratio),
            k3_ratio=float(k3_ratio),
        )

    def run(
        self,
        strategies: Optional[List[str]] = None,
        smoothness_levels: Optional[List[float]] = None,
    ) -> ResultsAggregator:
        """Run simulations for all strategies across smoothness levels.

        Args:
            strategies: List of strategy names to run. If None, runs all.
            smoothness_levels: List of smoothness levels. If None, uses config.

        Returns:
            ResultsAggregator containing all results.
        """
        if strategies is None:
            strategies = StrategyFactory.list_strategies()

        if smoothness_levels is None:
            smoothness_levels = self.config.smoothness_levels

        aggregator = ResultsAggregator()
        aggregator.config = self.config.to_dict()

        for strategy_name in strategies:
            for smoothness in smoothness_levels:
                result = self.run_single_strategy(strategy_name, smoothness)
                aggregator.add_result(result)

        return aggregator


def create_runner(
    config: Optional[SimulationConfig] = None, **kwargs
) -> SimulationRunner:
    """Create a simulation runner with config.

    Args:
        config: SimulationConfig instance. If None, creates from kwargs.
        **kwargs: Config parameters (used if config is None).

    Returns:
        SimulationRunner instance.
    """
    if config is None:
        config = SimulationConfig(**kwargs)
    return SimulationRunner(config)
