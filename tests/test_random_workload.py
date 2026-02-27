"""Tests for random workload (smoothness=0.0).

When smoothness=0.0, the workload is random - experts are selected uniformly at random.
With a small cache relative to the number of experts, we expect high miss rates.
"""

import random
import pytest
from moe_simulator.core.cache import ExpertCache
from moe_simulator.strategies.cache_only import CacheOnlyStrategy
from moe_simulator.strategies.fixed_split import FixedSplitStrategy
from moe_simulator.strategies.hybrid import HybridStrategy
from moe_simulator.strategies.lru import LRUStrategy
from moe_simulator.strategies.pim_only import PIMOnlyStrategy


def run_random_workload(
    strategy, num_experts, K, k1_limit, k2_limit, cache_size, num_tokens=100, seed=42
):
    """Run random workload and return cache statistics.

    With random workload and small cache, expect high miss rate.
    """
    random.seed(seed)
    cache = ExpertCache(capacity=cache_size)
    strategy_instance = strategy()

    total_requests = 0
    total_hits = 0

    for _ in range(num_tokens):
        scores = [random.random() for _ in range(num_experts)]
        k1_list, k2_list, k3_list = strategy_instance.select_experts(
            scores, cache, K=K, k1_limit=k1_limit, k2_limit=k2_limit
        )
        total_requests += K
        total_hits += len(k3_list)

    miss_rate = 1.0 - (total_hits / total_requests) if total_requests > 0 else 0.0
    hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

    return {
        "total_requests": total_requests,
        "total_hits": total_hits,
        "miss_rate": miss_rate,
        "hit_rate": hit_rate,
        "cache_size": cache_size,
        "num_experts": num_experts,
    }


class TestRandomWorkloadSmoothnessZero:
    """Test random workload (smoothness=0.0)."""

    def test_hybrid_random_workload_high_miss_rate(self):
        """Test Hybrid with random workload should have high miss rate."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=16,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=4,
            num_tokens=100,
        )

        assert result["miss_rate"] > 0.5, (
            f"Expected high miss rate, got {result['miss_rate']:.2%}"
        )

    def test_lru_random_workload_high_miss_rate(self):
        """Test LRU with random workload should have high miss rate."""
        result = run_random_workload(
            LRUStrategy,
            num_experts=16,
            K=2,
            k1_limit=2,
            k2_limit=0,
            cache_size=4,
            num_tokens=100,
        )

        assert result["miss_rate"] > 0.5, (
            f"Expected high miss rate, got {result['miss_rate']:.2%}"
        )

    def test_cache_only_random_workload(self):
        """Test Cache-Only with random workload."""
        result = run_random_workload(
            CacheOnlyStrategy,
            num_experts=16,
            K=2,
            k1_limit=0,
            k2_limit=2,
            cache_size=4,
            num_tokens=100,
        )

        assert result["miss_rate"] > 0.5, (
            f"Expected high miss rate, got {result['miss_rate']:.2%}"
        )


class TestRandomWorkloadMissRate:
    """Test miss rate calculations under random workload."""

    def test_miss_rate_greater_than_90_percent(self):
        """Test that miss rate > 90% when cache is small relative to experts."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=32,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=2,
            num_tokens=200,
            seed=123,
        )

        assert result["miss_rate"] > 0.9, (
            f"Expected miss rate > 90%, got {result['miss_rate']:.2%}"
        )

    def test_miss_rate_with_larger_cache(self):
        """Test miss rate with larger cache still high but lower."""
        result_small = run_random_workload(
            HybridStrategy,
            num_experts=16,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=2,
            num_tokens=100,
        )

        result_large = run_random_workload(
            HybridStrategy,
            num_experts=16,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=8,
            num_tokens=100,
        )

        assert result_large["miss_rate"] < result_small["miss_rate"], (
            "Larger cache should have lower miss rate"
        )

    def test_hit_rate_approaches_zero(self):
        """Test that hit rate approaches 0 with very small cache."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=64,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=2,
            num_tokens=200,
            seed=456,
        )

        assert result["hit_rate"] < 0.1, (
            f"Expected hit rate < 10%, got {result['hit_rate']:.2%}"
        )


class TestRandomWorkloadDifferentK:
    """Test random workload with different K values."""

    def test_k_equals_1(self):
        """Test random workload with K=1."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=16,
            K=1,
            k1_limit=1,
            k2_limit=0,
            cache_size=2,
            num_tokens=100,
        )

        assert result["total_requests"] == 100

    def test_k_equals_4(self):
        """Test random workload with K=4."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=16,
            K=4,
            k1_limit=2,
            k2_limit=2,
            cache_size=4,
            num_tokens=100,
        )

        assert result["total_requests"] == 400

    def test_k_equals_8(self):
        """Test random workload with K=8."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=32,
            K=8,
            k1_limit=4,
            k2_limit=4,
            cache_size=8,
            num_tokens=50,
        )

        assert result["total_requests"] == 400


class TestRandomWorkloadCompareStrategies:
    """Compare different strategies under random workload."""

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LRUStrategy,
            HybridStrategy,
            PIMOnlyStrategy,
            CacheOnlyStrategy,
        ],
    )
    def test_all_strategies_high_miss_rate(self, strategy_class):
        """Test that all strategies have high miss rate with random workload."""
        result = run_random_workload(
            strategy_class,
            num_experts=16,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=2,
            num_tokens=100,
        )

        if strategy_class == PIMOnlyStrategy:
            assert result["hit_rate"] == 0.0, "PIM-only should always have 0 hits"
        else:
            assert result["miss_rate"] > 0.5, (
                f"{strategy_class.__name__} expected miss rate > 50%"
            )


class TestRandomWorkloadEdgeCases:
    """Test edge cases for random workload."""

    def test_cache_size_equals_num_experts(self):
        """Test when cache size equals number of experts (no eviction)."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=4,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=4,
            num_tokens=100,
        )

        assert result["cache_size"] == result["num_experts"]

    def test_single_token(self):
        """Test with single token."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=8,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=4,
            num_tokens=1,
        )

        assert result["total_requests"] == 2

    def test_small_num_tokens(self):
        """Test with small number of tokens."""
        result = run_random_workload(
            HybridStrategy,
            num_experts=8,
            K=2,
            k1_limit=1,
            k2_limit=1,
            cache_size=2,
            num_tokens=10,
        )

        assert result["total_requests"] == 20
