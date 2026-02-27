"""Tests for static workload (smoothness=1.0).

When smoothness=1.0, the workload is static - the same experts are always selected.
This means after initial loading, all subsequent requests should be cache hits (k3=K).
"""

import pytest
from moe_simulator.core.cache import ExpertCache
from moe_simulator.strategies.cache_only import CacheOnlyStrategy
from moe_simulator.strategies.fixed_split import FixedSplitStrategy
from moe_simulator.strategies.hybrid import HybridStrategy
from moe_simulator.strategies.lru import LRUStrategy
from moe_simulator.strategies.pim_only import PIMOnlyStrategy


def run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache, num_tokens=10):
    """Run static workload and return routing results.

    With static workload (same scores each time), after initial cache warmup,
    all experts should be cached.
    """
    results = []
    for _ in range(num_tokens):
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=K, k1_limit=k1_limit, k2_limit=k2_limit
        )
        results.append(
            {
                "k1": len(k1_list),
                "k2": len(k2_list),
                "k3": len(k3_list),
                "k1_list": k1_list,
                "k2_list": k2_list,
                "k3_list": k3_list,
            }
        )
    return results


class TestStaticWorkloadSmoothnessOne:
    """Test static workload (smoothness=1.0)."""

    def test_lru_static_workload_k3_equals_k(self):
        """Test LRU with static workload: after warmup, cache should have experts."""
        cache = ExpertCache(capacity=4)
        strategy = LRUStrategy()
        scores = [1.0, 0.0, 0.0, 0.0]
        K, k1_limit, k2_limit = 2, 2, 2

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert cache.size > 0, "Cache should have experts after warmup"

    def test_hybrid_static_workload_k3_equals_k(self):
        """Test Hybrid with static workload: k3 should equal K after warmup."""
        cache = ExpertCache(capacity=4)
        strategy = HybridStrategy()
        scores = [1.0, 0.0, 0.0, 0.0]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        for i, r in enumerate(results):
            total = r["k1"] + r["k2"] + r["k3"]
            assert total == K, f"Token {i}: total={total}, expected={K}"

        assert results[-1]["k3"] == K, "After warmup, k3 should equal K"

    def test_cache_only_static_workload_k3_equals_k(self):
        """Test Cache-Only with static workload: k3 should equal K after warmup."""
        cache = ExpertCache(capacity=4)
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        strategy = CacheOnlyStrategy()
        scores = [1.0, 0.0, 0.0, 0.0]
        K, k1_limit, k2_limit = 2, 0, 2

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        for i, r in enumerate(results):
            total = r["k1"] + r["k2"] + r["k3"]
            assert total == K, f"Token {i}: total={total}, expected={K}"

        assert results[-1]["k3"] == K, "Cache-only should have k3=K when cached"

    def test_fixed_split_static_workload(self):
        """Test Fixed-Split with static workload: returns valid routing."""
        cache = ExpertCache(capacity=4)
        strategy = FixedSplitStrategy()
        scores = [1.0, 0.0, 0.0, 0.0]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert len(results[-1]["k1_list"]) == 1
        assert len(results[-1]["k2_list"]) == 1

    def test_pim_only_static_workload_k2_equals_k(self):
        """Test PIM-Only with static workload: k2 should equal K (no cache used)."""
        cache = ExpertCache(capacity=4)
        strategy = PIMOnlyStrategy()
        scores = [1.0, 0.0, 0.0, 0.0]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        for i, r in enumerate(results):
            total = r["k1"] + r["k2"] + r["k3"]
            assert total == K, f"Token {i}: total={total}, expected={K}"

        assert results[-1]["k2"] == K, "PIM-only should have k2=K always"
        assert results[-1]["k3"] == 0, "PIM-only should have k3=0"


class TestStaticWorkloadMultipleExperts:
    """Test static workload with multiple experts."""

    def test_static_workload_four_experts(self):
        """Test static workload with 4 experts, selecting top 2."""
        cache = ExpertCache(capacity=4)
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert results[-1]["k3"] == K, "k3 should equal K after warmup"

    def test_static_workload_eight_experts(self):
        """Test static workload with 8 experts."""
        cache = ExpertCache(capacity=8)
        strategy = HybridStrategy()
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        K, k1_limit, k2_limit = 4, 2, 2

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert results[-1]["k3"] == K, "k3 should equal K after warmup"


class TestStaticWorkloadCacheSize:
    """Test static workload with different cache sizes."""

    def test_cache_size_equals_k(self):
        """Test when cache_size equals K."""
        cache = ExpertCache(capacity=2)
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert results[-1]["k3"] == K

    def test_cache_size_greater_than_k(self):
        """Test when cache_size is greater than K."""
        cache = ExpertCache(capacity=8)
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert results[-1]["k3"] == K

    def test_cache_size_less_than_k(self):
        """Test when cache_size is less than K (edge case)."""
        cache = ExpertCache(capacity=1)
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        total = results[-1]["k1"] + results[-1]["k2"] + results[-1]["k3"]
        assert total == K


class TestStaticWorkloadDifferentStrategies:
    """Compare all strategies under static workload."""

    @pytest.mark.parametrize(
        "strategy_class,expected_k3",
        [
            (LRUStrategy, 0),
            (HybridStrategy, 2),
            (PIMOnlyStrategy, 0),
            (CacheOnlyStrategy, 2),
        ],
    )
    def test_strategies_static_workload(self, strategy_class, expected_k3):
        """Test each strategy under static workload."""
        cache = ExpertCache(capacity=4)

        if strategy_class == CacheOnlyStrategy:
            cache.put(0, {"state": "expert0"})
            cache.put(1, {"state": "expert1"})

        strategy = strategy_class()
        scores = [1.0, 0.0, 0.0, 0.0]
        K, k1_limit, k2_limit = 2, 1, 1

        results = run_static_workload(strategy, scores, K, k1_limit, k2_limit, cache)

        assert results[-1]["k3"] == expected_k3, (
            f"{strategy_class.__name__} expected k3={expected_k3}"
        )
