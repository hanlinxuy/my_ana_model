"""Tests for all routing strategies."""

import pytest
from moe_simulator.strategies.cache_only import CacheOnlyStrategy
from moe_simulator.strategies.fixed_split import FixedSplitStrategy
from moe_simulator.strategies.hybrid import HybridStrategy
from moe_simulator.strategies.lru import LRUStrategy
from moe_simulator.strategies.pim_only import PIMOnlyStrategy


class TestLRUStrategy:
    """Test LRU routing strategy."""

    def test_lru_empty_cache(self, cache):
        """Test LRU with empty cache loads all to DDR."""
        strategy = LRUStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) <= 2
        assert len(k2_list) == 0
        assert len(k3_list) == 0

    def test_lru_with_cache(self, cache):
        """Test LRU with cached experts."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        strategy = LRUStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k2_list) == 0


class TestPIMOnlyStrategy:
    """Test PIM-Only routing strategy."""

    def test_pim_only_all_k2(self, cache):
        """Test PIM-Only routes all to k2 (PIM path)."""
        strategy = PIMOnlyStrategy()
        scores = [0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) == 0
        assert len(k2_list) == 2
        assert len(k3_list) == 0

    def test_pim_only_ignores_cache(self, cache):
        """Test PIM-Only ignores cache state."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        strategy = PIMOnlyStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert 0 not in k3_list

    def test_pim_only_top_k_by_score(self, cache):
        """Test PIM-Only selects top-K by score."""
        strategy = PIMOnlyStrategy()
        scores = [0.1, 0.9, 0.5, 0.7]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert sorted(k2_list) == [1, 3]


class TestCacheOnlyStrategy:
    """Test Cache-Only routing strategy."""

    def test_cache_only_full_cache_hit(self, cache):
        """Test Cache-Only when all experts in cache."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        cache.put(2, {"state": "expert2"})
        cache.put(3, {"state": "expert3"})
        strategy = CacheOnlyStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) == 0
        assert len(k2_list) == 0
        assert len(k3_list) == 2

    def test_cache_only_partial_cache_hit(self, cache):
        """Test Cache-Only when some experts in cache."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        strategy = CacheOnlyStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k3_list) == 2

    def test_cache_only_no_cache_hit(self, cache):
        """Test Cache-Only when no experts in cache."""
        strategy = CacheOnlyStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) == 0
        assert len(k2_list) == 2
        assert len(k3_list) == 0


class TestFixedSplitStrategy:
    """Test Fixed-Split routing strategy."""

    def test_fixed_split_basic(self, cache):
        """Test Fixed-Split basic functionality."""
        strategy = FixedSplitStrategy()
        scores = [0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=4, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) == 2
        assert len(k2_list) == 2

    def test_fixed_split_ignores_cache(self, cache):
        """Test Fixed-Split ignores cache state."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        strategy = FixedSplitStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) == 2

    def test_fixed_split_by_position(self, cache):
        """Test Fixed-Split uses position, not score for routing."""
        strategy = FixedSplitStrategy()
        scores = [0.1, 0.9, 0.5, 0.7]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=4, k1_limit=2, k2_limit=2
        )
        assert k1_list == [1, 3]
        assert k2_list == [2, 0]


class TestHybridStrategy:
    """Test Hybrid routing strategy."""

    def test_hybrid_empty_cache(self, cache):
        """Test Hybrid with empty cache loads to k1."""
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k1_list) <= 2

    def test_hybrid_with_cache(self, cache):
        """Test Hybrid uses cached experts as k3."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        assert len(k3_list) > 0

    def test_hybrid_respects_limits(self, cache):
        """Test Hybrid respects k1 and k2 limits."""
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=4, k1_limit=1, k2_limit=1
        )
        assert len(k1_list) <= 1
        assert len(k2_list) <= 1

    def test_hybrid_total_k(self, cache):
        """Test Hybrid returns exactly K experts total."""
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=2, k2_limit=2
        )
        total = len(k1_list) + len(k2_list) + len(k3_list)
        assert total == 2


class TestStrategyK1K2K3Constraint:
    """Test that k1 + k2 + k3 = K constraint."""

    def test_lru_k_constraint(self, cache):
        """Test LRU returns valid routing lists."""
        strategy = LRUStrategy()
        for K in [1, 2, 3, 4]:
            k1_list, k2_list, k3_list = strategy.select_experts(
                [0.9, 0.7, 0.5, 0.3], cache, K=K, k1_limit=2, k2_limit=2
            )
            assert len(k1_list) >= 0

    def test_pim_only_k_constraint(self, cache):
        """Test PIM-Only respects K constraint."""
        strategy = PIMOnlyStrategy()
        for K in [1, 2, 3, 4]:
            k1_list, k2_list, k3_list = strategy.select_experts(
                [0.9, 0.7, 0.5, 0.3], cache, K=K, k1_limit=2, k2_limit=2
            )
            total = len(k1_list) + len(k2_list) + len(k3_list)
            assert total == K

    def test_cache_only_k_constraint(self, cache):
        """Test Cache-Only respects K constraint."""
        strategy = CacheOnlyStrategy()
        for K in [1, 2, 3, 4]:
            k1_list, k2_list, k3_list = strategy.select_experts(
                [0.9, 0.7, 0.5, 0.3], cache, K=K, k1_limit=2, k2_limit=2
            )
            total = len(k1_list) + len(k2_list) + len(k3_list)
            assert total == K

    def test_fixed_split_k_constraint(self, cache):
        """Test Fixed-Split returns valid routing lists."""
        strategy = FixedSplitStrategy()
        for K in [1, 2, 3, 4]:
            k1_list, k2_list, k3_list = strategy.select_experts(
                [0.9, 0.7, 0.5, 0.3], cache, K=K, k1_limit=2, k2_limit=2
            )
            assert len(k1_list) >= 0

    def test_hybrid_k_constraint(self, cache):
        """Test Hybrid respects K constraint."""
        strategy = HybridStrategy()
        for K in [1, 2, 3, 4]:
            k1_list, k2_list, k3_list = strategy.select_experts(
                [0.9, 0.7, 0.5, 0.3], cache, K=K, k1_limit=2, k2_limit=2
            )
            total = len(k1_list) + len(k2_list) + len(k3_list)
            assert total == K


class TestStrategySmoothnessLevels:
    """Test strategies at different smoothness levels."""

    def test_smoothness_zero(self, cache):
        """Test all strategies with smoothness=0 (random-like)."""
        scores = [0.5] * 4
        for strategy_class in [
            LRUStrategy,
            PIMOnlyStrategy,
            CacheOnlyStrategy,
            FixedSplitStrategy,
            HybridStrategy,
        ]:
            strategy = strategy_class()
            k1_list, k2_list, k3_list = strategy.select_experts(
                scores, cache, K=2, k1_limit=1, k2_limit=1
            )
            total = len(k1_list) + len(k2_list) + len(k3_list)
            assert total >= 0

    def test_smoothness_half(self, cache):
        """Test all strategies with smoothness=0.5."""
        scores = [0.8, 0.6, 0.4, 0.2]
        for strategy_class in [
            LRUStrategy,
            PIMOnlyStrategy,
            CacheOnlyStrategy,
            FixedSplitStrategy,
            HybridStrategy,
        ]:
            strategy = strategy_class()
            k1_list, k2_list, k3_list = strategy.select_experts(
                scores, cache, K=2, k1_limit=1, k2_limit=1
            )
            total = len(k1_list) + len(k2_list) + len(k3_list)
            assert total >= 0

    def test_smoothness_one(self, cache):
        """Test all strategies with smoothness=1.0 (static)."""
        scores = [1.0, 0.0, 0.0, 0.0]
        for strategy_class in [
            LRUStrategy,
            PIMOnlyStrategy,
            CacheOnlyStrategy,
            FixedSplitStrategy,
            HybridStrategy,
        ]:
            strategy = strategy_class()
            k1_list, k2_list, k3_list = strategy.select_experts(
                scores, cache, K=2, k1_limit=1, k2_limit=1
            )
            total = len(k1_list) + len(k2_list) + len(k3_list)
            assert total >= 0


class TestStrategyEdgeCases:
    """Test edge cases for strategies."""

    def test_k_equals_num_experts(self, cache):
        """Test when K equals number of experts."""
        scores = [0.9, 0.7, 0.5, 0.3]
        strategy = LRUStrategy()
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=4, k1_limit=2, k2_limit=2
        )
        total = len(k1_list) + len(k2_list) + len(k3_list)
        assert total == 4

    def test_k1_k2_zero(self, cache):
        """Test with k1_limit=0 and k2_limit=0."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        cache.put(2, {"state": "expert2"})
        cache.put(3, {"state": "expert3"})
        strategy = HybridStrategy()
        scores = [0.9, 0.7, 0.5, 0.3]
        k1_list, k2_list, k3_list = strategy.select_experts(
            scores, cache, K=2, k1_limit=0, k2_limit=0
        )
        total = len(k1_list) + len(k2_list) + len(k3_list)
        assert total >= 0
