"""Tests for ExpertCache."""

import pytest
from moe_simulator.core.cache import ExpertCache


class TestExpertCacheBasics:
    """Test basic ExpertCache operations."""

    def test_create_valid_cache(self):
        """Test creating cache with valid capacity."""
        cache = ExpertCache(capacity=4)
        assert cache.capacity == 4
        assert cache.size == 0

    def test_create_invalid_cache_zero(self):
        """Test creating cache with zero capacity raises error."""
        with pytest.raises(ValueError, match="positive"):
            ExpertCache(capacity=0)

    def test_create_invalid_cache_negative(self):
        """Test creating cache with negative capacity raises error."""
        with pytest.raises(ValueError, match="positive"):
            ExpertCache(capacity=-1)


class TestExpertCachePutGet:
    """Test put and get operations."""

    def test_put_and_get(self, cache):
        """Test basic put and get operations."""
        cache.put(0, {"state": "loaded"})
        result = cache.get(0)
        assert result == {"state": "loaded"}

    def test_get_miss(self, cache):
        """Test get on empty cache returns None."""
        result = cache.get(0)
        assert result is None

    def test_get_after_eviction(self, cache_small):
        """Test get after expert is evicted."""
        cache_small.put(0, {"state": "expert0"})
        cache_small.put(1, {"state": "expert1"})
        assert cache_small.size == 2
        cache_small.put(2, {"state": "expert2"})
        result = cache_small.get(0)
        assert result is None

    def test_update_existing(self, cache):
        """Test updating an existing expert's state."""
        cache.put(0, {"state": "v1"})
        cache.put(0, {"state": "v2"})
        result = cache.get(0)
        assert result == {"state": "v2"}


class TestExpertCacheEvict:
    """Test eviction operations."""

    def test_evict_when_not_full(self, cache):
        """Test evict when cache is not full evicts LRU item."""
        cache.put(0, {"state": "expert0"})
        result = cache.evict()
        assert result == 0

    def test_evict_when_full(self, cache_small):
        """Test evict when cache is full evicts LRU."""
        cache_small.put(0, {"state": "expert0"})
        cache_small.put(1, {"state": "expert1"})
        result = cache_small.evict()
        assert result == 0

    def test_multiple_evictions(self, cache_small):
        """Test multiple evictions."""
        cache_small.put(0, {"state": "expert0"})
        cache_small.put(1, {"state": "expert1"})
        cache_small.evict()
        cache_small.put(2, {"state": "expert2"})
        cache_small.evict()
        result = cache_small.evict()
        assert result == 2


class TestExpertCacheLRU:
    """Test LRU update operations."""

    def test_update_lru(self, cache):
        """Test updating LRU order."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        cache.update_lru(0)
        cache.put(2, {"state": "expert2"})
        result = cache.evict()
        assert result == 1

    def test_update_lru_not_in_cache(self, cache):
        """Test updating LRU for non-existent expert does nothing."""
        cache.put(0, {"state": "expert0"})
        cache.update_lru(99)
        assert cache.size == 1


class TestExpertCacheContains:
    """Test contains method."""

    def test_contains_true(self, cache):
        """Test contains returns True for cached expert."""
        cache.put(0, {"state": "expert0"})
        assert cache.contains(0) is True

    def test_contains_false(self, cache):
        """Test contains returns False for non-cached expert."""
        cache.put(0, {"state": "expert0"})
        assert cache.contains(1) is False


class TestExpertCacheClear:
    """Test clear operation."""

    def test_clear(self, cache):
        """Test clearing cache resets all state."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        cache.get(0)
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestExpertCacheHitRate:
    """Test hit rate calculation."""

    def test_hit_rate_no_requests(self, cache):
        """Test hit rate with no requests is 0."""
        assert cache.hit_rate == 0.0

    def test_hit_rate_all_hits(self, cache):
        """Test hit rate with all hits is 1.0."""
        cache.put(0, {"state": "expert0"})
        cache.get(0)
        cache.get(0)
        cache.get(0)
        assert cache.hit_rate == 1.0

    def test_hit_rate_all_misses(self, cache):
        """Test hit rate with all misses is 0.0."""
        cache.get(0)
        cache.get(1)
        cache.get(2)
        assert cache.hit_rate == 0.0

    def test_hit_rate_mixed(self, cache):
        """Test hit rate with mixed hits and misses."""
        cache.put(0, {"state": "expert0"})
        cache.get(0)
        cache.get(1)
        cache.get(2)
        cache.get(0)
        assert cache.hits == 2
        assert cache.misses == 2
        assert cache.hit_rate == 0.5


class TestExpertCacheGetCachedExperts:
    """Test get_cached_experts method."""

    def test_get_cached_experts_empty(self, cache):
        """Test get_cached_experts on empty cache."""
        assert cache.get_cached_experts() == []

    def test_get_cached_experts_multiple(self, cache):
        """Test get_cached_experts returns all cached IDs."""
        cache.put(0, {"state": "expert0"})
        cache.put(1, {"state": "expert1"})
        cache.put(2, {"state": "expert2"})
        result = cache.get_cached_experts()
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_get_cached_experts_after_eviction(self, cache_small):
        """Test get_cached_experts after eviction."""
        cache_small.put(0, {"state": "expert0"})
        cache_small.put(1, {"state": "expert1"})
        cache_small.put(2, {"state": "expert2"})
        result = cache_small.get_cached_experts()
        assert len(result) == 2
        assert 0 not in result


class TestExpertCacheGetStats:
    """Test get_stats method."""

    def test_get_stats(self, cache):
        """Test get_stats returns all statistics."""
        cache.put(0, {"state": "expert0"})
        cache.get(0)
        cache.get(1)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["capacity"] == 4
        assert stats["hit_rate"] == 0.5


class TestExpertCacheBoundaryCases:
    """Test boundary cases."""

    def test_cache_size_equals_num_experts(self):
        """Test cache size equals number of experts."""
        cache = ExpertCache(capacity=8)
        for i in range(8):
            cache.put(i, {"state": f"expert{i}"})
        assert cache.size == 8
        assert cache.hit_rate == 0.0

    def test_cache_size_greater_than_num_experts(self):
        """Test cache size greater than number of experts."""
        cache = ExpertCache(capacity=16)
        for i in range(8):
            cache.put(i, {"state": f"expert{i}"})
        assert cache.size == 8

    def test_put_same_expert_repeatedly(self, cache):
        """Test putting same expert repeatedly."""
        for _ in range(10):
            cache.put(0, {"state": "expert0"})
        assert cache.size == 1
        assert cache.hit_rate == 0.0

    def test_get_after_clear(self, cache):
        """Test get after clear returns None."""
        cache.put(0, {"state": "expert0"})
        cache.clear()
        result = cache.get(0)
        assert result is None
