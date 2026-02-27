"""Expert cache implementation with LRU eviction policy.

Provides an LRU cache for managing expert states in MoE routing simulation.
Tracks hit/miss rates for performance analysis.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExpertCache:
    """LRU cache for expert states with hit/miss tracking.

    Implements a Least Recently Used eviction policy for caching
    expert states. Tracks cache hits and misses for performance analysis.

    Attributes:
        capacity: Maximum number of experts that can be cached.
        _cache: Internal OrderedDict storing expert states.
        hits: Number of cache hits since initialization.
        misses: Number of cache misses since initialization.
    """

    capacity: int
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    hits: int = field(default=0, repr=False)
    misses: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate cache capacity."""
        if self.capacity <= 0:
            raise ValueError("Cache capacity must be positive")

    def get(self, expert_id: int) -> Optional[Any]:
        """Retrieve expert state from cache.

        Updates LRU order on hit.

        Args:
            expert_id: Identifier of the expert to retrieve.

        Returns:
            Cached state if found, None otherwise.
        """
        if expert_id in self._cache:
            self.hits += 1
            self._cache.move_to_end(expert_id)
            return self._cache[expert_id]
        self.misses += 1
        return None

    def put(self, expert_id: int, state: Any) -> None:
        """Store expert state in cache.

        Evicts least recently used expert if cache is full.

        Args:
            expert_id: Identifier of the expert.
            state: State to cache for the expert.
        """
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
        else:
            if len(self._cache) >= self.capacity:
                self.evict()
        self._cache[expert_id] = state

    def evict(self) -> Optional[int]:
        """Evict least recently used expert from cache.

        Returns:
            Expert ID that was evicted, or None if cache is empty.
        """
        if self._cache:
            evicted_id, _ = self._cache.popitem(last=False)
            return evicted_id
        return None

    def update_lru(self, expert_id: int) -> None:
        """Update LRU order for an expert.

        Moves the specified expert to the end (most recently used).

        Args:
            expert_id: Identifier of the expert to update.
        """
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)

    def contains(self, expert_id: int) -> bool:
        """Check if expert is in cache.

        Args:
            expert_id: Identifier of the expert.

        Returns:
            True if expert is cached, False otherwise.
        """
        return expert_id in self._cache

    def clear(self) -> None:
        """Clear all cached experts and reset hit/miss counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_cached_experts(self) -> List[int]:
        """Get list of all cached expert IDs.

        Returns:
            List of expert IDs currently in cache.
        """
        return list(self._cache.keys())

    @property
    def size(self) -> int:
        """Current number of cached experts."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction.

        Returns:
            Hit rate between 0.0 and 1.0, or 0.0 if no requests made.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, capacity, and hit_rate.
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "capacity": self.capacity,
            "hit_rate": self.hit_rate,
        }
