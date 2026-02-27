"""LRU (Least Recently Used) routing strategy.

Loads all requested experts to DDR cache. Uses LRU eviction when cache is full.
No PIM path used - all traffic goes through DDR.
"""

from typing import List, Tuple

from moe_simulator.core.cache import ExpertCache
from moe_simulator.strategies.base import RoutingStrategy


class LRUStrategy(RoutingStrategy):
    """LRU-based routing strategy.

    Loads all experts to DDR cache. On cache miss, loads the expert to cache
    (evicting LRU if full). No PIM path is used.
    """

    name: str = "lru"

    def select_experts(
        self,
        scores: List[float],
        cache: ExpertCache,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts using LRU strategy.

        All experts are loaded to DDR cache (k1). No PIM path (k2=0).
        Cache miss triggers load and LRU update.

        Args:
            scores: Routing scores for each expert.
            cache: ExpertCache instance.
            K: Number of experts to select.
            k1_limit: Maximum experts from cache (unused - we load all to cache).
            k2_limit: Maximum experts from PIM (unused - we don't use PIM).

        Returns:
            Tuple of (k1_list, k2_list, k3_list):
            - k1_list: All experts to load via DDR
            - k2_list: Empty (no PIM path)
            - k3_list: Empty (all handled via k1)
        """
        num_experts = len(scores)
        top_k_indices = sorted(
            range(num_experts), key=lambda i: scores[i], reverse=True
        )[:K]

        # Find experts not in cache
        not_in_cache = [idx for idx in top_k_indices if not cache.contains(idx)]
        not_in_cache_count = len(not_in_cache)

        # Load all to DDR: k1 = min(K, not_in_cache_count)
        k1 = min(K, not_in_cache_count)
        k1_list = not_in_cache[:k1]

        # Update LRU on every access (both hits and misses)
        for idx in top_k_indices:
            if cache.contains(idx):
                cache.update_lru(idx)
            else:
                # Cache miss - load to cache and update LRU
                cache.put(idx, None)
                cache.update_lru(idx)

        # No PIM path
        k2_list: List[int] = []
        k3_list: List[int] = []

        return (k1_list, k2_list, k3_list)
