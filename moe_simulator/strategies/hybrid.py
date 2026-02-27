"""Hybrid routing strategy.

Smart routing strategy that combines cache-aware and score-based selection:
- k1: Load new experts to cache (DDR path)
- k2: Use non-cached experts (PIM path)
- k3: Fill remaining from cache

Selection priority:
1. If expert in cache → k3 (update LRU)
2. If not in cache and k1 not full → k1 (load to cache, update LRU)
3. If not in cache and k1 full but k2 not full → k2 (no cache update)
4. If k1+k2 full → fill from cache (highest score, update LRU)
"""

from typing import List, Tuple

from moe_simulator.core.cache import ExpertCache
from moe_simulator.strategies.base import RoutingStrategy


class HybridStrategy(RoutingStrategy):
    """Hybrid routing strategy combining cache and score-based selection.

    This is the main smart routing strategy that optimally balances between
    DDR cache loading (k1), PIM path (k2), and cache hits (k3).
    """

    name: str = "hybrid"

    def select_experts(
        self,
        scores: List[float],
        cache: ExpertCache,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts using hybrid strategy.

        Sorts experts by score descending and distributes them across
        k1 (cache load), k2 (PIM), and k3 (cache hit) based on availability
        and limits.

        Args:
            scores: List of routing scores for each expert (higher = preferred).
            cache: ExpertCache instance tracking cached experts.
            K: Total number of experts to select.
            k1_limit: Maximum number of experts to load to cache.
            k2_limit: Maximum number of experts from non-cached pool.

        Returns:
            Tuple of (k1_list, k2_list, k3_list):
            - k1_list: Expert IDs to load via DDR (cache load)
            - k2_list: Expert IDs to use via PIM (no cache update)
            - k3_list: Expert IDs from cache to fill remaining
        """
        num_experts = len(scores)

        # Sort experts by score descending
        sorted_experts = sorted(
            range(num_experts), key=lambda i: scores[i], reverse=True
        )

        k1_list: List[int] = []
        k2_list: List[int] = []
        k3_list: List[int] = []

        for expert in sorted_experts:
            # Stop when we've selected K experts
            current_total = len(k1_list) + len(k2_list) + len(k3_list)
            if current_total >= K:
                break

            if cache.contains(expert):
                # Expert in cache → k3 (cache hit)
                k3_list.append(expert)
                cache.update_lru(expert)
            else:
                # Expert not in cache
                if len(k1_list) < k1_limit:
                    # k1 not full → load to cache (DDR path)
                    k1_list.append(expert)
                    cache.put(expert, None)  # triggers eviction if needed
                    cache.update_lru(expert)
                elif len(k2_list) < k2_limit:
                    # k1 full but k2 not full → k2 (PIM path, no cache update)
                    k2_list.append(expert)
                else:
                    # Both k1 and k2 full → fill from cache
                    cached_experts = [
                        e
                        for e in cache._cache.keys()
                        if e not in k3_list and e not in k1_list
                    ]
                    if cached_experts:
                        # Select highest scoring cached expert
                        best = max(cached_experts, key=lambda e: scores[e])
                        k3_list.append(best)
                        cache.update_lru(best)

        return (k1_list, k2_list, k3_list)
