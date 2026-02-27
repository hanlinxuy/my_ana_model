"""Cache-Only routing strategy.

Routes only cached experts (k1_list), bypassing Flash→PIM path.
If more experts needed beyond cache hits, they also go through k2 (PIM path).
"""

from typing import List, Tuple

from moe_simulator.strategies.base import RoutingStrategy
from moe_simulator.core.cache import ExpertCache


class CacheOnlyStrategy(RoutingStrategy):
    """Cache-Only routing strategy.

    Only uses cached experts (k1_list). If cache_hit_count >= K, selects top-K
    from cache. If cache_hit_count < K, fills remaining from PIM (k2_list).

    This is a conservative strategy that only uses what's already in cache,
    avoiding Flash loading latency.
    """

    name: str = "cache_only"

    def select_experts(
        self,
        scores: List[float],
        cache: ExpertCache,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts using Cache-Only path.

        k1 = 0 (never load from Flash to DDR)
        k3 = min(K, cache_hit_count) - experts already in cache
        k2 = K - k3 - remaining experts go to PIM

        Args:
            scores: List of routing scores for each expert (higher = preferred).
            cache: ExpertCache instance tracking cached experts.
            K: Number of experts to select.
            k1_limit: Maximum k1 experts (ignored - k1=0 for cache-only).
            k2_limit: Maximum k2 experts (used for remaining capacity).

        Returns:
            Tuple of (k1_list, k2_list, k3_list):
            - k1_list: Empty list (k1=0, no load from Flash)
            - k2_list: Remaining experts to fill capacity (PIM path)
            - k3_list: Cached experts selected (cache hits)
        """
        # k1 = 0: Never load from Flash to DDR
        k1_list: List[int] = []

        # Get cached expert IDs
        cached_expert_ids = cache.get_cached_experts()
        cache_hit_count = len(cached_expert_ids)

        # k3 = min(K, cache_hit_count) - experts already in cache
        k3_count = min(K, cache_hit_count)

        # Get experts sorted by score (descending)
        expert_ids = list(range(len(scores)))
        sorted_experts = sorted(expert_ids, key=lambda i: scores[i], reverse=True)

        # Select top K experts from cache (k3_list)
        cached_sorted = [e for e in sorted_experts if e in cached_expert_ids]
        k3_list: List[int] = cached_sorted[:k3_count]

        # k2 = K - k3: remaining experts go to PIM
        k2_needed = K - k3_count

        # Select remaining experts from non-cached pool for PIM path
        non_cached_sorted = [e for e in sorted_experts if e not in cached_expert_ids]
        k2_list: List[int] = non_cached_sorted[:k2_needed]

        return (k1_list, k2_list, k3_list)
