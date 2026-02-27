"""PIM-Only routing strategy.

Routes all experts through the Flash→PIM path (k2_list), bypassing cache.
"""

from typing import List, Tuple

from moe_simulator.strategies.base import RoutingStrategy
from moe_simulator.core.cache import ExpertCache


class PIMOnlyStrategy(RoutingStrategy):
    """PIM-Only routing strategy.

    Routes all selected experts through the Flash→PIM path (k2_list).
    Does not use cached experts (k1_list) or fallback experts (k3_list).
    Does not update the LRU cache.
    """

    name: str = "pim_only"

    def select_experts(
        self,
        scores: List[float],
        cache: ExpertCache,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts using PIM-Only path.

        All experts go through k2_list (Flash→PIM), none through k1 or k3.
        This strategy ignores the cache and always routes through PIM.

        Args:
            scores: List of routing scores for each expert (unused).
            cache: ExpertCache instance (unused - PIM-only doesn't use cache).
            K: Number of experts to select.
            k1_limit: Maximum k1 experts (ignored).
            k2_limit: Maximum k2 experts (ignored, uses K instead).

        Returns:
            Tuple of (k1_list, k2_list, k3_list):
            - k1_list: Empty list (no cache hits)
            - k2_list: Top K expert IDs (PIM path)
            - k3_list: Empty list (no fallback)
        """
        # All experts go through PIM path (k2_list)
        k1_list: List[int] = []
        k3_list: List[int] = []

        # Select top K experts for PIM path
        expert_ids = list(range(len(scores)))
        # Sort by score descending (higher is better)
        sorted_experts = sorted(expert_ids, key=lambda i: scores[i], reverse=True)
        k2_list: List[int] = sorted_experts[:K]

        return (k1_list, k2_list, k3_list)
