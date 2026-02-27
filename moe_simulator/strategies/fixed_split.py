"""Fixed-Split routing strategy.

Fixed position-based split: first k1_limit experts → DDR,
next k2_limit experts → PIM, remaining → k3_list.

Ignores cache state (doesn't check if expert is in cache).
Baseline strategy for comparison.
"""

from typing import List, Tuple

from moe_simulator.core.cache import ExpertCache
from moe_simulator.strategies.base import RoutingStrategy


class FixedSplitStrategy(RoutingStrategy):
    """Fixed position-based routing strategy.

    Sorts experts by score descending, then splits by fixed position:
    - k1_list: First k1_limit experts (by position, not cache)
    - k2_list: Next k2_limit experts
    - k3_list: Remaining experts to fill capacity

    Does NOT check cache state - uses fixed position regardless of
    whether experts are already cached.
    """

    name: str = "fixed_split"

    def select_experts(
        self,
        scores: List[float],
        cache: ExpertCache,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts using fixed position split.

        Sorts experts by score descending, then assigns:
        - Positions 0 to k1_limit-1 → k1_list (DDR)
        - Positions k1_limit to k1_limit+k2_limit-1 → k2_list (PIM)
        - Remaining positions → k3_list

        Args:
            scores: List of routing scores for each expert (higher = preferred).
            cache: ExpertCache instance (ignored - cache state not checked).
            K: Number of experts to select for each tier.
            k1_limit: Maximum number of experts for DDR path.
            k2_limit: Maximum number of experts for PIM path.

        Returns:
            Tuple of (k1_list, k2_list, k3_list):
            - k1_list: Expert IDs for DDR path (first k1_limit by position)
            - k2_list: Expert IDs for PIM path (next k2_limit by position)
            - k3_list: Additional expert IDs (remaining)
        """
        num_experts = len(scores)

        # Sort by score descending to get ranking
        sorted_indices = sorted(
            range(num_experts), key=lambda i: scores[i], reverse=True
        )

        # Fixed position split
        k1_list = sorted_indices[:k1_limit]
        k2_list = sorted_indices[k1_limit : k1_limit + k2_limit]
        k3_list = sorted_indices[k1_limit + k2_limit :]

        return (k1_list, k2_list, k3_list)
