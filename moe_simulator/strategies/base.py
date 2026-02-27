"""Abstract base class for MoE routing strategies.

Defines the interface that all routing strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any

from moe_simulator.core.cache import ExpertCache


class RoutingStrategy(ABC):
    """Abstract base class for expert routing strategies.

    Defines the interface for selecting experts in a Mixture of Experts
    system. All concrete routing strategies must implement the
    select_experts method.

    Attributes:
        name: Human-readable name of the routing strategy.
    """

    name: str = "BaseRoutingStrategy"

    @abstractmethod
    def select_experts(
        self,
        scores: List[float],
        cache: ExpertCache,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts based on routing scores and cache state.

        This is the core method that implements the routing logic.
        It selects experts using a tiered approach:
        - k1_list: Top K experts from cache (if available)
        - k2_list: Top K experts from non-cached, sorted by score
        - k3_list: Remaining experts to fill capacity

        Args:
            scores: List of routing scores for each expert (higher = preferred).
            cache: ExpertCache instance tracking cached experts.
            K: Number of experts to select for each tier.
            k1_limit: Maximum number of experts to select from cache.
            k2_limit: Maximum number of experts to select from non-cached.

        Returns:
            Tuple of (k1_list, k2_list, k3_list) where each is a list
            of expert IDs:
            - k1_list: Expert IDs selected from cache
            - k2_list: Expert IDs selected from non-cached pool
            - k3_list: Additional expert IDs to fill remaining capacity
        """
        ...
