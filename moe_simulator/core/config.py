"""Router configuration dataclass.

Defines all parameters for the MoE router simulation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RouterConfig:
    """Configuration for MoE router simulation.

    Contains all parameters needed to configure the router,
    expert cache, and routing strategy.

    Attributes:
        num_experts: Total number of available experts.
        K: Number of experts to select per token (top-K routing).
        cache_size: Maximum number of experts to keep in cache.
        k1: First threshold for load balancing strategy.
        k2: Second threshold for load balancing strategy.
        bandwidths: Optional per-expert bandwidth/cost values.
    """

    num_experts: int
    K: int
    cache_size: int
    k1: float
    k2: float
    bandwidths: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.K > self.num_experts:
            raise ValueError("K cannot exceed num_experts")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.k1 < 0 or self.k2 < 0:
            raise ValueError("k1 and k2 must be non-negative")
        if self.bandwidths is not None:
            if len(self.bandwidths) != self.num_experts:
                raise ValueError(
                    f"bandwidths length ({len(self.bandwidths)}) "
                    f"must match num_experts ({self.num_experts})"
                )
            if any(b < 0 for b in self.bandwidths):
                raise ValueError("bandwidths must be non-negative")

    def get_bandwidth(self, expert_id: int) -> float:
        """Get bandwidth for a specific expert.

        Args:
            expert_id: Identifier of the expert.

        Returns:
            Bandwidth cost for the expert, or 1.0 if not specified.
        """
        if self.bandwidths is not None:
            return self.bandwidths[expert_id]
        return 1.0
