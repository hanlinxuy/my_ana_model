"""MoE Router Simulator - Core module for Mixture-of-Experts routing simulation.

This package provides core data structures for simulating expert routing
in mixture-of-experts models, including:
- ExpertCache: LRU cache management for expert caching
- RouterConfig: Configuration dataclass for router parameters
- TokenRouter: Core routing decision logic
"""

__version__ = "0.1.0"

from moe_simulator.core.cache import ExpertCache
from moe_simulator.core.config import RouterConfig
from moe_simulator.core.router import TokenRouter

__all__ = [
    "ExpertCache",
    "RouterConfig",
    "TokenRouter",
]
