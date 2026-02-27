"""MoE routing strategies package.

This module provides the interface and factory for expert routing strategies
in the Mixture of Experts simulator.
"""

from moe_simulator.strategies.base import RoutingStrategy
from moe_simulator.strategies.factory import StrategyFactory

__all__ = ["RoutingStrategy", "StrategyFactory"]
