"""Factory for creating routing strategy instances.

Provides a registry-based factory pattern for instantiating
routing strategies by name.
"""

from typing import Dict, Type, Optional

from moe_simulator.strategies.base import RoutingStrategy


class StrategyFactory:
    """Factory for creating routing strategy instances.

    Implements a registry pattern where strategies can be registered
    by name and later instantiated via the create method.

    Example:
        >>> from moe_simulator.strategies.factory import StrategyFactory
        >>> from moe_simulator.strategies.base import RoutingStrategy

        >>> class MyStrategy(RoutingStrategy):
        ...     name = "my_strategy"
        ...     def select_experts(self, scores, cache, K, k1_limit, k2_limit):
        ...         return ([], [], [])

        >>> factory = StrategyFactory()
        >>> factory.register("my_strategy", MyStrategy)
        >>> strategy = factory.create("my_strategy")
    """

    _registry: Dict[str, Type[RoutingStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[RoutingStrategy]) -> None:
        """Register a routing strategy class.

        Args:
            name: Unique identifier for the strategy.
            strategy_class: RoutingStrategy subclass to register.

        Raises:
            TypeError: If strategy_class is not a RoutingStrategy subclass.
            ValueError: If name is already registered.
        """
        if not issubclass(strategy_class, RoutingStrategy):
            raise TypeError(
                f"{strategy_class.__name__} must be a RoutingStrategy subclass"
            )
        if name in cls._registry:
            raise ValueError(f"Strategy '{name}' is already registered")
        cls._registry[name] = strategy_class

    @classmethod
    def create(cls, name: str) -> RoutingStrategy:
        """Create a routing strategy instance by name.

        Args:
            name: Identifier of the strategy to create.

        Returns:
            New instance of the requested strategy.

        Raises:
            KeyError: If no strategy is registered under the given name.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        return cls._registry[name]()

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names.

        Returns:
            List of registered strategy identifiers.
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies.

        Useful for testing or resetting the registry.
        """
        cls._registry.clear()
