"""Results aggregator for simulation output.

Provides data structures and CSV output for simulation results.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SimulationResult:
    """Result of a single simulation run.

    Contains metrics for a specific strategy at a specific
    smoothness level.
    """

    strategy: str
    smoothness: float
    num_tokens: int

    # Routing distribution metrics
    k1_count: int = 0  # Cache miss path (DDR)
    k2_count: int = 0  # PIM path
    k3_count: int = 0  # Cache hit path

    # Latency metrics
    avg_latency: float = 0.0
    total_latency: float = 0.0

    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_miss_count: int = 0

    # Additional metrics
    k1_ratio: float = 0.0
    k2_ratio: float = 0.0
    k3_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "strategy": self.strategy,
            "smoothness": self.smoothness,
            "num_tokens": self.num_tokens,
            "k1_count": self.k1_count,
            "k2_count": self.k2_count,
            "k3_count": self.k3_count,
            "k1_ratio": self.k1_ratio,
            "k2_ratio": self.k2_ratio,
            "k3_ratio": self.k3_ratio,
            "avg_latency": self.avg_latency,
            "total_latency": self.total_latency,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_miss_count": self.cache_miss_count,
        }

    @property
    def K(self) -> int:
        """Get total K (experts per token)."""
        return self.k1_count + self.k2_count + self.k3_count


@dataclass
class ResultsAggregator:
    """Aggregates and manages simulation results.

    Collects results from multiple simulation runs and provides
    output in various formats.
    """

    results: List[SimulationResult] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None

    def add_result(self, result: SimulationResult) -> None:
        """Add a result to the aggregator.

        Args:
            result: SimulationResult to add.
        """
        self.results.append(result)

    def add_results(self, results: List[SimulationResult]) -> None:
        """Add multiple results.

        Args:
            results: List of SimulationResult to add.
        """
        self.results.extend(results)

    def get_results_for_strategy(self, strategy: str) -> List[SimulationResult]:
        """Get results for a specific strategy.

        Args:
            strategy: Name of the strategy.

        Returns:
            List of results for the strategy.
        """
        return [r for r in self.results if r.strategy == strategy]

    def get_results_for_smoothness(self, smoothness: float) -> List[SimulationResult]:
        """Get results for a specific smoothness level.

        Args:
            smoothness: Smoothness level.

        Returns:
            List of results for the smoothness level.
        """
        return [r for r in self.results if r.smoothness == smoothness]

    def to_csv(self, path: Union[str, Path]) -> None:
        """Write results to CSV file.

        Args:
            path: Path to output CSV file.
        """
        if not self.results:
            return

        path = Path(path)
        fieldnames = list(self.results[0].to_dict().keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        """Convert results to list of dictionaries.

        Returns:
            List of result dictionaries.
        """
        return [r.to_dict() for r in self.results]

    def get_summary_table(self) -> str:
        """Generate a formatted summary table for console output.

        Returns:
            Formatted table string.
        """
        if not self.results:
            return "No results to display."

        # Group by strategy
        strategies = sorted(set(r.strategy for r in self.results))
        smoothness_levels = sorted(set(r.smoothness for r in self.results))

        # Build table
        lines = []

        # Header
        header = (
            f"{'Strategy':<15} {'Smoothness':>10} "
            f"{'K1':>6} {'K2':>6} {'K3':>6} "
            f"{'Avg Latency':>12} {'Cache Hit%':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        # Sort results by strategy then smoothness
        sorted_results = sorted(self.results, key=lambda r: (r.strategy, r.smoothness))

        for result in sorted_results:
            line = (
                f"{result.strategy:<15} {result.smoothness:>10.2f} "
                f"{result.k1_count:>6} {result.k2_count:>6} {result.k3_count:>6} "
                f"{result.avg_latency:>12.4f} {result.cache_hit_rate:>11.2f}%"
            )
            lines.append(line)

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self.config = None

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)

    def __bool__(self) -> bool:
        """Return True if there are results."""
        return bool(self.results)
