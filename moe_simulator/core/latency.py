"""Bandwidth model and latency calculation for MoE simulator.

Provides classes for modeling memory bandwidth and computing
token processing latency based on routing decisions and cache state.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BandwidthModel:
    """Model for memory bandwidth between different storage types.

    Stores normalized bandwidth values representing data transfer
    costs between Flash memory, DDR, NPU, and PIM modules.

    Attributes:
        flash_to_ddr: Bandwidth from Flash to DDR (normalized, default=1).
        ddr_to_npu: Bandwidth from DDR to NPU (normalized, default=8).
        flash_to_pim: Bandwidth from Flash to PIM (normalized, default=4).
    """

    flash_to_ddr: float = 1.0
    ddr_to_npu: float = 8.0
    flash_to_pim: float = 4.0

    def __post_init__(self) -> None:
        """Validate bandwidth values."""
        if self.flash_to_ddr < 0:
            raise ValueError("flash_to_ddr must be non-negative")
        if self.ddr_to_npu < 0:
            raise ValueError("ddr_to_npu must be non-negative")
        if self.flash_to_pim < 0:
            raise ValueError("flash_to_pim must be non-negative")

    def to_dict(self) -> Dict[str, float]:
        """Convert bandwidth model to dictionary.

        Returns:
            Dictionary mapping bandwidth path names to values.
        """
        return {
            "flash_to_ddr": self.flash_to_ddr,
            "ddr_to_npu": self.ddr_to_npu,
            "flash_to_pim": self.flash_to_pim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "BandwidthModel":
        """Create BandwidthModel from dictionary.

        Args:
            data: Dictionary with bandwidth values.

        Returns:
            New BandwidthModel instance.
        """
        return cls(
            flash_to_ddr=data.get("flash_to_ddr", 1.0),
            ddr_to_npu=data.get("ddr_to_npu", 8.0),
            flash_to_pim=data.get("flash_to_pim", 4.0),
        )


class LatencyCalculator:
    """Calculator for token processing latency in MoE system.

    Computes latency based on routing decisions, cache state, and
    bandwidth model. Supports configurable latency coefficients.

    The latency formula is:
        max(k1 * (flash_to_ddr + ddr_to_npu),
            k2 * flash_to_pim,
            k3 * ddr_to_npu)

    Where:
        - k1: Coefficient for cache miss path (Flash → DDR → NPU)
        - k2: Coefficient for PIM path (Flash → PIM)
        - k3: Coefficient for cached path (DDR → NPU)
    """

    def __init__(
        self,
        bandwidth_model: Optional[BandwidthModel] = None,
        k1: float = 1.0,
        k2: float = 1.0,
        k3: float = 1.0,
    ) -> None:
        """Initialize latency calculator.

        Args:
            bandwidth_model: Bandwidth model for latency calculation.
                           Uses default if None.
            k1: Coefficient for cache miss latency component.
            k2: Coefficient for PIM path latency component.
            k3: Coefficient for cached path latency component.
        """
        self.bandwidth_model = bandwidth_model or BandwidthModel()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def calculate(
        self,
        k1: float,
        k2: float,
        k3: float,
    ) -> float:
        """Calculate token latency based on routing coefficients.

        Computes the maximum latency across all possible paths,
        representing the bottleneck in the system.

        Args:
            k1: Coefficient for cache miss path (Flash → DDR → NPU).
                Represents number of tokens requiring full load from Flash.
            k2: Coefficient for PIM path (Flash → PIM).
                Represents number of tokens processed by PIM.
            k3: Coefficient for cached path (DDR → NPU).
                Represents number of tokens with experts in cache.

        Returns:
            Maximum latency across all paths.
        """
        # Cache miss path: Flash → DDR → NPU
        cache_miss_latency = k1 * (
            self.bandwidth_model.flash_to_ddr + self.bandwidth_model.ddr_to_npu
        )

        # PIM path: Flash → PIM
        pim_latency = k2 * self.bandwidth_model.flash_to_pim

        # Cached path: DDR → NPU (only if cache hit)
        cached_latency = k3 * self.bandwidth_model.ddr_to_npu

        # Return maximum latency (bottleneck)
        return max(cache_miss_latency, pim_latency, cached_latency)

    def calculate_with_defaults(self) -> float:
        """Calculate latency using stored default coefficients.

        Returns:
            Latency computed with self.k1, self.k2, self.k3.
        """
        return self.calculate(self.k1, self.k2, self.k3)

    def calculate_boundary(
        self,
        k1: float,
        k2: float,
        k3: float,
        epsilon: float = 1e-9,
    ) -> float:
        """Calculate latency with boundary case handling.

        Handles edge cases where coefficients are zero by using
        a small epsilon value for comparison.

        Args:
            k1: Coefficient for cache miss path.
            k2: Coefficient for PIM path.
            k3: Coefficient for cached path.
            epsilon: Small value for float comparison.

        Returns:
            Maximum latency, or 0.0 if all inputs are effectively zero.
        """
        # Handle boundary cases
        effective_k1 = k1 if k1 > epsilon else 0.0
        effective_k2 = k2 if k2 > epsilon else 0.0
        effective_k3 = k3 if k3 > epsilon else 0.0

        # If all coefficients are zero, return 0
        if effective_k1 == 0.0 and effective_k2 == 0.0 and effective_k3 == 0.0:
            return 0.0

        return self.calculate(effective_k1, effective_k2, effective_k3)

    def get_latency_components(
        self,
        k1: float,
        k2: float,
        k3: float,
    ) -> Dict[str, float]:
        """Get individual latency components for analysis.

        Args:
            k1: Coefficient for cache miss path.
            k2: Coefficient for PIM path.
            k3: Coefficient for cached path.

        Returns:
            Dictionary with individual latency components.
        """
        return {
            "cache_miss": k1
            * (self.bandwidth_model.flash_to_ddr + self.bandwidth_model.ddr_to_npu),
            "pim": k2 * self.bandwidth_model.flash_to_pim,
            "cached": k3 * self.bandwidth_model.ddr_to_npu,
        }

    def set_coefficients(
        self,
        k1: Optional[float] = None,
        k2: Optional[float] = None,
        k3: Optional[float] = None,
    ) -> None:
        """Update latency coefficients.

        Args:
            k1: New k1 coefficient, or None to keep current.
            k2: New k2 coefficient, or None to keep current.
            k3: New k3 coefficient, or None to keep current.
        """
        if k1 is not None:
            self.k1 = k1
        if k2 is not None:
            self.k2 = k2
        if k3 is not None:
            self.k3 = k3

    def __repr__(self) -> str:
        """String representation of latency calculator."""
        return (
            f"LatencyCalculator("
            f"bandwidth={self.bandwidth_model.to_dict()}, "
            f"k1={self.k1}, k2={self.k2}, k3={self.k3})"
        )
