"""Configuration loader for YAML/JSON config files.

Provides loading and validation of simulation configuration from
YAML or JSON files with support for default values.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from moe_simulator.core.config import RouterConfig
from moe_simulator.core.latency import BandwidthModel


@dataclass
class SimulationConfig:
    """Complete configuration for simulation run.

    Contains all parameters needed to configure the simulation,
    including router config, bandwidth model, and simulation settings.
    """

    # Router parameters (defaults from plan)
    num_experts: int = 128
    K: int = 8
    cache_size: int = 32
    k1: float = 3.0
    k2: float = 2.0

    # Bandwidth model
    flash_to_ddr: float = 1.0
    ddr_to_npu: float = 8.0
    flash_to_pim: float = 4.0

    # Simulation settings
    num_tokens: int = 1000
    smoothness_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.9, 0.99, 1.0]
    )
    random_seed: Optional[int] = 42

    def to_router_config(self) -> RouterConfig:
        """Convert to RouterConfig.

        Returns:
            RouterConfig instance.
        """
        return RouterConfig(
            num_experts=self.num_experts,
            K=self.K,
            cache_size=self.cache_size,
            k1=self.k1,
            k2=self.k2,
        )

    def to_bandwidth_model(self) -> BandwidthModel:
        """Convert to BandwidthModel.

        Returns:
            BandwidthModel instance.
        """
        return BandwidthModel(
            flash_to_ddr=self.flash_to_ddr,
            ddr_to_npu=self.ddr_to_npu,
            flash_to_pim=self.flash_to_pim,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "num_experts": self.num_experts,
            "K": self.K,
            "cache_size": self.cache_size,
            "k1": self.k1,
            "k2": self.k2,
            "flash_to_ddr": self.flash_to_ddr,
            "ddr_to_npu": self.ddr_to_npu,
            "flash_to_pim": self.flash_to_pim,
            "num_tokens": self.num_tokens,
            "smoothness_levels": self.smoothness_levels,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Create from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            SimulationConfig instance.
        """
        return cls(
            num_experts=data.get("num_experts", 128),
            K=data.get("K", 8),
            cache_size=data.get("cache_size", 32),
            k1=data.get("k1", 3.0),
            k2=data.get("k2", 2.0),
            flash_to_ddr=data.get("flash_to_ddr", 1.0),
            ddr_to_npu=data.get("ddr_to_npu", 8.0),
            flash_to_pim=data.get("flash_to_pim", 4.0),
            num_tokens=data.get("num_tokens", 1000),
            smoothness_levels=data.get("smoothness_levels", [0.0, 0.5, 0.9, 0.99, 1.0]),
            random_seed=data.get("random_seed", 42),
        )


def load_config(path: Union[str, Path]) -> SimulationConfig:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to config file.

    Returns:
        SimulationConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If file format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    with open(path, "r") as f:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    return SimulationConfig.from_dict(data)


def save_config(config: SimulationConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file.

    Args:
        config: SimulationConfig to save.
        path: Path to output file.
    """
    path = Path(path)
    data = config.to_dict()
    suffix = path.suffix.lower()

    with open(path, "w") as f:
        if suffix in (".yaml", ".yml"):
            yaml.dump(data, f, default_flow_style=False)
        elif suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
