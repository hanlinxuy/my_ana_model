# AGENTS.md - AI Agent Guidelines for This Repository

## Project Overview

This is a **MoE (Mixture of Experts) Router Simulator** project. It simulates different routing strategies for MoE architectures with multi-level storage (DDR + Flash/PIM).

### Directory Structure

```
pim_estimation/
├── lru_router/          # Original LRU router (PyTorch)
├── moe_simulator/       # New simulator (pure Python, numpy)
│   ├── core/            # Core modules (cache, config, latency)
│   └── strategies/      # Routing strategies
├── tests/               # Test suite (111 tests)
└── examples/            # Example configs
```

---

## Build, Lint, Test Commands

### Python Projects

```bash
# Install dependencies (for moe_simulator)
pip install -e .                    # or: pip install -e moe_simulator/

# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest                              # run all tests
pytest tests/                       # run specific test directory
pytest -v                          # verbose output

# Run a single test
pytest tests/test_cache.py::TestExpertCache::test_cache_put_get
pytest tests/test_strategies.py::TestHybridStrategy::test_hybrid_basic -v

# Run tests matching a pattern
pytest -k "latency"                # tests with 'latency' in name
pytest -k "test_hybrid"            # all hybrid strategy tests

# Coverage
pytest --cov=moe_simulator         # with coverage
pytest --cov=moe_simulator --cov-report=html

# Linting
ruff check .                       # check
ruff check --fix .                 # auto-fix

# Type checking (if mypy installed)
mypy moe_simulator/

# Format
ruff format .
```

### LRU Router (PyTorch)

```bash
cd lru_router
pip install -e .
pytest tests/                       # if tests exist
```

---

## Code Style Guidelines

### Imports

**Order imports by type:**

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import OrderedDict

# Third party
import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass, field

# Local application (use relative imports within package)
from moe_simulator.core.cache import ExpertCache
from moe_simulator.core.config import RouterConfig
```

**Avoid:**
- Wildcard imports: `from X import *`
- Importing too many modules in one line

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `cache.py`, `latency_model.py` |
| Classes | PascalCase | `ExpertCache`, `RouterConfig`, `HybridStrategy` |
| Functions/variables | snake_case | `def calculate_latency()`, `hit_rate`, `cache_miss` |
| Constants | UPPER_SNAKE_CASE | `MAX_EXPERTS = 128`, `DEFAULT_K = 8` |
| Private attributes | snake_case with prefix `_` | `_cache`, `_bandwidths` |

### Type Annotations

**Python:**

```python
from typing import Dict, List, Optional, Any

class ExpertCache:
    """Docstring with Args and Returns."""

    capacity: int
    hits: int = 0
    misses: int = 0

    def get(self, expert_id: int) -> Optional[Any]:
        """Get expert from cache.

        Args:
            expert_id: The expert identifier.

        Returns:
            Cached state if found, None otherwise.
        """
        ...

    def put(self, expert_id: int, state: Any) -> None:
        """Store expert in cache."""
        ...

# For complex return types, use typing.NamedTuple or TypedDict
from typing import TypedDict

class CacheStats(TypedDict):
    hits: int
    misses: int
    hit_rate: float
```

### Dataclasses

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class RouterConfig:
    """Configuration for MoE router simulation."""

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
        if self.K > self.num_experts:
            raise ValueError("K cannot exceed num_experts")
```

### Docstrings

**Use Google-style docstrings:**

```python
def calculate_latency(k1_count: int, k2_count: int, k3_count: int) -> float:
    """Calculate token latency based on expert routing.

    Uses the formula: latency = max(k1×1.125, k2×0.25, k3×0.125)
    where the coefficients come from bandwidth model.

    Args:
        k1_count: Number of experts routed through DDR (Flash→DDR→NPU).
        k2_count: Number of experts routed through PIM (Flash→PIM).
        k3_count: Number of experts from cache (DDR→NPU).

    Returns:
        Total latency for processing this token.

    Raises:
        ValueError: If any count is negative.
    """
    ...
```

### Error Handling

1. **Never suppress errors silently:**
   ```python
   # BAD
   try:
       result = compute()
   except Exception:
       pass

   # GOOD
   try:
       result = compute()
   except ValueError as e:
       logger.error(f"Invalid input: {e}")
       raise
   ```

2. **Use specific exceptions:**
   ```python
   # BAD
   if x < 0:
       raise Exception("Negative")

   # GOOD
   if x < 0:
       raise ValueError(f"Expected non-negative, got {x}")
   ```

3. **Validate inputs early:**
   ```python
   def __init__(self, capacity: int):
       if capacity <= 0:
           raise ValueError("Capacity must be positive")
       self.capacity = capacity
   ```

### File Organization

```
moe_simulator/
├── __init__.py              # Package exports
├── main.py                  # CLI entry point
├── core/
│   ├── __init__.py
│   ├── cache.py             # ExpertCache
│   ├── config.py            # RouterConfig
│   ├── latency.py           # BandwidthModel, LatencyCalculator
│   ├── config_loader.py     # YAML/JSON config loading
│   ├── runner.py            # Simulation runner
│   └── results.py           # Results aggregation
└── strategies/
    ├── __init__.py
    ├── base.py              # Abstract RoutingStrategy
    ├── factory.py           # StrategyFactory
    ├── lru.py               # LRUStrategy
    ├── pim_only.py          # PIMOnlyStrategy
    ├── hybrid.py            # HybridStrategy
    ├── fixed_split.py       # FixedSplitStrategy
    └── cache_only.py        # CacheOnlyStrategy
```

### Test Conventions

```python
import pytest
from moe_simulator.core.cache import ExpertCache

class TestExpertCache:
    """Tests for ExpertCache class."""

    def test_cache_put_get(self, cache):
        """Test basic put and get operations."""
        cache.put(1, {"data": "test"})
        result = cache.get(1)
        assert result == {"data": "test"}

    def test_cache_hit_rate(self, cache):
        """Test hit rate calculation."""
        cache.put(1, "a")
        cache.get(1)  # hit
        cache.get(2)  # miss
        assert cache.hit_rate == 0.5

    @pytest.mark.parametrize("capacity,expected", [
        (1, 1),
        (5, 5),
        (10, 10),
    ])
    def test_cache_capacity(self, capacity, expected):
        """Test cache respects capacity limits."""
        c = ExpertCache(capacity=capacity)
        for i in range(15):
            c.put(i, i)
        assert c.size == expected
```

### Constants and Configuration

**Default values (from config):**
```python
DEFAULT_NUM_EXPERTS = 128
DEFAULT_K = 8
DEFAULT_CACHE_SIZE = 32
DEFAULT_K1 = 3
DEFAULT_K2 = 2
DEFAULT_SMOOTHNESS_LEVELS = [0.0, 0.5, 0.9, 0.99, 1.0]

# Bandwidth model (normalized)
FLASH_TO_DDR = 1.0
DDR_TO_NPU = 8.0
FLASH_TO_PIM = 4.0
```

---

## Key Patterns

### Strategy Pattern

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def select_experts(
        self,
        scores: List[float],
        cache: Any,
        K: int,
        k1_limit: int,
        k2_limit: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Select experts and categorize them.

        Returns:
            (k1_list, k2_list, k3_list) - expert IDs for each category.
        """
        ...

# Usage
strategy = StrategyFactory.create("hybrid")
k1, k2, k3 = strategy.select_experts(scores, cache, K=8, k1_limit=3, k2_limit=2)
```

### Latency Calculation

```python
# Bandwidth model (normalized, expert size = 1)
latency_k1 = k1 * (flash_to_ddr + ddr_to_npu)  # 1.125
latency_k2 = k2 * flash_to_pim                  # 0.25
latency_k3 = k3 * ddr_to_npu                    # 0.125

token_latency = max(latency_k1, latency_k2, latency_k3)
```

---

## Common Tasks

### Running a Single Test

```bash
# By test function name
pytest tests/test_cache.py::test_cache_hit_rate -v

# By class and method
pytest tests/test_strategies.py::TestHybridStrategy::test_hybrid_basic -v

# With coverage for single test
pytest tests/test_latency.py::test_latency_formula -v --cov=moe_simulator.core.latency
```

### Adding a New Strategy

1. Create `strategies/new_strategy.py`:
   ```python
   from .base import RoutingStrategy

   class NewStrategy(RoutingStrategy):
       def select_experts(self, scores, cache, K, k1_limit, k2_limit):
           # Implementation
           return k1_list, k2_list, k3_list
   ```

2. Register in `strategies/__init__.py` or use factory

3. Add tests in `tests/test_strategies.py`

---

## Notes

- The `moe_simulator` package uses **pure Python** (no PyTorch)
- Tests are in the root `tests/` directory
- Use `PYTHONPATH=.` when running from root
- The original `lru_router/` uses PyTorch for differentiable routing
