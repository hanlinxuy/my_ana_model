# MoE Router Simulator

A simulation framework for evaluating different routing strategies in Mixture-of-Experts (MoE) architectures with multi-level storage hierarchies (DDR + Flash/PIM).

## Overview

This project simulates various expert routing strategies for MoE models under different storage configurations:

- **DDR + NPU**: High-bandwidth path for compute-intensive experts
- **Flash + PIM**: Lower-bandwidth path for memory-bound experts  
- **DDR Cache**: LRU-managed cache for expert reuse

### Key Features

- **Multiple Routing Strategies**: LRU, PIM-Only, Hybrid, Fixed-Split, Cache-Only
- **Configurable Bandwidth Model**: Normalized bandwidth ratios (Flash→DDR, DDR→NPU, Flash→PIM)
- **Flexible Workload Patterns**: Adjustable smoothness for temporal locality
- **Comprehensive Metrics**: Latency, cache hit rate, k1/k2/k3 distribution

## Installation

```bash
# Clone the repository
git clone git@github.com:hanlinxuy/my_ana_model.git
cd my_ana_model

# Install dependencies
pip install -e moe_simulator/
# Or use PYTHONPATH
export PYTHONPATH=.
```

## Quick Start

```bash
# Run with default config (all strategies, smoothness levels)
PYTHONPATH=. python3 moe_simulator/main.py

# Run specific strategies
PYTHONPATH=. python3 moe_simulator/main.py --strategies lru,hybrid,pim-only

# Custom config
PYTHONPATH=. python3 moe_simulator/main.py --config examples/config.yaml --output results.csv
```

## CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | built-in defaults |
| `--strategies` | Comma-separated strategy list | all strategies |
| `--smoothness` | Comma-separated smoothness levels | 0.0, 0.5, 0.9, 0.99, 1.0 |
| `--output` | CSV output path | results.csv |
| `--num-tokens` | Number of tokens to simulate | 1000 |
| `--num-experts` | Total number of experts | 128 |
| `--K` | Experts selected per token | 8 |
| `--cache-size` | DDR cache capacity | 32 |
| `--k1` | DDR load budget | 3 |
| `--k2` | PIM compute budget | 2 |
| `--quiet` | Suppress console output | false |

## Architecture

### Storage Hierarchy

```
┌─────────────────────────────────────────┐
│               Flash Storage              │
│         (Slow, Large Capacity)          │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
   Load to DDR    Direct to PIM
        │             │
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│     DDR      │ │     PIM      │
│  (Fast RAM)  │ │ (In-Memory)  │
└──────┬───────┘ └──────────────┘
       │
       ▼
┌──────────────┐
│     NPU      │
│  (Compute)   │
└──────────────┘
```

### Bandwidth Model (Normalized)

| Path | Bandwidth | Latency per Expert |
|------|----------|-------------------|
| Flash → DDR → NPU | 1 + 8 | 1.125 |
| Flash → PIM | 4 | 0.25 |
| DDR → NPU (cache hit) | 8 | 0.125 |

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| **LRU** | All experts loaded to DDR, pure LRU management |
| **PIM-Only** | All experts via Flash→PIM, no DDR caching |
| **Hybrid** | Smart allocation: k1→DDR, k2→PIM, k3→cache |
| **Fixed-Split** | Fixed position split (first k1→DDR, next k2→PIM) |
| **Cache-Only** | Only use cached experts, rest via PIM |

### Latency Calculation

For each token, the latency is the **maximum** of parallel paths:

```
latency = max(k1 × 1.125, k2 × 0.25, k3 × 0.125)
```

Where:
- k1 = number of experts via Flash→DDR→NPU
- k2 = number of experts via Flash→PIM  
- k3 = number of experts via DDR cache hit

## Project Structure

```
pim_estimation/
├── moe_simulator/
│   ├── core/
│   │   ├── cache.py          # ExpertCache (LRU)
│   │   ├── config.py         # RouterConfig
│   │   ├── latency.py        # BandwidthModel, LatencyCalculator
│   │   ├── config_loader.py  # YAML/JSON loader
│   │   ├── runner.py         # Simulation runner
│   │   └── results.py        # Results aggregation
│   ├── strategies/
│   │   ├── base.py          # RoutingStrategy (ABC)
│   │   ├── factory.py       # StrategyFactory
│   │   ├── lru.py           # LRUStrategy
│   │   ├── pim_only.py      # PIMOnlyStrategy
│   │   ├── hybrid.py        # HybridStrategy
│   │   ├── fixed_split.py   # FixedSplitStrategy
│   │   └── cache_only.py   # CacheOnlyStrategy
│   └── main.py              # CLI entry point
├── tests/                   # Test suite (111 tests)
├── examples/
│   └── config.yaml         # Example configuration
└── AGENTS.md               # Agent guidelines
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_strategies.py -v

# Run with coverage
pytest tests/ --cov=moe_simulator
```

## Example Output

```
strategy   smoothness   avg_latency   cache_hit_rate
---------  -----------  ------------   -------------
lru        0.5         13.56          0.0%
lru        1.0         0.07           0.0%
hybrid     0.5         51.27          80.0%
hybrid     1.0         63.94          99.9%
pim_only   0.5         32.0           0.0%
pim_only   1.0         32.0           0.0%
```

## Original LRU Router

The `lru_router/` directory contains the original PyTorch implementation of the LRU-aware differentiable router. This is kept for reference but the new simulator (`moe_simulator/`) is the recommended way to evaluate routing strategies.

## License

MIT License

## Author

Hanlin Xu - hanlinxuy@gmail.com
