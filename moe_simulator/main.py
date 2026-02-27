"""Main entry point for MoE Simulator CLI.

Provides command-line interface for running routing strategy simulations.
"""

import argparse
import sys
from pathlib import Path

try:
    from moe_simulator.core import SimulationConfig, create_runner, load_config
except ModuleNotFoundError:
    # Fallback for running directly
    from core import SimulationConfig, create_runner, load_config


def parse_args(args=None):
    """Parse command-line arguments.

    Args:
        args: Arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="MoE Router Simulator - Run routing strategy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python -m moe_simulator

  # Run specific strategies
  python -m moe_simulator --strategies lru,pim-only,hybrid

  # Use custom config file
  python -m moe_simulator --config config.yaml

  # Save results to CSV
  python -m moe_simulator --output results.csv
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML or JSON config file",
    )

    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of strategies to run (default: all)",
    )

    parser.add_argument(
        "--smoothness",
        type=str,
        help="Comma-separated list of smoothness levels (default: from config)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to output CSV file",
    )

    parser.add_argument(
        "--num-tokens",
        type=int,
        help="Number of tokens to simulate (overrides config)",
    )

    parser.add_argument(
        "--num-experts",
        type=int,
        help="Number of experts (overrides config)",
    )

    parser.add_argument(
        "--K",
        type=int,
        help="Number of experts per token (overrides config)",
    )

    parser.add_argument(
        "--cache-size",
        type=int,
        help="Cache size (overrides config)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    return parser.parse_args(args)


def main(args=None):
    """Main entry point.

    Args:
        args: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success).
    """
    parsed = parse_args(args)

    # Load configuration
    if parsed.config:
        try:
            config = load_config(parsed.config)
        except FileNotFoundError:
            print(f"Error: Config file not found: {parsed.config}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1
    else:
        config = SimulationConfig()

    # Override config with command-line arguments
    if parsed.num_tokens is not None:
        config.num_tokens = parsed.num_tokens
    if parsed.num_experts is not None:
        config.num_experts = parsed.num_experts
    if parsed.K is not None:
        config.K = parsed.K
    if parsed.cache_size is not None:
        config.cache_size = parsed.cache_size

    # Parse strategies
    strategies = None
    if parsed.strategies:
        strategies = [s.strip() for s in parsed.strategies.split(",")]
        valid_strategies = [
            "lru",
            "pim_only",
            "pim-only",
            "cache_only",
            "cache-only",
            "fixed_split",
            "fixed-split",
            "hybrid",
        ]
        for s in strategies:
            if s not in valid_strategies:
                print(f"Warning: Unknown strategy '{s}'", file=sys.stderr)
        # Normalize strategy names
        strategies = [s.replace("-", "_") for s in strategies]

    # Parse smoothness levels
    smoothness_levels = None
    if parsed.smoothness:
        try:
            smoothness_levels = [float(s) for s in parsed.smoothness.split(",")]
        except ValueError:
            print("Error: Invalid smoothness level", file=sys.stderr)
            return 1

    # Print configuration
    if not parsed.quiet:
        print("=" * 60)
        print("MoE Router Simulator")
        print("=" * 60)
        print(f"Num experts: {config.num_experts}")
        print(f"K (experts/token): {config.K}")
        print(f"Cache size: {config.cache_size}")
        print(f"k1 limit: {config.k1}")
        print(f"k2 limit: {config.k2}")
        print(f"Num tokens: {config.num_tokens}")
        print(
            f"Bandwidths: flash->DDR={config.flash_to_ddr}, "
            f"DDR->NPU={config.ddr_to_npu}, flash->PIM={config.flash_to_pim}"
        )
        print(f"Strategies: {strategies or 'all'}")
        print(f"Smoothness levels: {smoothness_levels or config.smoothness_levels}")
        print("=" * 60)

    # Create runner and run simulation
    runner = create_runner(config)
    aggregator = runner.run(strategies=strategies, smoothness_levels=smoothness_levels)

    # Print results table
    if not parsed.quiet:
        print("\nResults:")
        print(aggregator.get_summary_table())

    # Write CSV output
    if parsed.output:
        aggregator.to_csv(parsed.output)
        if not parsed.quiet:
            print(f"\nResults written to: {parsed.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
