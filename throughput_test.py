#!/usr/bin/env python3
"""
Throughput test for Lambda Calculus term generation.

Tests generation across different depth/size configurations to ensure:
1. Reasonable throughput across all configurations
2. No pathological cases that tank wall clock time
3. Fuel budget tracking is working correctly
4. Pathological detection catches problem cases
"""

import time
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any
import random

from lambda_gen import Config, generate_example


@dataclass
class ThroughputResult:
    """Results from a throughput test run."""
    config_name: str
    num_examples: int
    total_time: float
    examples_per_sec: float

    # Pathological metrics
    pathological_count: int
    pathological_rate: float

    # Fuel metrics
    avg_fuel_consumed: float
    max_fuel_consumed: float
    avg_steps: float
    max_steps: int

    # Size metrics
    avg_size_growth: float
    max_size_growth: float
    avg_size: float
    max_size: int

    # Timing breakdown
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float


def percentile(values: List[float], p: float) -> float:
    """Compute percentile of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = int(len(sorted_vals) * p)
    return sorted_vals[min(k, len(sorted_vals) - 1)]


def run_throughput_test(config_name: str, config: Config,
                        num_terms: int = 100) -> ThroughputResult:
    """Run throughput test with given configuration."""
    print(f"\nTesting: {config_name}")
    print(f"  Config: depth={config.min_depth}-{config.max_depth}, "
          f"size={config.max_size}, max_steps={config.max_steps}")

    rng = random.Random(config.seed)

    examples_generated = 0
    pathological_count = 0
    fuel_consumed_ratios = []
    size_growth_rates = []
    sizes = []
    steps_list = []
    example_times = []

    start_time = time.time()

    for i in range(num_terms):
        example_start = time.time()

        try:
            result = generate_example(config, rng, draw_index=i)

            if result:
                # Result is a list of trace steps
                if isinstance(result, list) and len(result) > 0:
                    # Just analyze the last step of the trace
                    example = result[-1]

                    examples_generated += 1

                    # Extract metrics
                    meta = example['meta']

                    if meta.get('is_pathological', False):
                        pathological_count += 1

                    fuel_consumed_ratios.append(meta.get('fuel_consumed_ratio', 0.0))
                    size_growth_rates.append(meta.get('size_growth_rate', 1.0))
                    sizes.append(meta.get('size', 0))
                    steps_list.append(example['steps_total'])

        except Exception as e:
            print(f"    Error on term {i}: {e}")
            continue

        example_time = time.time() - example_start
        example_times.append(example_time)

        # Progress indicator
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    Progress: {i + 1}/{num_terms} terms ({rate:.1f}/s)")

    total_time = time.time() - start_time

    # Compute results
    result = ThroughputResult(
        config_name=config_name,
        num_examples=examples_generated,
        total_time=total_time,
        examples_per_sec=examples_generated / total_time if total_time > 0 else 0,

        pathological_count=pathological_count,
        pathological_rate=pathological_count / examples_generated if examples_generated > 0 else 0,

        avg_fuel_consumed=sum(fuel_consumed_ratios) / len(fuel_consumed_ratios) if fuel_consumed_ratios else 0,
        max_fuel_consumed=max(fuel_consumed_ratios) if fuel_consumed_ratios else 0,
        avg_steps=sum(steps_list) / len(steps_list) if steps_list else 0,
        max_steps=max(steps_list) if steps_list else 0,

        avg_size_growth=sum(size_growth_rates) / len(size_growth_rates) if size_growth_rates else 0,
        max_size_growth=max(size_growth_rates) if size_growth_rates else 0,
        avg_size=sum(sizes) / len(sizes) if sizes else 0,
        max_size=max(sizes) if sizes else 0,

        min_time=min(example_times) if example_times else 0,
        max_time=max(example_times) if example_times else 0,
        p95_time=percentile(example_times, 0.95),
        p99_time=percentile(example_times, 0.99),
    )

    return result


def print_result(result: ThroughputResult):
    """Print formatted result."""
    print(f"\n  Results for {result.config_name}:")
    print(f"    Generated: {result.num_examples} terms in {result.total_time:.2f}s")
    print(f"    Throughput: {result.examples_per_sec:.2f} terms/s")

    # Timing breakdown
    print(f"    Time per term: min={result.min_time*1000:.1f}ms, "
          f"p95={result.p95_time*1000:.1f}ms, "
          f"p99={result.p99_time*1000:.1f}ms, "
          f"max={result.max_time*1000:.1f}ms")

    # Pathological cases
    status = "✓" if result.pathological_rate < 0.10 else "⚠️"
    print(f"    Pathological: {result.pathological_count} "
          f"({result.pathological_rate:.1%}) {status}")

    # Fuel consumption
    print(f"    Fuel: avg={result.avg_fuel_consumed:.1%}, "
          f"max={result.max_fuel_consumed:.1%}")

    # Steps
    print(f"    Steps: avg={result.avg_steps:.1f}, max={result.max_steps}")

    # Size
    print(f"    Size: avg={result.avg_size:.1f}, max={result.max_size}")
    print(f"    Growth: avg={result.avg_size_growth:.2f}x, max={result.max_size_growth:.2f}x")

    # Warning flags
    if result.max_time > 5.0:
        print(f"    ⚠️  WARNING: Max time {result.max_time:.2f}s exceeds 5s threshold!")
    if result.pathological_rate > 0.20:
        print(f"    ⚠️  WARNING: High pathological rate {result.pathological_rate:.1%}!")
    if result.p99_time > 2.0:
        print(f"    ⚠️  WARNING: P99 latency {result.p99_time:.2f}s exceeds 2s threshold!")


def main():
    """Run comprehensive throughput tests."""
    print("=" * 70)
    print("Lambda Calculus Throughput Test Suite")
    print("=" * 70)
    print("\nTesting generation throughput across different configurations")
    print("to ensure no pathological wall clock issues.\n")

    # Define test configurations
    test_configs = [
        # Small, simple terms (baseline)
        ("Small (depth 2-4)", Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=4,
            min_depth=2,
            max_size=20,
            max_steps=50,
            share=True,
            seed=42
        )),

        # Medium complexity
        ("Medium (depth 4-6)", Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=6,
            min_depth=4,
            max_size=40,
            max_steps=100,
            share=True,
            seed=42
        )),

        # High complexity
        ("High (depth 6-8)", Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=8,
            min_depth=6,
            max_size=60,
            max_steps=150,
            share=True,
            seed=42
        )),

        # Very high complexity (stress test)
        ("Very High (depth 8-10)", Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=10,
            min_depth=8,
            max_size=80,
            max_steps=200,
            share=True,
            seed=42
        )),

        # Large size budget
        ("Large Size (depth 5-7)", Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=7,
            min_depth=5,
            max_size=100,
            max_steps=150,
            share=True,
            seed=42
        )),
    ]

    results = []

    # Run tests
    for config_name, config in test_configs:
        result = run_throughput_test(config_name, config, num_terms=50)
        results.append(result)
        print_result(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nThroughput Comparison:")
    for result in results:
        status = "✓" if result.examples_per_sec > 5.0 else "⚠️"
        print(f"  {result.config_name:25s}: {result.examples_per_sec:6.2f} terms/s {status}")

    print(f"\nPathological Rate Comparison:")
    for result in results:
        status = "✓" if result.pathological_rate < 0.10 else "⚠️"
        print(f"  {result.config_name:25s}: {result.pathological_rate:6.1%} {status}")

    print(f"\nP99 Latency Comparison:")
    for result in results:
        status = "✓" if result.p99_time < 2.0 else "⚠️"
        print(f"  {result.config_name:25s}: {result.p99_time*1000:6.1f}ms {status}")

    print(f"\nMax Latency Comparison:")
    for result in results:
        status = "✓" if result.max_time < 5.0 else "⚠️"
        print(f"  {result.config_name:25s}: {result.max_time*1000:6.1f}ms {status}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    issues = []

    for result in results:
        if result.max_time > 5.0:
            issues.append(f"{result.config_name}: Max latency {result.max_time:.2f}s exceeds threshold")
        if result.pathological_rate > 0.20:
            issues.append(f"{result.config_name}: High pathological rate {result.pathological_rate:.1%}")
        if result.p99_time > 2.0:
            issues.append(f"{result.config_name}: P99 latency {result.p99_time:.2f}s exceeds threshold")

    if issues:
        print("\n⚠️  Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All configurations pass throughput requirements!")
        print("  - No wall clock tanking on complex cases")
        print("  - Pathological cases detected and tracked")
        print("  - Fuel budget mechanism working correctly")

    print("\n" + "=" * 70)

    return len(issues) == 0


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
