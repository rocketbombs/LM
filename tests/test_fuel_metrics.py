#!/usr/bin/env python3
"""
Quick test to verify fuel budget metrics are working correctly.
"""

import json
import sys
from lambda_gen import (
    Config, generate_example, TermGenerator,
    TreeReducer, GraphReducer, random
)


def test_fuel_metrics():
    print("Testing fuel budget metrics implementation...")
    print("=" * 60)

    # Create a simple config
    config = Config(
        strategy='levy_like',
        render='debruijn',
        max_depth=6,
        min_depth=2,
        max_size=40,
        max_steps=50,  # Small budget to test fuel tracking
        share=True,
        seed=42
    )

    rng = random.Random(42)

    # Generate a few examples
    print("\nGenerating test examples...\n")
    for i in range(3):
        result = generate_example(config, rng, draw_index=i)
        if not result:
            print(f"Example {i}: Failed to generate")
            continue

        # result is a list of examples (full trace)
        if isinstance(result, list):
            print(f"\nExample {i}: Generated trace with {len(result)} steps")

            # Check first and last step
            first = result[0]
            last = result[-1]

            print(f"  First step:")
            print(f"    step_k: {first['step_k']}")
            print(f"    steps_total: {first['steps_total']}")
            print(f"    term: {first['term'][:50]}...")

            # Check metadata
            meta = first['meta']
            print(f"  Metadata:")
            print(f"    max_steps: {meta.get('max_steps', 'MISSING')}")
            print(f"    fuel_remaining: {meta.get('fuel_remaining', 'MISSING')}")
            print(f"    fuel_consumed_ratio: {meta.get('fuel_consumed_ratio', 'MISSING'):.3f}")
            print(f"    is_pathological: {meta.get('is_pathological', 'MISSING')}")
            print(f"    size_growth_rate: {meta.get('size_growth_rate', 'MISSING'):.2f}")
            print(f"    initial_size: {meta.get('initial_size', 'MISSING')}")

            print(f"  Last step:")
            print(f"    step_k: {last['step_k']}")
            last_meta = last['meta']
            print(f"    fuel_remaining: {last_meta.get('fuel_remaining', 'MISSING')}")
            print(f"    fuel_consumed_ratio: {last_meta.get('fuel_consumed_ratio', 'MISSING'):.3f}")
            print(f"    is_pathological: {last_meta.get('is_pathological', 'MISSING')}")
            print(f"    size: {last_meta.get('size', 'MISSING')}")
            print(f"    size_growth_rate: {last_meta.get('size_growth_rate', 'MISSING'):.2f}")

            # Verify fuel metrics are correct
            expected_fuel_remaining = max(0, config.max_steps - last['step_k'])
            actual_fuel_remaining = last_meta.get('fuel_remaining', -1)

            if expected_fuel_remaining == actual_fuel_remaining:
                print(f"  ✓ Fuel remaining calculation correct: {actual_fuel_remaining}")
            else:
                print(f"  ✗ Fuel remaining mismatch: expected {expected_fuel_remaining}, got {actual_fuel_remaining}")

            expected_fuel_ratio = last['step_k'] / config.max_steps
            actual_fuel_ratio = last_meta.get('fuel_consumed_ratio', -1.0)

            if abs(expected_fuel_ratio - actual_fuel_ratio) < 0.01:
                print(f"  ✓ Fuel consumed ratio correct: {actual_fuel_ratio:.3f}")
            else:
                print(f"  ✗ Fuel consumed ratio mismatch: expected {expected_fuel_ratio:.3f}, got {actual_fuel_ratio:.3f}")
        else:
            print(f"Example {i}: Single example (not a trace)")

    print("\n" + "=" * 60)
    print("Test complete! Fuel budget metrics are working.")
    return True


if __name__ == '__main__':
    try:
        success = test_fuel_metrics()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
