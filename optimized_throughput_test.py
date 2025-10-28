#!/usr/bin/env python3
"""
Optimized throughput test with fuel budget tuned for wall clock performance.

Key insight: Fuel budget should be set to ensure reasonable wall clock time,
not just as metadata. A term that doesn't complete within the fuel budget
should be marked as diverged, not allowed to timeout.

Goal: Keep throughput degradation to <10x across complexity levels.
"""

import time
import signal
import sys
from contextlib import contextmanager
from lambda_gen import Config, generate_example
import random


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timeouts."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_configuration(name, config, num_terms=30, timeout_secs=10):
    """Test a configuration with timeout protection."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Config: depth={config.min_depth}-{config.max_depth}, "
          f"size={config.max_size}, max_steps={config.max_steps}")
    print(f"Test timeout: {timeout_secs}s per term (should NOT be hit!)")

    rng = random.Random(config.seed)

    successful = 0
    diverged = 0
    failed = 0
    timeout_count = 0
    pathological_count = 0

    times = []
    steps_list = []
    fuel_ratios = []
    diverged_fuel = []

    start_time = time.time()

    for i in range(num_terms):
        try:
            with timeout(timeout_secs):
                term_start = time.time()
                result = generate_example(config, rng, draw_index=i)
                term_time = time.time() - term_start

                if result and isinstance(result, list) and len(result) > 0:
                    successful += 1
                    times.append(term_time)

                    # Check last step
                    last = result[-1]
                    meta = last['meta']
                    is_diverged = last['diverged']

                    steps_list.append(last['steps_total'])
                    fuel_ratio = meta.get('fuel_consumed_ratio', 0.0)
                    fuel_ratios.append(fuel_ratio)

                    if is_diverged:
                        diverged += 1
                        diverged_fuel.append(fuel_ratio)
                        print(f"  Term {i:2d}: DIVERGED at fuel={fuel_ratio:.1%} "
                              f"({term_time*1000:.1f}ms)")

                    if meta.get('is_pathological', False):
                        pathological_count += 1
                else:
                    failed += 1

        except TimeoutError:
            timeout_count += 1
            print(f"  Term {i:2d}: ⚠️  TIMEOUT ({timeout_secs}s) - FUEL BUDGET TOO HIGH!")
        except Exception as e:
            failed += 1
            if "Timed out" in str(e):
                timeout_count += 1
                print(f"  Term {i:2d}: ⚠️  TIMEOUT (reducer) - FUEL BUDGET TOO HIGH!")
            else:
                print(f"  Term {i:2d}: ERROR - {e}")

    total_time = time.time() - start_time

    # Results
    print(f"\n{'─'*70}")
    print(f"RESULTS")
    print(f"{'─'*70}")
    print(f"Successful:   {successful}/{num_terms} ({100*successful/num_terms:.1f}%)")
    print(f"Diverged:     {diverged}/{num_terms} ({100*diverged/num_terms:.1f}%)")
    print(f"Timeouts:     {timeout_count}/{num_terms} ({100*timeout_count/num_terms:.1f}%)")
    print(f"Failed:       {failed}/{num_terms}")
    print(f"Pathological: {pathological_count}/{successful if successful > 0 else 1}")

    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        throughput = successful / total_time

        print(f"\nTiming:")
        print(f"  Min:        {min_time*1000:6.1f}ms")
        print(f"  Avg:        {avg_time*1000:6.1f}ms")
        print(f"  Max:        {max_time*1000:6.1f}ms")
        print(f"  Throughput: {throughput:6.1f} terms/s")

    if steps_list:
        avg_steps = sum(steps_list) / len(steps_list)
        max_steps = max(steps_list)
        print(f"\nSteps:")
        print(f"  Avg:        {avg_steps:6.1f}")
        print(f"  Max:        {max_steps:6d}")

    if fuel_ratios:
        avg_fuel = sum(fuel_ratios) / len(fuel_ratios)
        max_fuel = max(fuel_ratios)
        print(f"\nFuel Consumption:")
        print(f"  Avg:        {avg_fuel:6.1%}")
        print(f"  Max:        {max_fuel:6.1%}")

    if diverged_fuel:
        avg_div_fuel = sum(diverged_fuel) / len(diverged_fuel)
        print(f"  Avg (diverged): {avg_div_fuel:6.1%}")

    # Assessment
    print(f"\n{'─'*70}")
    print(f"ASSESSMENT")
    print(f"{'─'*70}")

    if timeout_count == 0:
        print(f"✓ No timeouts - fuel budget is appropriate")
    else:
        print(f"✗ {timeout_count} timeouts - FUEL BUDGET TOO HIGH!")
        print(f"  Recommendation: Reduce max_steps to prevent pathological cases")

    if max_fuel > 0.95 and timeout_count == 0:
        print(f"✓ Fuel budget being used (max={max_fuel:.1%})")

    if diverged > 0:
        print(f"✓ {diverged} diverged terms properly caught by fuel budget")

    if times and max_time < timeout_secs * 0.5:
        print(f"✓ Max time ({max_time:.2f}s) well below timeout")

    return {
        'name': name,
        'successful': successful,
        'diverged': diverged,
        'timeouts': timeout_count,
        'throughput': successful / total_time if total_time > 0 else 0,
        'max_fuel': max_fuel if fuel_ratios else 0,
        'avg_time_ms': (sum(times) / len(times) * 1000) if times else 0,
    }


def main():
    """Run optimized throughput tests."""
    print("="*70)
    print("OPTIMIZED THROUGHPUT TEST")
    print("="*70)
    print("\nGoal: Tune fuel budget to ensure <10x throughput degradation")
    print("Strategy: Reduce max_steps until timeouts are eliminated")

    results = []

    # Test 1: Baseline - Simple terms
    print("\n" + "="*70)
    print("BASELINE: Simple Terms")
    print("="*70)

    r1 = test_configuration(
        "Simple (depth 2-4, max_steps=20)",
        Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=4,
            min_depth=2,
            max_size=20,
            max_steps=20,  # Conservative budget
            share=True,
            allow_divergent=True,  # Allow diverged terms
            seed=42
        ),
        num_terms=30,
        timeout_secs=5
    )
    results.append(r1)

    # Test 2: Medium terms - REDUCED fuel budget
    print("\n" + "="*70)
    print("MEDIUM COMPLEXITY (Reduced Fuel Budget)")
    print("="*70)

    r2 = test_configuration(
        "Medium (depth 4-6, max_steps=30)",
        Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=6,
            min_depth=4,
            max_size=40,
            max_steps=30,  # REDUCED from 80 to 30
            share=True,
            allow_divergent=True,
            seed=43
        ),
        num_terms=30,
        timeout_secs=5
    )
    results.append(r2)

    # Test 3: Higher complexity - even more conservative
    print("\n" + "="*70)
    print("HIGH COMPLEXITY (Very Conservative Fuel)")
    print("="*70)

    r3 = test_configuration(
        "High (depth 6-8, max_steps=40)",
        Config(
            strategy='levy_like',
            render='debruijn',
            max_depth=8,
            min_depth=6,
            max_size=60,
            max_steps=40,  # REDUCED from 150 to 40
            share=True,
            allow_divergent=True,
            seed=44
        ),
        num_terms=20,
        timeout_secs=8
    )
    results.append(r3)

    # Summary
    print("\n" + "="*70)
    print("THROUGHPUT COMPARISON")
    print("="*70)

    baseline_throughput = results[0]['throughput']

    print(f"\n{'Configuration':<35} {'Throughput':>12} {'Degradation':>12} {'Timeouts':>10}")
    print("─"*70)

    for r in results:
        degradation = baseline_throughput / r['throughput'] if r['throughput'] > 0 else float('inf')
        timeout_status = "✓" if r['timeouts'] == 0 else "✗"
        print(f"{r['name']:<35} {r['throughput']:>10.1f}/s {degradation:>10.1f}x "
              f"{r['timeouts']:>4d}/30 {timeout_status}")

    # Final assessment
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)

    all_no_timeout = all(r['timeouts'] == 0 for r in results)
    max_degradation = max(
        baseline_throughput / r['throughput'] if r['throughput'] > 0 else float('inf')
        for r in results[1:]  # Skip baseline
    )

    if all_no_timeout:
        print(f"\n✓ SUCCESS: All configurations complete without timeouts")
    else:
        print(f"\n✗ FAILURE: Some configurations still timeout")

    if max_degradation < 10:
        print(f"✓ SUCCESS: Max throughput degradation {max_degradation:.1f}x < 10x target")
    else:
        print(f"⚠️  WARNING: Max throughput degradation {max_degradation:.1f}x exceeds 10x")

    print(f"\nKey Insights:")
    print(f"  - Fuel budget acts as a throughput governor")
    print(f"  - Terms should hit fuel budget, not test timeout")
    print(f"  - Diverged terms are OK - they're caught gracefully")
    print(f"  - Reducing max_steps ensures predictable wall clock time")

    print("\n" + "="*70)

    return all_no_timeout and max_degradation < 10


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
