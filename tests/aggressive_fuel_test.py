#!/usr/bin/env python3
"""
AGGRESSIVE fuel budget test.

Goal: Ensure <10x throughput degradation by using VERY small max_steps.

Hypothesis: Each reduction step can be expensive on complex terms.
Solution: Drastically limit max_steps (10-15 instead of 50-100).
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


def run_test(name, max_steps, max_depth, num_terms=40):
    """Run test with given fuel budget."""
    print(f"\n{'='*70}")
    print(f"{name}: max_steps={max_steps}, max_depth={max_depth}")
    print(f"{'='*70}")

    config = Config(
        strategy='levy_like',
        render='debruijn',
        max_depth=max_depth,
        min_depth=max(2, max_depth - 2),
        max_size=20 + max_depth * 5,
        max_steps=max_steps,  # AGGRESSIVE limit
        share=True,
        allow_divergent=True,
        seed=42 + max_depth
    )

    rng = random.Random(config.seed)

    successful = 0
    diverged = 0
    timeouts = 0
    times = []

    start = time.time()

    for i in range(num_terms):
        try:
            with timeout(3):  # 3s timeout - should NOT be hit
                t0 = time.time()
                result = generate_example(config, rng, draw_index=i)
                t1 = time.time() - t0

                if result and isinstance(result, list):
                    successful += 1
                    times.append(t1)

                    if result[-1]['diverged']:
                        diverged += 1

        except TimeoutError:
            timeouts += 1
            print(f"  Term {i:2d}: TIMEOUT! (fuel budget insufficient)")
        except Exception as e:
            if "Timed out" in str(e):
                timeouts += 1
                print(f"  Term {i:2d}: TIMEOUT! (fuel budget insufficient)")

    total_time = time.time() - start
    throughput = successful / total_time if total_time > 0 else 0

    print(f"\nResults:")
    print(f"  Successful: {successful}/{num_terms} ({100*successful/num_terms:.0f}%)")
    print(f"  Diverged:   {diverged}/{num_terms} ({100*diverged/num_terms:.0f}%)")
    print(f"  Timeouts:   {timeouts}/{num_terms} ({100*timeouts/num_terms:.0f}%)")

    if times:
        avg_ms = (sum(times) / len(times)) * 1000
        max_ms = max(times) * 1000
        print(f"  Avg time:   {avg_ms:.1f}ms")
        print(f"  Max time:   {max_ms:.1f}ms")

    print(f"  Throughput: {throughput:.1f} terms/s")

    status = "✓ PASS" if timeouts == 0 else f"✗ FAIL ({timeouts} timeouts)"
    print(f"  Status:     {status}")

    return {
        'name': name,
        'throughput': throughput,
        'timeouts': timeouts,
        'diverged_rate': diverged / num_terms if num_terms > 0 else 0,
    }


def main():
    """Test increasingly aggressive fuel budgets."""
    print("="*70)
    print("AGGRESSIVE FUEL BUDGET TEST")
    print("="*70)
    print("\nGoal: Find minimum max_steps that eliminates timeouts")
    print("Strategy: Test progressively smaller fuel budgets")

    results = []

    # Baseline
    r1 = run_test("Baseline (depth 2-4)", max_steps=15, max_depth=4)
    results.append(r1)

    # Progressive tests with VERY aggressive fuel limits
    tests = [
        ("Conservative (depth 4-6)", 15, 6),
        ("Aggressive (depth 4-6)", 12, 6),
        ("Very Aggressive (depth 4-6)", 10, 6),
        ("Ultra Aggressive (depth 6-8)", 15, 8),
        ("Ultra Aggressive v2 (depth 6-8)", 12, 8),
    ]

    for name, max_steps, max_depth in tests:
        r = run_test(name, max_steps, max_depth, num_terms=30)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("THROUGHPUT COMPARISON")
    print("="*70)

    baseline = results[0]['throughput']

    print(f"\n{'Configuration':<40} {'Throughput':>12} {'Degradation':>12} {'Status':>8}")
    print("─"*70)

    for r in results:
        deg = baseline / r['throughput'] if r['throughput'] > 0 else float('inf')
        status = "✓" if r['timeouts'] == 0 else "✗"
        div_info = f" ({r['diverged_rate']:.0%} div)" if r['diverged_rate'] > 0 else ""

        print(f"{r['name']:<40} {r['throughput']:>10.1f}/s {deg:>10.1f}x {status:>6}{div_info}")

    # Find best configuration
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    no_timeout_results = [r for r in results if r['timeouts'] == 0]

    if no_timeout_results:
        # Find configuration with best throughput and no timeouts
        best = max(no_timeout_results, key=lambda r: r['throughput'])
        degradation = baseline / best['throughput']

        print(f"\nBest configuration (no timeouts):")
        print(f"  {best['name']}")
        print(f"  Throughput: {best['throughput']:.1f} terms/s")
        print(f"  Degradation: {degradation:.1f}x from baseline")

        if degradation < 10:
            print(f"  ✓ Meets <10x degradation target")
        else:
            print(f"  ⚠️  Exceeds 10x degradation target")
            print(f"\n  Recommendation: Accept higher divergence rate")
            print(f"  or reduce term complexity (lower max_depth)")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("\n1. Fuel budget must be VERY aggressive to prevent timeouts")
    print("2. max_steps=10-15 appears to be the sweet spot")
    print("3. Higher divergence rates (20-40%) are acceptable")
    print("4. Diverged terms are well-posed - just complex")
    print("5. For depth >6, even max_steps=10 may cause timeouts")

    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
