#!/usr/bin/env python3
"""
Quick throughput test with timeout protection.

Demonstrates that:
1. Small/simple terms generate quickly
2. Some complex terms can be pathological (hang)
3. Our fuel budget tracking catches these cases
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

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_configuration(name, config, num_terms=20, timeout_secs=5):
    """Test a configuration with timeout protection."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Config: depth={config.min_depth}-{config.max_depth}, "
          f"size={config.max_size}, max_steps={config.max_steps}")
    print(f"  Timeout: {timeout_secs}s per term")

    rng = random.Random(config.seed)

    successful = 0
    failed = 0
    timeout_count = 0
    pathological_count = 0

    times = []
    steps_list = []
    fuel_ratios = []

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

                    steps_list.append(last['steps_total'])
                    fuel_ratios.append(meta.get('fuel_consumed_ratio', 0.0))

                    if meta.get('is_pathological', False):
                        pathological_count += 1
                        print(f"    Term {i}: PATHOLOGICAL "
                              f"(steps={last['steps_total']}, "
                              f"fuel={meta.get('fuel_consumed_ratio', 0):.1%}, "
                              f"growth={meta.get('size_growth_rate', 1):.2f}x)")
                else:
                    failed += 1

        except TimeoutError:
            timeout_count += 1
            print(f"    Term {i}: TIMEOUT (>{timeout_secs}s) - pathological case!")
        except Exception as e:
            failed += 1
            print(f"    Term {i}: ERROR - {e}")

    total_time = time.time() - start_time

    # Results
    print(f"\n  Results:")
    print(f"    Successful: {successful}/{num_terms}")
    print(f"    Failed: {failed}/{num_terms}")
    print(f"    Timeouts: {timeout_count}/{num_terms}")
    print(f"    Pathological (detected): {pathological_count}/{successful if successful > 0 else 1}")

    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        throughput = successful / total_time

        print(f"    Avg time: {avg_time*1000:.1f}ms")
        print(f"    Max time: {max_time*1000:.1f}ms")
        print(f"    Throughput: {throughput:.1f} terms/s")

    if steps_list:
        avg_steps = sum(steps_list) / len(steps_list)
        max_steps = max(steps_list)
        print(f"    Avg steps: {avg_steps:.1f}")
        print(f"    Max steps: {max_steps}")

    if fuel_ratios:
        avg_fuel = sum(fuel_ratios) / len(fuel_ratios)
        max_fuel = max(fuel_ratios)
        print(f"    Avg fuel consumed: {avg_fuel:.1%}")
        print(f"    Max fuel consumed: {max_fuel:.1%}")

    # Assessment
    print(f"\n  Assessment:")
    if timeout_count == 0:
        print(f"    ✓ No timeouts - all terms generated within {timeout_secs}s")
    else:
        print(f"    ⚠️  {timeout_count} timeouts - some terms are pathological!")

    if pathological_count > 0:
        print(f"    ✓ Pathological detection working - caught {pathological_count} cases")

    return {
        'successful': successful,
        'timeouts': timeout_count,
        'pathological': pathological_count
    }


def main():
    """Run quick throughput tests."""
    print("="*60)
    print("Quick Throughput Test with Timeout Protection")
    print("="*60)
    print("\nThis test demonstrates:")
    print("  1. Small/simple terms generate quickly")
    print("  2. Complex terms can hang (pathological cases)")
    print("  3. Our fuel budget tracking catches these cases")

    # Test 1: Simple terms (should be fast, no issues)
    print("\n" + "="*60)
    print("TEST 1: Small Simple Terms (baseline)")
    print("="*60)

    simple_config = Config(
        strategy='levy_like',
        render='debruijn',
        max_depth=4,
        min_depth=2,
        max_size=20,
        max_steps=30,
        share=True,
        seed=42
    )

    result1 = test_configuration(
        "Simple Terms",
        simple_config,
        num_terms=30,
        timeout_secs=3
    )

    # Test 2: Medium terms (may have some pathological cases)
    print("\n" + "="*60)
    print("TEST 2: Medium Complexity Terms")
    print("="*60)

    medium_config = Config(
        strategy='levy_like',
        render='debruijn',
        max_depth=6,
        min_depth=3,
        max_size=40,
        max_steps=80,
        share=True,
        seed=43  # Different seed
    )

    result2 = test_configuration(
        "Medium Terms",
        medium_config,
        num_terms=20,
        timeout_secs=5
    )

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"\nSimple Terms:")
    print(f"  Successful: {result1['successful']}")
    print(f"  Timeouts: {result1['timeouts']}")
    print(f"  Pathological: {result1['pathological']}")

    print(f"\nMedium Terms:")
    print(f"  Successful: {result2['successful']}")
    print(f"  Timeouts: {result2['timeouts']}")
    print(f"  Pathological: {result2['pathological']}")

    print(f"\nConclusions:")
    if result1['timeouts'] == 0:
        print(f"  ✓ Simple terms generate reliably")
    else:
        print(f"  ⚠️  Simple terms had {result1['timeouts']} timeouts")

    total_pathological = result1['pathological'] + result2['pathological']
    total_timeouts = result1['timeouts'] + result2['timeouts']

    if total_pathological > 0:
        print(f"  ✓ Pathological detection working ({total_pathological} detected)")

    if total_timeouts > 0:
        print(f"  ⚠️  Some terms timeout - this is the throughput issue!")
        print(f"     Total timeouts: {total_timeouts}")
        print(f"     Our fuel budget tracking helps identify these cases")

    print(f"\n  Key insight: The fuel budget metrics allow us to:")
    print(f"    - Detect pathological cases in generated data")
    print(f"    - Track fuel consumption patterns")
    print(f"    - Identify terms that will be slow")
    print(f"    - Make the model aware of reduction costs")

    print("\n" + "="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
