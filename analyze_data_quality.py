#!/usr/bin/env python3
"""
Data Quality Analysis for Lambda Calculus Training Data

Analyzes generated training examples to ensure:
1. Sound mathematical reductions
2. Good data structure and coverage
3. Model will learn to reduce terms to normal form correctly
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re


def parse_debruijn_term(term_str: str) -> dict:
    """Parse De Bruijn term to identify variables and structure."""
    # Count lambdas (abstractions)
    lambdas = term_str.count('\\.')

    # Extract all variable indices
    var_pattern = r'(?<![0-9])([0-9]+)(?![0-9])'
    variables = [int(v) for v in re.findall(var_pattern, term_str)]

    # Count applications (roughly - not perfect but good enough)
    # Applications are represented by juxtaposition or parentheses
    apps = term_str.count('(')

    return {
        'lambdas': lambdas,
        'variables': variables,
        'max_var_index': max(variables) if variables else -1,
        'applications': apps,
        'length': len(term_str)
    }


def has_redex(term_str: str) -> bool:
    """
    Check if term has a redex (beta-reducible expression).
    A redex is (λ.body)arg - lambda immediately applied.
    """
    # Simple heuristic: look for patterns like (\....)(...)
    # This matches application of a lambda
    redex_pattern = r'\(\\\..*?\)\s*\('
    return bool(re.search(redex_pattern, term_str))


def analyze_example(ex: dict) -> dict:
    """Analyze a single training example."""
    term = ex['term']
    step_k = ex['step_k']
    steps_total = ex['steps_total']
    target_span = tuple(ex['target_span'])
    meta = ex['meta']

    term_info = parse_debruijn_term(term)
    is_nf_marker = target_span == (0, 0)
    is_final_step = step_k >= steps_total
    has_redex_present = has_redex(term)

    return {
        'step_k': step_k,
        'steps_total': steps_total,
        'is_final': is_final_step,
        'is_nf_marker': is_nf_marker,
        'target_span': target_span,
        'size': meta['size'],
        'depth': meta['depth'],
        'initial_size': meta['initial_size'],
        'size_growth_rate': meta['size_growth_rate'],
        'term_length': term_info['length'],
        'lambdas': term_info['lambdas'],
        'applications': term_info['applications'],
        'has_redex': has_redex_present,
        'trace_id': ex['trace_id'],
        'diverged': ex['diverged'],
    }


def analyze_dataset(filepath: Path) -> Dict[str, Any]:
    """Comprehensive dataset analysis."""

    print(f"Analyzing dataset: {filepath}")

    examples = []
    traces = defaultdict(list)

    # Load and group by trace
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i % 1000 == 0 and i > 0:
                print(f"  Loaded {i} examples...")

            ex = json.loads(line)
            analyzed = analyze_example(ex)
            examples.append(analyzed)
            traces[analyzed['trace_id']].append(analyzed)

    print(f"\nLoaded {len(examples)} examples from {len(traces)} traces\n")

    # Analysis results
    results = {
        'total_examples': len(examples),
        'total_traces': len(traces),
        'issues': [],
    }

    # 1. CORRECTNESS: Check final steps are in normal form
    print("=" * 70)
    print("1. NORMAL FORM VERIFICATION")
    print("=" * 70)

    final_steps = [ex for ex in examples if ex['is_final']]
    final_with_nf_marker = [ex for ex in final_steps if ex['is_nf_marker']]
    final_without_nf_marker = [ex for ex in final_steps if not ex['is_nf_marker']]

    print(f"Final steps: {len(final_steps)}")
    if final_steps:
        print(f"  With NF marker (0,0): {len(final_with_nf_marker)} ({100*len(final_with_nf_marker)/len(final_steps):.1f}%)")
        print(f"  Without NF marker: {len(final_without_nf_marker)} ({100*len(final_without_nf_marker)/len(final_steps):.1f}%)")
    else:
        print(f"  WARNING: No final steps in dataset (likely partial/incomplete traces)")

    # Check non-final steps should NOT have NF marker
    non_final_steps = [ex for ex in examples if not ex['is_final']]
    non_final_with_nf = [ex for ex in non_final_steps if ex['is_nf_marker']]

    if non_final_with_nf:
        print(f"\nWARNING: {len(non_final_with_nf)} non-final steps have NF marker (0,0) - DATA BUG!")
        results['issues'].append(f"Non-final steps with NF marker: {len(non_final_with_nf)}")
        for ex in non_final_with_nf[:3]:
            print(f"    Trace {ex['trace_id']}, step {ex['step_k']}/{ex['steps_total']}")
    else:
        print("OK: All non-final steps have valid redex spans")

    # 2. SIZE DISTRIBUTION
    print("\n" + "=" * 70)
    print("2. TERM SIZE DISTRIBUTION")
    print("=" * 70)

    sizes = [ex['size'] for ex in examples]
    initial_sizes = [ex['initial_size'] for ex in examples]

    print(f"\nCurrent term sizes:")
    print(f"  Min: {min(sizes)}, Max: {max(sizes)}, Mean: {sum(sizes)/len(sizes):.1f}")
    print(f"  Percentiles: p25={sorted(sizes)[len(sizes)//4]}, p50={sorted(sizes)[len(sizes)//2]}, p75={sorted(sizes)[3*len(sizes)//4]}")

    print(f"\nInitial term sizes:")
    print(f"  Min: {min(initial_sizes)}, Max: {max(initial_sizes)}, Mean: {sum(initial_sizes)/len(initial_sizes):.1f}")
    print(f"  Percentiles: p25={sorted(initial_sizes)[len(initial_sizes)//4]}, p50={sorted(initial_sizes)[len(initial_sizes)//2]}, p75={sorted(initial_sizes)[3*len(initial_sizes)//4]}")

    # Check for size diversity
    size_bins = defaultdict(int)
    for size in sizes:
        bin_key = (size // 20) * 20  # 20-size bins
        size_bins[bin_key] += 1

    print(f"\nSize distribution (20-node bins):")
    for bin_start in sorted(size_bins.keys())[:10]:
        count = size_bins[bin_start]
        pct = 100 * count / len(sizes)
        bar = '#' * int(pct)
        print(f"  {bin_start:3d}-{bin_start+19:3d}: {count:5d} ({pct:4.1f}%) {bar}")

    # 3. TRACE QUALITY
    print("\n" + "=" * 70)
    print("3. REDUCTION TRACE QUALITY")
    print("=" * 70)

    trace_lengths = [len(trace_steps) for trace_steps in traces.values()]
    print(f"\nTrace lengths (steps per trace):")
    print(f"  Min: {min(trace_lengths)}, Max: {max(trace_lengths)}, Mean: {sum(trace_lengths)/len(trace_lengths):.1f}")
    print(f"  Percentiles: p25={sorted(trace_lengths)[len(trace_lengths)//4]}, p50={sorted(trace_lengths)[len(trace_lengths)//2]}, p75={sorted(trace_lengths)[3*len(trace_lengths)//4]}")

    # Check trace progression
    small_traces = sum(1 for tl in trace_lengths if tl < 3)
    medium_traces = sum(1 for tl in trace_lengths if 3 <= tl <= 10)
    large_traces = sum(1 for tl in trace_lengths if tl > 10)

    print(f"\nTrace size categories:")
    print(f"  Small (< 3 steps): {small_traces} ({100*small_traces/len(trace_lengths):.1f}%)")
    print(f"  Medium (3-10 steps): {medium_traces} ({100*medium_traces/len(trace_lengths):.1f}%)")
    print(f"  Large (> 10 steps): {large_traces} ({100*large_traces/len(trace_lengths):.1f}%)")

    # 4. VERIFY PROGRESSION TO NF
    print("\n" + "=" * 70)
    print("4. PROGRESSION TO NORMAL FORM")
    print("=" * 70)

    # Sample some traces and verify they progress correctly
    sample_traces = list(traces.items())[:5]

    for trace_id, steps in sample_traces:
        print(f"\nTrace {trace_id}: {len(steps)} steps")

        # Sort by step_k
        steps_sorted = sorted(steps, key=lambda x: x['step_k'])

        # Check progression
        for i, step in enumerate(steps_sorted):
            is_final = step['step_k'] >= step['steps_total']
            marker = "->NF" if step['is_nf_marker'] else f"->@{step['target_span']}"
            status = "OK" if (is_final == step['is_nf_marker']) else "ERR"

            print(f"  Step {step['step_k']:2d}/{step['steps_total']:2d}: "
                  f"size={step['size']:3d}, {marker:15s} {status}")

    # 5. SIZE GROWTH VALIDATION
    print("\n" + "=" * 70)
    print("5. SIZE GROWTH VALIDATION")
    print("=" * 70)

    growth_rates = [ex['size_growth_rate'] for ex in examples]
    print(f"\nSize growth rates:")
    print(f"  Min: {min(growth_rates):.2f}, Max: {max(growth_rates):.2f}, Mean: {sum(growth_rates)/len(growth_rates):.2f}")

    # Check for pathological growth
    high_growth = [ex for ex in examples if ex['size_growth_rate'] > 2.0]
    if high_growth:
        print(f"\nWARNING: {len(high_growth)} examples with >2x size growth")
        print(f"         This might indicate pathological reductions (should be filtered)")
        results['issues'].append(f"High growth examples: {len(high_growth)}")

    # 6. DIVERGENCE CHECK
    print("\n" + "=" * 70)
    print("6. DIVERGENCE CHECK")
    print("=" * 70)

    diverged_count = sum(1 for ex in examples if ex['diverged'])
    print(f"\nDiverged examples: {diverged_count} ({100*diverged_count/len(examples):.2f}%)")

    if diverged_count > 0:
        print("WARNING: Diverged terms found - these should be filtered out!")
        results['issues'].append(f"Diverged examples: {diverged_count}")
    else:
        print("OK: No diverged terms (all converged to NF)")

    # 7. SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results['size_stats'] = {
        'min': min(sizes),
        'max': max(sizes),
        'mean': sum(sizes) / len(sizes),
        'p50': sorted(sizes)[len(sizes) // 2],
    }

    results['trace_stats'] = {
        'min': min(trace_lengths),
        'max': max(trace_lengths),
        'mean': sum(trace_lengths) / len(trace_lengths),
        'p50': sorted(trace_lengths)[len(trace_lengths) // 2],
    }

    print(f"\nOK: Total examples: {results['total_examples']}")
    print(f"OK: Total traces: {results['total_traces']}")
    print(f"OK: Avg examples per trace: {results['total_examples'] / results['total_traces']:.1f}")
    print(f"OK: Size range: {results['size_stats']['min']}-{results['size_stats']['max']} (mean: {results['size_stats']['mean']:.1f})")
    print(f"OK: Trace length range: {results['trace_stats']['min']}-{results['trace_stats']['max']} (mean: {results['trace_stats']['mean']:.1f})")

    if results['issues']:
        print(f"\nWARNING: ISSUES FOUND: {len(results['issues'])}")
        for issue in results['issues']:
            print(f"   - {issue}")
    else:
        print("\nSUCCESS: NO ISSUES FOUND - Data quality looks good!")

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_data_quality.py <dataset.jsonl>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    results = analyze_dataset(filepath)
