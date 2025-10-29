#!/usr/bin/env python3
"""
Diversity Checker for Training Data

Analyzes uniqueness, coverage, and diversity metrics for lambda calculus training data.
Essential for ensuring 10M+ examples don't have excessive duplication.
"""

import json
import sys
from collections import Counter, defaultdict
import statistics

def analyze_diversity(jsonl_path, max_examples=None):
    print(f"Analyzing diversity: {jsonl_path}")
    print("=" * 80)

    # Uniqueness tracking
    unique_terms = set()
    unique_traces = set()

    # Coverage metrics
    sizes = []
    depths = []
    step_counts = []
    growth_rates = []

    # Duplication tracking
    term_counts = Counter()
    trace_counts = Counter()

    # Read examples
    line_count = 0
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            example = json.loads(line)

            # Track uniqueness
            term = example['term']
            trace_id = example['trace_id']

            unique_terms.add(term)
            unique_traces.add(trace_id)

            term_counts[term] += 1
            trace_counts[trace_id] += 1

            # Track coverage
            meta = example['meta']
            sizes.append(meta['size'])
            depths.append(meta['depth'])
            step_counts.append(example['steps_total'])
            growth_rates.append(meta['size_growth_rate'])

            line_count += 1

    total_examples = line_count

    print(f"\n📊 UNIQUENESS METRICS (from {total_examples:,} examples)")
    print("-" * 80)

    # Term uniqueness
    unique_term_count = len(unique_terms)
    term_uniqueness_pct = 100 * unique_term_count / total_examples if total_examples > 0 else 0

    print(f"\nUnique Terms:")
    print(f"  Total: {unique_term_count:,} / {total_examples:,}")
    print(f"  Uniqueness: {term_uniqueness_pct:.2f}%")

    if term_uniqueness_pct < 70:
        print(f"  ❌ WARNING: Uniqueness <70%! Risk of overfitting on duplicates.")
    elif term_uniqueness_pct < 85:
        print(f"  ⚠️  CAUTION: Uniqueness <85%. Some duplication present.")
    else:
        print(f"  ✅ EXCELLENT: High uniqueness! Great diversity.")

    # Trace uniqueness
    unique_trace_count = len(unique_traces)
    trace_uniqueness_pct = 100 * unique_trace_count / total_examples if total_examples > 0 else 0

    print(f"\nUnique Traces:")
    print(f"  Total: {unique_trace_count:,}")
    print(f"  Uniqueness: {trace_uniqueness_pct:.2f}%")

    # Most common duplicates
    most_common_terms = term_counts.most_common(10)
    max_duplicates = most_common_terms[0][1] if most_common_terms else 0

    print(f"\nDuplication Analysis:")
    print(f"  Max occurrences of any term: {max_duplicates}")

    if max_duplicates > 100:
        print(f"  ❌ SEVERE: Term appears {max_duplicates}x! Check RNG seeding.")
        print(f"\n  Most duplicated terms:")
        for term, count in most_common_terms[:5]:
            print(f"    {count:4d}x: {term[:60]}...")
    elif max_duplicates > 50:
        print(f"  ⚠️  HIGH: Term appears {max_duplicates}x. Some duplication.")
    elif max_duplicates > 10:
        print(f"  ✓  MODERATE: Max {max_duplicates}x duplication. Acceptable.")
    else:
        print(f"  ✅ EXCELLENT: Max {max_duplicates}x duplication. Very diverse!")

    print(f"\n📈 COVERAGE METRICS")
    print("-" * 80)

    # Size coverage
    if sizes:
        print(f"\nSize Distribution:")
        print(f"  Range: {min(sizes)} - {max(sizes)} nodes")
        print(f"  Mean: {statistics.mean(sizes):.1f}")
        print(f"  Median: {statistics.median(sizes):.1f}")
        print(f"  Std Dev: {statistics.stdev(sizes):.1f}")

        size_buckets = [0] * 10
        for s in sizes:
            bucket = min(9, s // 20)
            size_buckets[bucket] += 1

        print(f"  Distribution:")
        for i, count in enumerate(size_buckets):
            pct = 100 * count / len(sizes)
            bucket_range = f"{i*20:3d}-{(i+1)*20:3d}"
            bar = "█" * int(pct / 2)
            print(f"    {bucket_range} nodes: {pct:5.1f}% {bar}")

    # Depth coverage
    if depths:
        print(f"\nDepth Distribution:")
        print(f"  Range: {min(depths)} - {max(depths)}")
        print(f"  Mean: {statistics.mean(depths):.1f}")
        print(f"  Median: {statistics.median(depths):.1f}")
        print(f"  Std Dev: {statistics.stdev(depths):.1f}")

        depth_counts = Counter(depths)
        print(f"  Top depths:")
        for depth, count in depth_counts.most_common(10):
            pct = 100 * count / len(depths)
            print(f"    Depth {depth:2d}: {pct:5.1f}% ({count:,} examples)")

    # Step coverage
    if step_counts:
        print(f"\nReduction Steps Distribution:")
        print(f"  Range: {min(step_counts)} - {max(step_counts)}")
        print(f"  Mean: {statistics.mean(step_counts):.1f}")
        print(f"  Median: {statistics.median(step_counts):.1f}")
        print(f"  Std Dev: {statistics.stdev(step_counts):.1f}")

    # Growth rate coverage
    if growth_rates:
        print(f"\nSize Growth Rate Distribution:")
        print(f"  Range: {min(growth_rates):.2f}x - {max(growth_rates):.2f}x")
        print(f"  Mean: {statistics.mean(growth_rates):.2f}x")
        print(f"  Median: {statistics.median(growth_rates):.2f}x")

        if max(growth_rates) > 2.5:
            print(f"  ⚠️  WARNING: Max growth {max(growth_rates):.2f}x > 2.5x threshold!")

    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 80)

    # Recommendations based on metrics
    if term_uniqueness_pct < 70:
        print("\n❌ CRITICAL: Low uniqueness!")
        print("  Action: Use time-based RNG seeds, increase parameter variation")
        print("  Command: cargo run --release -- generate data.jsonl 10000000 16 250")
    elif term_uniqueness_pct < 85:
        print("\n⚠️  Uniqueness could be better")
        print("  Consider: More parameter variation, ensure time-based seeds")
    else:
        print("\n✅ Excellent diversity! Ready for large-scale generation.")
        print("  For 10M examples, current settings provide good coverage.")

    # Estimate for 10M
    if total_examples >= 10000:
        estimated_10m_unique = int(unique_term_count * (10_000_000 / total_examples) * (term_uniqueness_pct / 100))
        print(f"\n📊 EXTRAPOLATION TO 10M EXAMPLES:")
        print(f"  Current sample: {total_examples:,} examples")
        print(f"  Current uniqueness: {term_uniqueness_pct:.1f}%")
        print(f"  Estimated unique at 10M: {estimated_10m_unique:,}")
        print(f"  Estimated duplicates at 10M: {10_000_000 - estimated_10m_unique:,}")

        if term_uniqueness_pct > 90:
            print(f"  ✅ EXCELLENT: >90% uniqueness at scale!")
        elif term_uniqueness_pct > 80:
            print(f"  ✓  GOOD: >80% uniqueness at scale")
        else:
            print(f"  ⚠️  CONCERN: <80% uniqueness may cause overfitting")

    print("\n" + "=" * 80)
    print("Diversity analysis complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_diversity.py <data.jsonl> [max_examples]")
        print("\nExamples:")
        print("  python check_diversity.py data.jsonl")
        print("  python check_diversity.py data.jsonl 50000")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    max_examples = int(sys.argv[2]) if len(sys.argv) > 2 else None

    analyze_diversity(jsonl_path, max_examples)
