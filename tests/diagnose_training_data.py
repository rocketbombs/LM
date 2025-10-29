#!/usr/bin/env python3
"""
Training Data Diagnostic Tool

Analyzes generated training data for quality issues that could cause
strange training behavior like high path flagging or abnormal throughput.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import statistics

def analyze_training_data(jsonl_path, num_samples=10000):
    """Comprehensive analysis of training data quality."""

    print(f"Analyzing: {jsonl_path}")
    print("=" * 80)

    # Statistics to collect
    term_lengths = []
    step_counts = []
    sizes = []
    depths = []
    is_pathological_counts = Counter()
    diverged_counts = Counter()
    target_spans = []
    steps_total_list = []
    time_consumed_ratios = []
    size_growth_rates = []
    avg_step_ms_list = []

    # Check for data issues
    issues = defaultdict(int)

    # Sample examples for inspection
    samples = []

    try:
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break

                try:
                    example = json.loads(line)

                    # Basic fields
                    term = example.get('term', '')
                    term_lengths.append(len(term))

                    step_k = example.get('step_k', 0)
                    step_counts.append(step_k)

                    steps_total = example.get('steps_total', 0)
                    steps_total_list.append(steps_total)

                    diverged = example.get('diverged', False)
                    diverged_counts[diverged] += 1

                    target_span = example.get('target_span', [0, 0])
                    target_spans.append(tuple(target_span))

                    # Metadata
                    meta = example.get('meta', {})
                    size = meta.get('size', 0)
                    sizes.append(size)

                    depth = meta.get('depth', 0)
                    depths.append(depth)

                    is_pathological = meta.get('is_pathological', False)
                    is_pathological_counts[is_pathological] += 1

                    time_consumed_ratio = meta.get('time_consumed_ratio', 0.0)
                    time_consumed_ratios.append(time_consumed_ratio)

                    size_growth_rate = meta.get('size_growth_rate', 1.0)
                    size_growth_rates.append(size_growth_rate)

                    avg_step_ms = meta.get('avg_step_ms', 0.0)
                    avg_step_ms_list.append(avg_step_ms)

                    # Check for issues
                    if len(term) == 0:
                        issues['empty_term'] += 1
                    if len(term) < 5:
                        issues['very_short_term'] += 1
                    if size == 0:
                        issues['zero_size'] += 1
                    if depth == 0:
                        issues['zero_depth'] += 1
                    if target_span == [0, 0] and step_k < steps_total - 1:
                        issues['premature_nf_marker'] += 1
                    if steps_total == 0:
                        issues['zero_steps'] += 1
                    if time_consumed_ratio > 1.0:
                        issues['time_ratio_over_1'] += 1
                    if size_growth_rate < 0:
                        issues['negative_growth'] += 1

                    # Collect samples
                    if len(samples) < 10:
                        samples.append(example)

                except json.JSONDecodeError as e:
                    issues['json_decode_error'] += 1
                    print(f"Line {i}: JSON decode error: {e}")
                except Exception as e:
                    issues['other_parse_error'] += 1
                    print(f"Line {i}: Parse error: {e}")

        total_examples = i + 1

    except FileNotFoundError:
        print(f"ERROR: File not found: {jsonl_path}")
        return
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

    print(f"\n📊 OVERALL STATISTICS (from {total_examples} examples)")
    print("-" * 80)

    # Term length statistics
    if term_lengths:
        print(f"\nTerm Length:")
        print(f"  Mean: {statistics.mean(term_lengths):.1f} chars")
        print(f"  Median: {statistics.median(term_lengths):.1f} chars")
        print(f"  Min: {min(term_lengths)} chars")
        print(f"  Max: {max(term_lengths)} chars")
        print(f"  StdDev: {statistics.stdev(term_lengths):.1f} chars" if len(term_lengths) > 1 else "")

    # Size statistics
    if sizes:
        print(f"\nTerm Size (node count):")
        print(f"  Mean: {statistics.mean(sizes):.1f}")
        print(f"  Median: {statistics.median(sizes):.1f}")
        print(f"  Min: {min(sizes)}")
        print(f"  Max: {max(sizes)}")

    # Depth statistics
    if depths:
        print(f"\nTerm Depth:")
        print(f"  Mean: {statistics.mean(depths):.1f}")
        print(f"  Median: {statistics.median(depths):.1f}")
        print(f"  Min: {min(depths)}")
        print(f"  Max: {max(depths)}")

    # Steps statistics
    if steps_total_list:
        print(f"\nSteps per Term:")
        print(f"  Mean: {statistics.mean(steps_total_list):.1f}")
        print(f"  Median: {statistics.median(steps_total_list):.1f}")
        print(f"  Min: {min(steps_total_list)}")
        print(f"  Max: {max(steps_total_list)}")

    # Pathological rate
    total_with_flag = sum(is_pathological_counts.values())
    if total_with_flag > 0:
        pathological_pct = (is_pathological_counts[True] / total_with_flag) * 100
        print(f"\n⚠️  Pathological Cases:")
        print(f"  {is_pathological_counts[True]} / {total_with_flag} ({pathological_pct:.1f}%)")

        if pathological_pct > 50:
            print(f"  🚨 WARNING: Over 50% pathological! This will hurt training.")

    # Diverged rate
    total_with_diverged = sum(diverged_counts.values())
    if total_with_diverged > 0:
        diverged_pct = (diverged_counts[True] / total_with_diverged) * 100
        print(f"\nDiverged Terms:")
        print(f"  {diverged_counts[True]} / {total_with_diverged} ({diverged_pct:.1f}%)")

    # Time consumption
    if time_consumed_ratios:
        print(f"\nTime Consumed Ratio:")
        print(f"  Mean: {statistics.mean(time_consumed_ratios):.3f}")
        print(f"  Median: {statistics.median(time_consumed_ratios):.3f}")
        print(f"  Max: {max(time_consumed_ratios):.3f}")

    # Size growth
    if size_growth_rates:
        print(f"\nSize Growth Rate:")
        print(f"  Mean: {statistics.mean(size_growth_rates):.2f}x")
        print(f"  Median: {statistics.median(size_growth_rates):.2f}x")
        print(f"  Max: {max(size_growth_rates):.2f}x")

    # Timing
    if avg_step_ms_list:
        print(f"\nAvg Step Time:")
        print(f"  Mean: {statistics.mean(avg_step_ms_list):.3f} ms")
        print(f"  Median: {statistics.median(avg_step_ms_list):.3f} ms")

    # Target span distribution
    if target_spans:
        span_counter = Counter(target_spans)
        nf_count = span_counter.get((0, 0), 0)
        nf_pct = (nf_count / len(target_spans)) * 100
        print(f"\nTarget Spans:")
        print(f"  Normal Form (0,0): {nf_count} ({nf_pct:.1f}%)")
        print(f"  Non-NF spans: {len(target_spans) - nf_count} ({100-nf_pct:.1f}%)")

    # Issues detected
    print(f"\n🔍 DATA QUALITY ISSUES")
    print("-" * 80)

    if issues:
        print("\nIssues found:")
        for issue, count in sorted(issues.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_examples) * 100
            print(f"  {issue}: {count} ({pct:.1f}%)")
    else:
        print("✅ No obvious data quality issues detected!")

    # Root cause analysis
    print(f"\n🎯 ROOT CAUSE ANALYSIS")
    print("-" * 80)

    if pathological_pct > 70:
        print("\n🚨 CRITICAL: >70% pathological cases!")
        print("\nLikely causes:")
        print("  1. Wall clock limit too low (100ms may be too tight)")
        print("  2. Terms too complex for the time budget")
        print("  3. Generator creating oversized terms")
        print("\nRecommended fixes:")
        print("  • Increase wall_clock_limit_ms: try 500ms or 1000ms")
        print("  • Reduce max_depth when generating terms")
        print("  • Check if Rust generator is respecting size limits")

    if statistics.mean(term_lengths) < 10:
        print("\n🚨 CRITICAL: Terms are too short!")
        print(f"  Mean length: {statistics.mean(term_lengths):.1f} chars")
        print("\nLikely causes:")
        print("  1. Generator producing trivial terms")
        print("  2. Size constraints too restrictive")
        print("\nRecommended fixes:")
        print("  • Check generator configuration (min_depth, max_depth)")
        print("  • Verify term generation logic")

    if statistics.mean(steps_total_list) < 2:
        print("\n⚠️  WARNING: Very few reduction steps!")
        print(f"  Mean steps: {statistics.mean(steps_total_list):.1f}")
        print("\nThis means terms are trivial or already in NF.")
        print("Training data should have variety of reduction lengths.")

    # High throughput diagnosis
    if statistics.mean(term_lengths) < 20:
        tokens_per_example = statistics.mean(term_lengths) * statistics.mean(steps_total_list)
        print(f"\n💨 HIGH THROUGHPUT EXPLANATION:")
        print(f"  Avg tokens per term: ~{statistics.mean(term_lengths):.1f}")
        print(f"  Avg steps per term: ~{statistics.mean(steps_total_list):.1f}")
        print(f"  Avg tokens per example: ~{tokens_per_example:.1f}")
        print("\n  With short terms and few steps, 160k tokens/sec is expected!")
        print("  This is NOT a training efficiency gain - you have trivial data.")

    # Sample examples
    print(f"\n📝 SAMPLE EXAMPLES")
    print("-" * 80)

    for i, sample in enumerate(samples[:3], 1):
        print(f"\nExample {i}:")
        print(f"  Term: {sample.get('term', 'N/A')[:80]}")
        print(f"  Step {sample.get('step_k')}/{sample.get('steps_total')}")
        print(f"  Target span: {sample.get('target_span')}")
        print(f"  Size: {sample.get('meta', {}).get('size')}, Depth: {sample.get('meta', {}).get('depth')}")
        print(f"  Pathological: {sample.get('meta', {}).get('is_pathological')}")
        print(f"  Time ratio: {sample.get('meta', {}).get('time_consumed_ratio', 0):.3f}")

    print("\n" + "=" * 80)
    print("Diagnosis complete!")
    print("\nNext steps:")
    print("  1. Review the statistics above")
    print("  2. Check the sample examples")
    print("  3. If data quality is poor, regenerate with better parameters")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_training_data.py <path_to_data.jsonl> [num_samples]")
        print("\nExample:")
        print("  python diagnose_training_data.py data.jsonl 10000")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    analyze_training_data(jsonl_path, num_samples)
