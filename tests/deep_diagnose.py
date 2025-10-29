#!/usr/bin/env python3
"""
Deep diagnostic for premature NF marker issue.
"""

import json
import sys
from collections import defaultdict

def deep_analyze(jsonl_path, max_examples=1000):
    print(f"Deep analysis of: {jsonl_path}")
    print("=" * 80)

    # Track problematic patterns
    premature_nf_examples = []
    diverged_with_nf = []
    trace_analysis = defaultdict(list)

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break

            example = json.loads(line)

            step_k = example.get('step_k', 0)
            steps_total = example.get('steps_total', 0)
            target_span = tuple(example.get('target_span', [0, 0]))
            diverged = example.get('diverged', False)
            term = example.get('term', '')
            trace_id = example.get('trace_id', '')

            # Track trace info
            trace_analysis[trace_id].append({
                'step_k': step_k,
                'target_span': target_span,
                'diverged': diverged,
                'term_len': len(term),
                'steps_total': steps_total
            })

            # Find premature NF markers
            if target_span == (0, 0) and step_k < steps_total:
                premature_nf_examples.append({
                    'step_k': step_k,
                    'steps_total': steps_total,
                    'diverged': diverged,
                    'term': term[:100],
                    'trace_id': trace_id
                })

            # Find diverged terms with NF markers
            if diverged and target_span == (0, 0):
                diverged_with_nf.append({
                    'step_k': step_k,
                    'steps_total': steps_total,
                    'term': term[:100],
                    'trace_id': trace_id
                })

    print(f"\n🔍 PREMATURE NF MARKERS: {len(premature_nf_examples)}")
    print("-" * 80)

    if premature_nf_examples:
        print("\nSample problematic examples:")
        for i, ex in enumerate(premature_nf_examples[:5]):
            print(f"\n{i+1}. Step {ex['step_k']}/{ex['steps_total']} (diverged={ex['diverged']})")
            print(f"   Term: {ex['term']}")
            print(f"   Trace: {ex['trace_id'][:16]}...")

    print(f"\n🔍 DIVERGED WITH NF: {len(diverged_with_nf)}")
    print("-" * 80)

    if diverged_with_nf:
        print("\nSample diverged terms marked as NF:")
        for i, ex in enumerate(diverged_with_nf[:5]):
            print(f"\n{i+1}. Step {ex['step_k']}/{ex['steps_total']}")
            print(f"   Term: {ex['term']}")

    print(f"\n🔍 TRACE ANALYSIS")
    print("-" * 80)

    # Analyze trace patterns
    trace_with_issues = []
    for trace_id, steps in trace_analysis.items():
        steps_sorted = sorted(steps, key=lambda x: x['step_k'])

        # Check if ANY non-final step has (0,0)
        has_issue = False
        for step in steps_sorted[:-1]:  # All but last
            if step['target_span'] == (0, 0):
                has_issue = True
                break

        if has_issue:
            trace_with_issues.append((trace_id, steps_sorted))

    print(f"\nTraces with premature NF: {len(trace_with_issues)}")

    if trace_with_issues:
        print("\nDetailed trace example:")
        trace_id, steps = trace_with_issues[0]
        print(f"Trace ID: {trace_id[:16]}...")
        print(f"Total steps: {len(steps)}")
        print(f"Steps_total field: {steps[0]['steps_total']}")
        print(f"Diverged: {steps[0]['diverged']}")
        print("\nStep-by-step:")
        for step in steps[:10]:  # Show first 10 steps
            span_marker = "❌" if step['target_span'] == (0, 0) else "✓"
            print(f"  Step {step['step_k']:3d}: target_span={step['target_span']} {span_marker} (term_len={step['term_len']})")

        if len(steps) > 10:
            print(f"  ... ({len(steps) - 10} more steps)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deep_diagnose.py <data.jsonl> [max_examples]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    max_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    deep_analyze(jsonl_path, max_examples)
