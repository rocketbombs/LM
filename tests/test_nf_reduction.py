#!/usr/bin/env python3
"""
Test that training data includes complete reduction to Normal Form.

This verifies that:
1. Reduction reaches NF when possible
2. Training examples include the final NF step
3. NF is marked with target_span (0, 0)
4. Model sees complete reduction traces
"""

import sys
sys.path.insert(0, '/home/user/LM')

import lambda_gen
from lambda_gen import Term, TermType, GraphReducer, Config

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")

def test_simple_identity():
    """Test (λ.0)(λ.0) reduces to λ.0 (NF)"""
    print_section("TEST 1: Simple Identity Application")

    # Create term: (λ.0)(λ.0)
    id_term = Term(TermType.ABS, body=Term(TermType.VAR, var=0))
    term = Term(TermType.APP, left=id_term, right=id_term)

    print(f"Initial term: (λ.0)(λ.0)")
    print(f"Expected NF: λ.0")

    # Reduce with graph reducer
    config = Config(
        min_depth=2,
        max_depth=6,
        max_size=50,
        max_steps=100,
        wall_clock_limit_ms=1000.0,
        seed=42,
        libraries=[]
    )

    reducer = GraphReducer(max_steps=100, wall_clock_limit_ms=1000.0)
    trace, diverged, thunk_evals, thunk_hits, total_time_ms = reducer.reduce(term)

    print(f"\nReduction completed:")
    print(f"  Diverged: {diverged}")
    print(f"  Steps: {len(trace)}")
    print(f"  Total time: {total_time_ms:.2f}ms")
    print(f"  Thunk evals: {thunk_evals}, hits: {thunk_hits}")

    # Check trace
    print(f"\nTrace details:")
    for i, (t, redex_path, step_ms) in enumerate(trace):
        from lambda_gen import Renderer
        rendered = Renderer.to_debruijn_with_spans(t)
        if redex_path is None:
            print(f"  Step {i}: {rendered.string} [NF]  ({step_ms:.2f}ms)")
        else:
            print(f"  Step {i}: {rendered.string} (redex at {redex_path})  ({step_ms:.2f}ms)")

    # Verify expectations
    assert not diverged, "Should reach NF, not diverge"
    assert len(trace) >= 2, "Should have at least initial term + NF"

    final_term, final_redex, _ = trace[-1]
    assert final_redex is None, f"Final step should be NF (redex_path=None), got {final_redex}"

    print(f"\n✓ Correctly reached NF")
    print(f"✓ Final step has redex_path=None")
    return True

def test_training_data_includes_nf():
    """Test that training data generation includes NF examples"""
    print_section("TEST 2: Training Data Includes NF")

    # Generate training examples for identity
    id_term = Term(TermType.ABS, body=Term(TermType.VAR, var=0))
    term = Term(TermType.APP, left=id_term, right=id_term)

    config = Config(
        min_depth=2,
        max_depth=6,
        max_size=50,
        max_steps=100,
        wall_clock_limit_ms=1000.0,
        seed=42,
        libraries=[],
        render='debruijn'
    )

    # Generate examples (this is done internally)
    reducer = GraphReducer(max_steps=100, wall_clock_limit_ms=1000.0)
    trace, diverged, thunk_evals, thunk_hits, total_time_ms = reducer.reduce(term)

    # Simulate example generation
    examples = []
    step_times = [step_ms for _, _, step_ms in trace]
    avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    steps_total = len(trace) - 1
    initial_size = trace[0][0].size()

    for step_k in range(len(trace)):
        current_term, redex_path, step_ms = trace[step_k]
        from lambda_gen import Renderer, get_redex_span
        result = Renderer.to_debruijn_with_spans(current_term)
        target_span = list(get_redex_span(current_term, redex_path, 'debruijn'))

        example = {
            'step_k': step_k,
            'steps_total': steps_total,
            'term': result.string,
            'target_span': target_span,
            'redex_path': redex_path,
            'diverged': diverged,
        }
        examples.append(example)

    print(f"Generated {len(examples)} training examples:")
    for ex in examples:
        is_nf = ex['redex_path'] is None
        span_str = f"span {ex['target_span']}"
        nf_marker = " [NF]" if is_nf else ""
        print(f"  Step {ex['step_k']}/{ex['steps_total']}: {ex['term']} ({span_str}){nf_marker}")

    # Verify NF is included
    final_example = examples[-1]
    assert final_example['redex_path'] is None, "Final example should be NF"
    assert final_example['target_span'] == [0, 0], f"NF should have span (0,0), got {final_example['target_span']}"

    print(f"\n✓ Training data includes {len(examples)} examples")
    print(f"✓ Final example is NF with target_span (0, 0)")
    print(f"✓ Model will learn to predict (0, 0) for NF")
    return True

def test_church_numeral_reduction():
    """Test more complex reduction: Church numeral 2 applied to successor and zero"""
    print_section("TEST 3: Church Numeral Reduction")

    # Church 2: λf.λx.f(f x)
    # Succ: λn.λf.λx.f(n f x)
    # Zero: λf.λx.x

    # For simplicity, let's test: (λx.λy.x)(λz.z)(λw.w)
    # This should reduce to: λw.w

    inner_abs = Term(TermType.ABS, body=Term(TermType.VAR, var=0))  # λz.z
    outer_abs = Term(TermType.ABS, body=Term(TermType.ABS, body=Term(TermType.VAR, var=1)))  # λx.λy.x

    app1 = Term(TermType.APP, left=outer_abs, right=inner_abs)  # (λx.λy.x)(λz.z)
    term = Term(TermType.APP, left=app1, right=inner_abs)  # ((λx.λy.x)(λz.z))(λw.w)

    print("Initial term: ((λx.λy.x)(λz.z))(λw.w)")
    print("Expected NF: λ.0")

    reducer = GraphReducer(max_steps=100, wall_clock_limit_ms=1000.0)
    trace, diverged, thunk_evals, thunk_hits, total_time_ms = reducer.reduce(term)

    print(f"\nReduction completed:")
    print(f"  Diverged: {diverged}")
    print(f"  Steps: {len(trace)}")

    for i, (t, redex_path, step_ms) in enumerate(trace):
        from lambda_gen import Renderer
        rendered = Renderer.to_debruijn_with_spans(t)
        if redex_path is None:
            print(f"  Step {i}: {rendered.string} [NF]")
        else:
            print(f"  Step {i}: {rendered.string}")

    assert not diverged, "Should reach NF"
    final_term, final_redex, _ = trace[-1]
    assert final_redex is None, "Should reach NF"

    print(f"\n✓ Complex term reduces to NF")
    return True

def test_nf_immediate():
    """Test that already-NF terms are handled correctly"""
    print_section("TEST 4: Already in Normal Form")

    # Term: λ.0 (already in NF)
    term = Term(TermType.ABS, body=Term(TermType.VAR, var=0))

    print("Term: λ.0 (already in NF)")

    reducer = GraphReducer(max_steps=100, wall_clock_limit_ms=1000.0)
    trace, diverged, thunk_evals, thunk_hits, total_time_ms = reducer.reduce(term)

    print(f"\nReduction completed:")
    print(f"  Diverged: {diverged}")
    print(f"  Steps: {len(trace)}")

    assert not diverged, "Should not diverge"
    assert len(trace) == 1, f"Should have exactly 1 step (the NF itself), got {len(trace)}"

    final_term, final_redex, _ = trace[0]
    assert final_redex is None, "Should immediately be NF"

    print(f"✓ Already-NF term correctly handled with 1 step")
    return True

def main():
    print("\n" + "="*70)
    print("TESTING: Complete Reduction to Normal Form")
    print("="*70)

    tests = [
        ("Simple Identity", test_simple_identity),
        ("Training Data Includes NF", test_training_data_includes_nf),
        ("Church Numeral Reduction", test_church_numeral_reduction),
        ("Already in NF", test_nf_immediate),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
                print(f"\n✓✓✓ {name} PASSED ✓✓✓")
        except AssertionError as e:
            failed += 1
            print(f"\n✗✗✗ {name} FAILED ✗✗✗")
            print(f"Error: {e}")
        except Exception as e:
            failed += 1
            print(f"\n✗✗✗ {name} ERROR ✗✗✗")
            print(f"Exception: {e}")
            import traceback
            traceback.print_exc()

    print_section("SUMMARY")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed! Model will see complete reduction traces to NF.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Early stopping issue detected!")
        sys.exit(1)

if __name__ == '__main__':
    main()
