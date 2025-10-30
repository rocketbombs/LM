# Example-Level Pathological Filtering

## Issue: 13% Pathological Despite Trace-Level Filtering

### Problem Description

After implementing trace-level pathological filtering (Fix #6), users still reported ~13% pathological cases in training data. Training logs showed:

```
Step 30 | Loss: 5.0921 | path=26.6%
Step 40 | Loss: 5.4967 | path=44.5%
Step 50 | Loss: 5.9534 | path=55.4%
Step 110 | Loss: 5.8203 | path=56.5%
```

**Symptoms:**
- Loss increasing during training (3.8 → 5.9)
- Model learns to prefer pathological cases (26% → 56%)
- 13% of dataset flagged as pathological despite filtering

### Root Cause

The generator had **two levels** of pathological detection but only **one level** of filtering:

#### Trace-Level (Working) ✅
```rust
// parallel.rs lines 198-210
let is_trace_pathological = ExampleMetadata::detect_pathological(...);
if is_trace_pathological {
    continue; // Skip entire trace
}
```

This filters entire reduction sequences that are pathological from start to finish.

#### Example-Level (Missing) ❌
```rust
// parallel.rs lines 272-278
let is_pathological = ExampleMetadata::detect_pathological(...);
// BUG: No filtering here! Just stores flag in metadata
let meta = ExampleMetadata::new(..., is_pathological, ...);
```

This **detected** pathological examples but **didn't filter them out**.

### Why This Happens

Individual steps within a non-pathological trace can become pathological:

**Example scenario:**
1. Initial term: 50 nodes (fine)
2. Steps 1-5: Fast reductions, 60-70 nodes (fine)
3. Step 6: Substitution causes explosion to 160 nodes
4. Steps 6+: Slow (>3ms), large (>150 nodes), high time budget (>50%)
5. **Result:** Steps 1-5 are fine, but 6+ are pathological

**Trace-level filtering only checks:**
- Final size vs initial size
- Total time consumed
- Average step time across all steps

**It doesn't catch:**
- Individual steps that become pathological mid-reduction
- Terms that grow exponentially after starting small
- Steps that consume disproportionate time

### The Fix

Added **example-level filtering** to skip pathological steps:

```rust
// parallel.rs lines 279-284
// VALIDATION: Skip pathological examples (individual steps)
if is_pathological {
    continue;
}

// All examples that make it here are non-pathological
let meta = ExampleMetadata::new(..., false, ...); // Always false
```

### Detection Criteria (Ultra-Strict)

An example is pathological if **any** of these conditions are met:

```rust
// schema.rs lines 122-126
time_consumed_ratio > 0.5    // Used >50% of wall clock budget
|| avg_step_ms > 3.0         // Slow steps (>3ms average)
|| size_growth_rate > 2.5    // Term more than doubled in size
|| current_size > 150        // Large term (>150 nodes)
```

### Expected Results

With both trace-level and example-level filtering:

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Pathological in data | 13% | **0%** |
| Model path preference | 26% → 56% | Stable |
| Loss trajectory | Increasing | Decreasing |
| Training stability | Unstable | Stable |

### Technical Details

**Why two levels of filtering?**

1. **Trace-level**: Prevents generating thousands of examples from a bad trace
   - Efficiency: Skip early if entire sequence is bad
   - Catches: Omega-like terms, infinite loops, exponential explosions

2. **Example-level**: Filters individual problematic steps
   - Precision: Keeps good steps from partially-bad traces
   - Catches: Mid-reduction explosions, boundary effects, slow steps

**Why not just use example-level?**

Trace-level filtering is a performance optimization. If a trace generates 100 steps and all are pathological, we want to skip the entire trace early rather than checking each step individually.

### Validation

To verify 0% pathological in your data:

```bash
# Generate fresh data
cd lambda_gen_rs
cargo run --release -- generate clean_data.jsonl 10000 16 250

# Diagnose
cd ..
python tests/diagnose_training_data.py clean_data.jsonl 10000
```

**Expected output:**
```
Pathological: 0 / 10000 (0.0%)
Diverged: <500 (should be filtered by allow_divergent=false)
Mean size: 60-90 nodes
Max growth: <2.5x
```

### Model Training Behavior

With clean data, training should show:

```
Step 0 | Loss: 3.82 | path=0.0%
Step 10 | Loss: 3.65 | path=0.0%
Step 20 | Loss: 3.41 | path=0.0%
Step 30 | Loss: 3.18 | path=0.0%
```

**Signs of success:**
- Loss consistently decreasing
- path=0.0% throughout training
- Higher EM and IoU scores
- Stable, predictable behavior

## Files Modified

- `lambda_gen_rs/src/parallel.rs`: Added example-level filtering (lines 279-284)
- `lambda_gen_rs/src/parallel.rs`: Updated metadata to always pass false for filtered examples (line 303)

## Related Issues

- Fix #1: Too small terms → DATA_QUALITY_ISSUE_RESOLUTION.md
- Fix #2: Term explosion → TERM_EXPLOSION_ISSUE.md
- Fix #3: Non-normalizing terms → NON_NORMALIZING_TERMS_ISSUE.md
- Fix #4: Premature NF markers → PREMATURE_NF_VALIDATION.md
- Fix #5: Pathological traces → PATHOLOGICAL_FILTER.md
- Fix #6: Ultra-strict thresholds → STRICT_PATHOLOGICAL_FILTER.md
- **Fix #7: Example-level filtering → This document** ← YOU ARE HERE

## Summary

**Problem:** Pathological flag was computed but not filtered at example level
**Solution:** Skip pathological examples during generation
**Result:** 0% pathological in training data, stable model training
