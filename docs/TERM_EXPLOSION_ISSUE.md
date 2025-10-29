# Term Explosion Issue and Resolution

## Problem Report (Second Data Quality Issue)

**Date**: 2024-10-29
**Severity**: CRITICAL
**Impact**: Training data unusable

### Symptoms

After fixing the initial data quality issue and regenerating data with `max_depth=12, max_size=200`, new problems emerged:

1. **55.4% pathological rate** (should be 20-30%)
2. **45.2% divergence rate** (terms hitting time limits)
3. **50% premature NF markers** (target_span=(0,0) at wrong steps)
4. **Massive term sizes**: mean 887 nodes, max 24,150 nodes
5. **Extreme depth**: mean depth 24, max depth 246

### Diagnostic Output

```
Term Size (node count):
  Mean: 887.1
  Max: 24150           ❌ EXPLOSION!

Size Growth Rate:
  Mean: 5.86x          ❌ Terms grow 5-6x during reduction
  Max: 134.17x         ❌ Some terms explode 134x!

Diverged Terms:
  45.2%                ❌ Nearly half hit time limits

Pathological Cases:
  55.4%                ❌ Over half flagged as problematic
```

## Root Cause Analysis

### The Exponential Growth Problem

**Issue 1: max_depth=12 Creates Huge Terms**

A binary tree of depth D has up to 2^D nodes:
- Depth 6: ~64 nodes
- Depth 8: ~256 nodes
- Depth 10: ~1,024 nodes
- **Depth 12: ~4,096 nodes** ❌

With random generation, average is ~1000-2000 nodes for depth 12.

**Issue 2: Terms Grow During Reduction**

Lambda calculus reduction can cause **term explosion**:

```
Example: (\x.x x)(\x.x x)  [omega combinator]
Step 1: (\x.x x)(\x.x x)   (size: 6)
Step 2: (\x.x x)(\x.x x)   (size: 6) - cycles forever!

Example: Church numerals
(\f.(\x.f(f x)))(\f.(\x.f(f x)))
After beta reduction, size can double or triple
```

**Mean growth rate: 5.86x**
- Initial term: ~150 nodes (avg)
- After reduction: ~887 nodes
- This exceeds all reasonable limits!

**Issue 3: Massive Terms Hit Time Limits**

With 100ms wall clock limit:
- Terms with 887 nodes take 200-500ms to reduce
- **45% of terms diverge** (hit time limit before completion)
- Diverged terms may have incorrect target_span markers

**Issue 4: Generator Retries Are Wasteful**

Generator logic (generator.rs:64-73):
```rust
for _attempt in 0..100 {
    let term = self.generate_term(rng, 0, self.config.max_depth, 0)?;
    if term.size() <= self.config.max_size {
        return Some(term);  // Success!
    }
}
None  // Failed after 100 attempts
```

With `max_depth=12` and `max_size=200`:
- Most generated terms are 500-2000 nodes
- Only ~5-10% are under 200 nodes
- Generator wastes 90-95% of attempts
- Successfully generated terms are ~180 nodes, which then grow to 900+

## Solution

### Balanced Parameters

The key insight: **Initial term size × Growth rate = Final size**

Target final size: ~300-600 nodes (reasonable for training)
Expected growth rate: ~5x
Therefore: Initial size should be ~60-120 nodes

For depth D, average term size ≈ 2^(D-2) with random generation:
- Depth 6: ~16 nodes
- Depth 7: ~32 nodes
- **Depth 8: ~64 nodes** ✅
- Depth 9: ~128 nodes
- Depth 10: ~256 nodes

### New Configuration

```rust
GeneratorConfig {
    max_depth: 8,      // Down from 12: avg ~64 nodes initial
    min_depth: 3,      // Keep: avoid trivial terms
    max_size: 100,     // Down from 200: hard cap on initial size
    allow_divergent: true,
}
```

**Expected results:**
- Initial term size: 50-100 nodes
- After 5x growth: 250-500 nodes ✅
- Reduction time: 50-150ms ✅
- Divergence rate: 10-20% ✅
- Pathological rate: 20-30% ✅

### Wall Clock Limit Recommendations

| Initial Size | Expected Growth | Final Size | Recommended Wall Clock |
|--------------|-----------------|------------|------------------------|
| 50-100       | 5x              | 250-500    | 200-300ms              |
| 100-150      | 5x              | 500-750    | 300-500ms              |
| 150-200      | 5x              | 750-1000   | 500-1000ms             |

For the new defaults (max_size=100), use:
```bash
cargo run --release -- generate data.jsonl 1000000 16 250
#                                               max_size↑  wall_clock↑
```

## Prevention

### Configuration Guidelines

**Rule of Thumb:**
```
max_size = target_final_size / expected_growth_rate
max_depth ≈ log2(max_size) + 2
wall_clock_ms = max_size * 2-3 ms

For max_size=100:
- max_depth = log2(100) + 2 ≈ 8-9
- wall_clock = 200-300ms
```

### Validation Checklist

After generating data, run diagnostics and verify:

✅ **Term size:**
- Mean: 200-600 nodes (final)
- Max: <2000 nodes
- Initial mean: 50-150 nodes

✅ **Growth rate:**
- Mean: 3-8x
- Max: <30x
- Terms that grow >30x are pathological

✅ **Divergence rate:**
- Target: 10-20%
- Acceptable: up to 30%
- >30% means wall_clock_ms is too low

✅ **Pathological rate:**
- Target: 15-30%
- Acceptable: up to 40%
- >40% means terms are too complex

✅ **Steps per term:**
- Mean: 10-50 steps
- Median: 5-20 steps
- Terms with >500 steps are probably pathological

### Per-Model Configuration

| Model Size | Target Complexity | max_depth | max_size | wall_clock_ms | Examples |
|------------|-------------------|-----------|----------|---------------|----------|
| Tiny (38M) | Simple            | 6-7       | 50-80    | 100-150       | 500k     |
| Small (75M)| Moderate          | 7-8       | 80-120   | 200-300       | 1-2M     |
| Medium (150M)| Complex         | 8-9       | 120-150  | 300-500       | 2-5M     |
| Large (700M)| Very Complex     | 9-10      | 150-200  | 500-1000      | 5-10M    |

**Never exceed max_depth=10** without careful profiling!

## Testing the Fix

### Step 1: Rebuild

```bash
cd X:\Code\LM\lambda_gen_rs
cargo build --release --target x86_64-pc-windows-gnu
```

### Step 2: Generate Test Data

```bash
# Small test set
cargo run --release --target x86_64-pc-windows-gnu -- generate test_data.jsonl 10000 16 250
```

### Step 3: Diagnose

```bash
cd X:\Code\LM
python tests/diagnose_training_data.py test_data.jsonl 5000
```

**Expected output:**
```
Term Size:
  Mean: 200-500 nodes     ✅
  Max: <2000 nodes        ✅

Size Growth Rate:
  Mean: 4-6x              ✅
  Max: <30x               ✅

Diverged Terms:
  10-20%                  ✅

Pathological Cases:
  20-35%                  ✅

Steps per Term:
  Mean: 15-40             ✅
  Median: 8-20            ✅

premature_nf_marker: <5% ✅
```

### Step 4: Deep Diagnosis (If Issues Persist)

```bash
python tests/deep_diagnose.py test_data.jsonl 1000
```

This will show:
- Exact traces with premature NF markers
- Which steps have incorrect target_spans
- Pattern analysis of problematic traces

### Step 5: Generate Production Data

Once validated:
```bash
# Full dataset with proper parameters
cargo run --release --target x86_64-pc-windows-gnu -- generate training_data.jsonl 1000000 16 250
```

## Technical Details

### Why Terms Explode

**Beta reduction can duplicate subterms:**

```
(\x.x x) BIG_TERM
→ BIG_TERM BIG_TERM    [size doubled!]
```

**Church numerals example:**
```
SUCC = \n.\f.\x.f(n f x)
TWO = \f.\x.f(f x)

SUCC TWO
= (\n.\f.\x.f(n f x))(\f.\x.f(f x))
→ \f.\x.f((\f.\x.f(f x)) f x)
→ \f.\x.f(f(f x))     [THREE - size grew]
```

**Worst case: Exponential explosion**

Terms like `(\x.x x x)(\x.x x x)` can explode exponentially:
```
Step 0: Size 6
Step 1: Size 18
Step 2: Size 54
Step 3: Size 162
Step 4: Size 486
...
```

### Why 45% Divergence Rate is Bad

High divergence means:
1. **Training sees incomplete patterns**: Model doesn't learn full reduction
2. **Biased data distribution**: Only "easy" terms complete, "hard" terms diverge
3. **Incorrect labels**: Diverged terms may have wrong target_spans
4. **Wasted computation**: 45% of generation time produces low-quality examples

Target: <20% divergence (only truly pathological terms)

### Premature NF Marker Issue

**Definition**: `target_span = (0,0)` at non-final steps

**Causes:**
1. Buggy find_redex returning None incorrectly
2. Diverged terms having redex_path=None in final step
3. Generator creating already-normal-form terms
4. Diagnostic script bug (less likely)

**Impact:** Model learns to predict (0,0) too early, stopping reduction prematurely

## Lessons Learned

1. **Depth and size must be balanced**: Can't just increase one
2. **Beta reduction causes term explosion**: Plan for 3-8x growth
3. **Wall clock limit must match complexity**: Bigger terms need more time
4. **Test with diagnostics immediately**: Don't generate millions of bad examples
5. **Monitor growth rates**: If mean >8x, parameters are wrong

## Summary

**What went wrong (round 2):**
- `max_depth=12` too deep → huge initial terms
- `max_size=200` too large → even huger terms
- Terms exploded 5.86x during reduction → 887 node average
- 45% couldn't complete in 100ms → diverged
- 55% flagged as pathological → bad training data

**What was fixed:**
- `max_depth`: 12 → 8 (more reasonable)
- `max_size`: 200 → 100 (allows growth headroom)
- Recommended `wall_clock`: 100ms → 250ms

**Expected results:**
- Initial size: 50-100 nodes
- Final size: 250-500 nodes (after 5x growth)
- Divergence: 10-20%
- Pathological: 20-30%
- **Good training data!** ✅

---

**Status**: Fixed in commit [hash]
**Previous Issue**: DATA_QUALITY_ISSUE_RESOLUTION.md (first fix)
**Related**: Generator defaults were too aggressive after first fix
