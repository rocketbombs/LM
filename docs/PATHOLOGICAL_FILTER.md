# Pathological Term Filter

## Problem

Even with `allow_divergent=false` (only normalizing terms) and premature NF validation, **~25% of traces were flagged as pathological**.

**Pathological**: Terms that normalize but exhibit problematic behavior:
- Extreme size growth (>3x)
- Very slow steps (>5ms average)
- Consume >80% of time budget
- Grow very large (>200 nodes)

### Impact on Training

Training on pathological reduction traces teaches the model:
- Inefficient reduction strategies
- Tolerance for exponential blowup
- Slow, wasteful computation patterns

**User requirement**: "ALL terms should be reducible or otherwise NF" - meaning CLEAN, efficient reduction traces.

## Root Cause

The pathological detection criteria are appropriate:

```rust
pub fn detect_pathological(
    time_consumed_ratio: f64,   // >0.8 = used >80% of budget
    avg_step_ms: f64,            // >5.0 = slow steps
    size_growth_rate: f64,       // >3.0 = tripled in size
    current_size: usize,         // >200 = very large
) -> bool {
    time_consumed_ratio > 0.8
        || avg_step_ms > 5.0
        || size_growth_rate > 3.0
        || current_size > 200
}
```

With `max_size=100`, terms can grow to 300+ nodes during reduction. This is:
- **Technically valid** (term still normalizes)
- **Practically problematic** (exponential blowup, slow reduction)

Examples that trigger this:
- Church numerals with large values (exponential expansion)
- Deeply nested applications that duplicate large subterms
- Terms that "almost diverge" but eventually normalize

**~25% of normalizing terms exhibit this behavior** - too many for clean training data.

## Solution: Filter Pathological Traces

Added validation in `parallel.rs` after checking divergence, **before** generating examples:

```rust
// Compute step times for averaging
let step_times: Vec<f64> = trace.steps.iter().map(|s| s.step_time_ms).collect();
let avg_step_ms = if !step_times.is_empty() {
    step_times.iter().sum::<f64>() / step_times.len() as f64
} else {
    0.0
};

// Check if trace is pathological (skip entire trace if so)
let initial_size = trace.steps[0].term.size();
let final_size = trace.steps.last().map(|s| s.term.size()).unwrap_or(initial_size);
let size_growth_rate = if initial_size > 0 {
    final_size as f64 / initial_size as f64
} else {
    1.0
};
let time_consumed_ratio = trace.total_time_ms / config.reduction_config.wall_clock_limit_ms;

let is_trace_pathological = ExampleMetadata::detect_pathological(
    time_consumed_ratio,
    avg_step_ms,
    size_growth_rate,
    final_size,
);

// VALIDATION: Skip pathological traces
if is_trace_pathological {
    // This trace exhibits pathological behavior
    // Skip to maintain clean training data
    continue;
}
```

### Logic

1. After reducing a term, compute trace-level metrics:
   - `avg_step_ms`: Average time per step
   - `size_growth_rate`: final_size / initial_size
   - `time_consumed_ratio`: total_time / wall_clock_limit
   - `final_size`: Size of term in normal form

2. Check if trace is pathological using existing criteria

3. **If pathological**: Skip entire trace (don't generate ANY examples from it)

4. **If clean**: Generate examples for all steps in trace

### Effect

- Pathological traces are **completely filtered out** during generation
- Only clean, efficient reduction traces reach training data
- **Pathological rate: ~25% → 0%** ✅

## Trade-offs

**Pros:**
- ✅ Guarantees 0% pathological examples
- ✅ Model learns only efficient reduction patterns
- ✅ Clean training data for baseline model
- ✅ Filters ~25% of normalizing terms that are problematic

**Cons:**
- ❌ Reduces total example count by ~25%
- ❌ Model won't see pathological behavior (can add later for robustness)
- ❌ Filters some technically valid reductions

**Justification**:
- For **baseline training**, clean data is essential
- Can generate **separate dataset** with pathological terms for advanced training
- Quality >> quantity for initial model development

## Expected Results

### Before Pathological Filter

```
Normalizing terms: 100%        ✅ (after allow_divergent=false)
Premature NF: 0%               ✅ (after validation)
Pathological: ~25%             ❌ Too many!
Clean, efficient traces: ~75%
```

### After Pathological Filter

```
Normalizing terms: 100%        ✅
Premature NF: 0%               ✅
Pathological: 0%               ✅ PERFECT!
Clean, efficient traces: 100%  ✅
```

## Performance Impact

**Generation throughput:**

With all filters active:
1. Generate term
2. Reduce (filters ~5-10% divergent)
3. Check pathological (filters ~25% of remaining)
4. Check premature NF per-example (filters ~16% of examples)

**Net effective rate:**
- Start: 100% generated
- After divergent filter: ~90-95%
- After pathological filter: ~70%
- After premature NF filter: ~59%

**Result**: ~60% of generated terms produce usable training data

**Trade-off**: Slower generation, but **100% of output is high quality**

## Testing

### Verify 0% Pathological

```powershell
# Rebuild
cd X:\Code\LM\lambda_gen_rs
cargo build --release --target x86_64-pc-windows-gnu

# Generate test data
cargo run --release --target x86_64-pc-windows-gnu -- generate test_data.jsonl 10000 16 250

# Diagnose
cd X:\Code\LM
python tests/diagnose_training_data.py test_data.jsonl 5000
```

**Expected output:**
```
⚠️  Pathological Cases:
  0 / 5000 (0.0%)               ✅ PERFECT!

🔍 DATA QUALITY ISSUES
--------------------------------------------------------------------------------
Issues found:
  (empty or only trivial issues)  ✅
```

### Verify All Examples Valid

```powershell
python tests/deep_diagnose.py test_data.jsonl 1000
```

Expected:
```
🔍 PREMATURE NF MARKERS: 0      ✅
🔍 DIVERGED WITH NF: 0          ✅
Traces with premature NF: 0     ✅
```

Plus checking manually that no examples have:
- `is_pathological: true` in metadata
- Extreme size_growth_rate
- Very high avg_step_ms

## Configuration Summary

**Final "clean data" configuration:**

```rust
GeneratorConfig {
    max_depth: 8,
    min_depth: 3,
    max_size: 100,
    allow_divergent: false,  // Filter non-normalizing
}

ReductionConfig {
    wall_clock_limit_ms: 250.0,
    max_steps: 500,
}

// Plus three validation filters:
// 1. Diverged terms (if allow_divergent=false)
// 2. Pathological traces (THIS FIX)
// 3. Premature NF markers (per-example)
```

**Pathological detection criteria:**
```rust
time_consumed_ratio > 0.8      // Used >80% of 250ms budget
|| avg_step_ms > 5.0           // Slow steps
|| size_growth_rate > 3.0      // Tripled from initial 100 nodes
|| current_size > 200          // Final size >200 nodes
```

These thresholds are appropriate for **clean baseline training**.

## Advanced Training (Future)

For robust models that handle difficult cases:

**Phase 1**: Train on clean data (THIS CONFIG)
- 0% pathological
- Learn efficient reduction patterns
- Baseline model

**Phase 2**: Add pathological data
- Set `allow_pathological=true` (new flag)
- Include 10-20% pathological traces
- Learn to handle difficult cases
- More robust model

**Phase 3**: Add divergent data (research)
- Set `allow_divergent=true`
- Include non-normalizing terms
- Learn early termination
- Research-grade model

## Summary

**Problem**: ~25% of normalizing terms were pathological (slow, exponential growth)

**Solution**: Filter out pathological traces before generating examples

**Result**:
- Pathological: 25% → 0% ✅
- All training data is clean, efficient reduction traces ✅
- Model learns optimal reduction patterns ✅

**Trade-off**: ~40% total filtering rate (divergent + pathological + premature NF), but **100% of output is high quality**

---

**Status**: Fixed and ready for testing
**Priority**: P1 - Required for clean baseline training
**Impact**: High - Ensures model learns efficient patterns
**Complexity**: Simple (compute metrics, check threshold, skip if bad)
