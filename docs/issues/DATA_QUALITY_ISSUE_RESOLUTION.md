# Data Quality Issue Resolution

## Problem Report

**Symptoms:**
- Training shows ~80% pathological cases
- Extremely high token throughput (160k tokens/second)
- Model training appears "too fast" and may not be learning meaningful patterns

## Root Cause Analysis

### The Issue

The Rust generator had **hardcoded defaults that were far too small**:

```rust
// OLD (BROKEN) DEFAULTS
GeneratorConfig {
    max_depth: 6,      // ❌ TOO SMALL - generates trivial terms
    min_depth: 2,
    max_size: 50,      // ❌ TOO SMALL
    allow_divergent: true,
}
```

### Why This Caused Problems

1. **High Pathological Rate (80%)**:
   - With `max_depth=6` and `max_size=50`, the generator creates TINY terms
   - These terms typically reduce in 1-2 steps
   - Most hit the pathological detection thresholds:
     - `time_consumed_ratio > 0.8` (even trivial ops can hit 80%)
     - `avg_step_ms > 5.0` (random variation can trigger this)
     - `size_growth_rate > 3.0` (small terms can triple easily)
     - `current_size > 200` (never triggered with max_size=50)

2. **Extremely High Token Throughput (160k tok/s)**:
   - Tiny terms = few tokens per example
   - Few reduction steps = few examples per term
   - Result: Training processes examples very quickly
   - **This is NOT a performance improvement** - it's a data quality problem!

3. **Poor Training Quality**:
   - Model sees only trivial reduction patterns
   - No complex nested structures
   - No realistic term sizes
   - Won't generalize to real lambda calculus problems

### Example of Bad Data

With old defaults, a typical generated term:
```
(\.(0 0))(\.(1 0))
```
- Length: 18 characters
- Depth: 3
- Steps: 1-2
- **Not representative of real lambda calculus problems!**

## Solution

### Code Fixes Applied

**File: `lambda_gen_rs/src/generator.rs`**
```rust
// NEW (FIXED) DEFAULTS
impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig {
            max_depth: 12,     // 2x increase for complex terms
            min_depth: 3,      // Avoid trivial terms
            max_size: 200,     // 4x increase for substantial terms
            allow_divergent: true,
        }
    }
}
```

**File: `lambda_gen_rs/src/main.rs`**
- Updated generate command to use better defaults
- Matches the default config for consistency

### Expected Results with Fixed Config

With `max_depth=12` and `max_size=200`:

- **Term length**: 50-150 characters (typical)
- **Term depth**: 4-12 levels
- **Reduction steps**: 3-20 steps (more variety)
- **Pathological rate**: 15-30% (healthy, not overwhelming)
- **Token throughput**: 20-40k tok/s (more realistic with complex data)

## Action Required

### Step 1: Verify Current Data Quality

Run the diagnostic script to confirm the issue:

```powershell
cd X:\Code\LM
python tests/diagnose_training_data.py data.jsonl 10000
```

**Expected diagnostic output:**
```
Term Length:
  Mean: ~15-25 chars   ❌ TOO SHORT

Term Size:
  Mean: ~10-20 nodes   ❌ TOO SMALL

Steps per Term:
  Mean: ~1-2 steps     ❌ TOO FEW

Pathological Cases:
  ~75-85%             ❌ TOO HIGH

ROOT CAUSE: max_depth=6, max_size=50 too restrictive
```

### Step 2: Rebuild Rust Generator

```powershell
cd X:\Code\LM\lambda_gen_rs
cargo build --release --target x86_64-pc-windows-gnu
```

### Step 3: Regenerate Training Data

**Option A: Default (Recommended)**
```powershell
# Use new defaults (max_depth=12, max_size=200, wall_clock=100ms)
cargo run --release --target x86_64-pc-windows-gnu -- generate training_data.jsonl 1000000 16 100
```

**Option B: More Complex (Advanced)**
```powershell
# Increase wall clock limit for even more complex terms
cargo run --release --target x86_64-pc-windows-gnu -- generate training_data.jsonl 1000000 16 500
```

**Option C: Curriculum Learning (Best)**

Generate multiple datasets with increasing complexity:

```powershell
# Stage 1: Medium complexity
cargo run --release --target x86_64-pc-windows-gnu -- generate data_medium.jsonl 250000 16 200

# Stage 2: High complexity
cargo run --release --target x86_64-pc-windows-gnu -- generate data_complex.jsonl 250000 16 500

# Stage 3: Very high complexity
cargo run --release --target x86_64-pc-windows-gnu -- generate data_advanced.jsonl 250000 16 1000

# Combine
cat data_medium.jsonl data_complex.jsonl data_advanced.jsonl > training_data.jsonl
```

### Step 4: Verify New Data Quality

```powershell
python tests/diagnose_training_data.py training_data.jsonl 10000
```

**Expected diagnostic output (GOOD):**
```
Term Length:
  Mean: ~60-100 chars   ✅ GOOD

Term Size:
  Mean: ~50-120 nodes   ✅ GOOD

Steps per Term:
  Mean: ~5-15 steps     ✅ GOOD

Pathological Cases:
  ~20-35%              ✅ HEALTHY

✅ Data quality looks good!
```

### Step 5: Retrain Model

```powershell
python lambda_train.py \
    --train-data training_data.jsonl \
    --output-dir runs/levy_75m_fixed \
    --d-model 768 \
    --n-layers 8 \
    --batch-tokens 16384 \
    --steps 100000
```

**Expected training behavior:**
- Token throughput: 20-40k tok/s (slower is GOOD with complex data)
- Pathological case rate during training: ~20-30%
- Loss should decrease steadily
- Model should learn meaningful reduction patterns

## Prevention

### For Future Data Generation

Always specify appropriate parameters based on your training goals:

**For Small/Fast Models (75M params):**
```powershell
cargo run --release -- generate data.jsonl 1000000 16 200
# max_depth=12 (default), max_size=200 (default), wall_clock=200ms
```

**For Large/Complex Models (700M params):**
```powershell
cargo run --release -- generate data.jsonl 5000000 16 500
# max_depth=12 (default), max_size=200 (default), wall_clock=500ms
```

**For Curriculum Learning:**
```powershell
# Start simple, increase complexity
for depth in 8 10 12 15; do
    cargo run --release -- generate data_depth${depth}.jsonl 250000 16 $((depth * 50))
done
```

### Configuration Guidelines

| Model Size | max_depth | max_size | wall_clock_ms | Examples |
|------------|-----------|----------|---------------|----------|
| Tiny (38M) | 8-10      | 100-150  | 100-200      | 500k     |
| Small (75M)| 10-12     | 150-200  | 200-300      | 1-2M     |
| Medium (150M)| 12-15   | 200-250  | 300-500      | 2-5M     |
| Large (700M)| 15-20    | 250-300  | 500-1000     | 5-10M    |

### Monitoring Data Quality

Always run the diagnostic script on generated data:

```powershell
python tests/diagnose_training_data.py <data_file> 10000
```

**Red flags:**
- Mean term length < 30 chars ❌
- Mean steps < 3 ❌
- Pathological rate > 60% ❌
- Mean size < 30 nodes ❌

**Good indicators:**
- Mean term length: 50-150 chars ✅
- Mean steps: 5-20 ✅
- Pathological rate: 15-35% ✅
- Mean size: 40-150 nodes ✅

## Summary

### What Went Wrong

1. Default `max_depth=6` and `max_size=50` were **far too small**
2. Generated trivial terms with 1-2 reduction steps
3. 80% flagged as pathological (not representative)
4. High token throughput was a **warning sign**, not a feature

### What Was Fixed

1. Updated defaults: `max_depth=12`, `max_size=200`
2. These create realistic, complex terms
3. Expected pathological rate: 20-30%
4. Token throughput will be lower (this is GOOD)

### Next Steps

1. ✅ Pull updated code from git
2. ✅ Rebuild Rust generator
3. ✅ Regenerate training data with new defaults
4. ✅ Verify data quality with diagnostic script
5. ✅ Retrain model

---

**Status**: Fixed in commit [hash]
**Date**: 2024-10-29
**Impact**: Critical - requires data regeneration
